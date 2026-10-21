from types import SimpleNamespace

import capnp
import pytest
import zstandard as zstd

from nnlc_tools import logreader


def test_iter_events_passes_skip_copy_and_preserves_source_order(monkeypatch, tmp_path):
    calls = []
    events = [SimpleNamespace(logMonoTime=3), SimpleNamespace(logMonoTime=1)]

    def read_multiple(file_handle, *, skip_copy=False):
        calls.append((file_handle.tell(), skip_copy))
        return iter(events)

    fake_capnp = SimpleNamespace(
        Event=SimpleNamespace(read_multiple=read_multiple),
    )
    monkeypatch.setattr(logreader, "capnp_log", fake_capnp)
    source = tmp_path / "rlog"
    source.write_bytes(b"framed log placeholder")
    reader = logreader.LogReader(source)
    try:
        assert reader._buffer is None
        assert [event.logMonoTime for event in reader.iter_events(skip_copy=True)] == [3, 1]
        assert calls == [(0, True)]
    finally:
        reader.close()


def test_sorted_iteration_remains_stable_and_copies_messages(monkeypatch, tmp_path):
    events = [
        SimpleNamespace(logMonoTime=2, source_order=0),
        SimpleNamespace(logMonoTime=1, source_order=1),
        SimpleNamespace(logMonoTime=1, source_order=2),
    ]
    calls = []

    def read_multiple(file_handle, *, skip_copy=False):
        calls.append(skip_copy)
        return iter(events)

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(read_multiple=read_multiple)),
    )
    source = tmp_path / "rlog"
    source.write_bytes(b"framed log placeholder")
    reader = logreader.LogReader(source, sort_by_time=True)
    try:
        assert [(event.logMonoTime, event.source_order) for event in reader] == [
            (1, 1), (1, 2), (2, 0),
        ]
        assert calls == [False]
    finally:
        reader.close()


def _write_zstd(path, payload):
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


def test_small_zstd_uses_in_memory_bytes_reader(monkeypatch, tmp_path):
    bytes_calls = []
    file_calls = []
    events = [SimpleNamespace(logMonoTime=7), SimpleNamespace(logMonoTime=8)]

    def read_multiple_bytes(buf):
        bytes_calls.append(bytes(buf))
        return iter(events)

    def read_multiple(file_handle, *, skip_copy=False):
        file_calls.append(skip_copy)
        raise AssertionError("small zstd logs should not use the file reader")

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple=read_multiple,
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    payload = b"framed log placeholder"
    _write_zstd(source, payload)
    reader = logreader.LogReader(source)
    try:
        assert reader._buffer == payload
        assert reader._file is None
        assert [event.logMonoTime for event in reader.iter_events(skip_copy=True)] == [7, 8]
        assert bytes_calls == [payload]
        assert file_calls == []
    finally:
        reader.close()


def test_oversize_zstd_still_uses_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(logreader, "IN_MEMORY_MAX_COMPRESSED_BYTES", 1)
    file_calls = []
    events = [SimpleNamespace(logMonoTime=4)]

    def read_multiple(file_handle, *, skip_copy=False):
        file_calls.append((file_handle.tell(), skip_copy, file_handle.fileno() >= 0))
        return iter(events)

    def read_multiple_bytes(buf):
        raise AssertionError("oversize zstd logs should not use the bytes reader")

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple=read_multiple,
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    _write_zstd(source, b"framed log placeholder that compresses above one byte")
    reader = logreader.LogReader(source)
    try:
        assert reader._buffer is None
        assert reader._file is not None
        assert [event.logMonoTime for event in reader.iter_events(skip_copy=True)] == [4]
        assert file_calls == [(0, True, True)]
    finally:
        reader.close()


def test_uncompressed_overflow_spills_to_temporary_file(monkeypatch, tmp_path):
    monkeypatch.setattr(logreader, "IN_MEMORY_MAX_UNCOMPRESSED_BYTES", 8)
    file_calls = []
    events = [SimpleNamespace(logMonoTime=9)]

    def read_multiple(file_handle, *, skip_copy=False):
        file_calls.append(skip_copy)
        return iter(events)

    def read_multiple_bytes(buf):
        raise AssertionError("oversize uncompressed payload should spill")

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple=read_multiple,
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    _write_zstd(source, b"0123456789abcdef")
    reader = logreader.LogReader(source)
    try:
        assert reader._buffer is None
        assert reader._temporary_file is True
        assert [event.logMonoTime for event in reader.iter_events(skip_copy=False)] == [9]
        assert file_calls == [False]
    finally:
        reader.close()


def test_truncated_capnp_trailer_stops_after_last_valid_event(monkeypatch, tmp_path):
    events = [SimpleNamespace(logMonoTime=1), SimpleNamespace(logMonoTime=2)]

    def read_multiple_bytes(buf):
        def iterator():
            yield events[0]
            yield events[1]
            raise capnp.KjException(
                "capnp/serialize.c++:59: failed: expected array.size() >= "
                "offset + segmentSize [16 >= 38]; Message ends prematurely "
                "in first segment."
            )
        return iterator()

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    _write_zstd(source, b"framed log placeholder")
    reader = logreader.LogReader(source)
    try:
        assert [event.logMonoTime for event in reader.iter_events(skip_copy=True)] == [1, 2]
    finally:
        reader.close()


def test_truncated_capnp_without_events_still_raises(monkeypatch, tmp_path):
    def read_multiple_bytes(buf):
        def iterator():
            raise capnp.KjException("Message ends prematurely in first segment.")
            yield  # pragma: no cover
        return iterator()

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    _write_zstd(source, b"framed log placeholder")
    reader = logreader.LogReader(source)
    try:
        with pytest.raises(capnp.KjException, match="Message ends prematurely"):
            list(reader.iter_events(skip_copy=True))
    finally:
        reader.close()


def test_in_memory_iteration_can_be_replayed_for_exact_sort_fallback(monkeypatch, tmp_path):
    events = [SimpleNamespace(logMonoTime=3), SimpleNamespace(logMonoTime=1)]

    def read_multiple_bytes(buf):
        return iter(list(events))

    monkeypatch.setattr(
        logreader,
        "capnp_log",
        SimpleNamespace(Event=SimpleNamespace(
            read_multiple_bytes=read_multiple_bytes,
        )),
    )
    source = tmp_path / "rlog.zst"
    _write_zstd(source, b"framed log placeholder")
    reader = logreader.LogReader(source)
    try:
        first = [event.logMonoTime for event in reader.iter_events(skip_copy=True)]
        second = [event.logMonoTime for event in reader.iter_events(skip_copy=False)]
        assert first == [3, 1]
        assert second == [3, 1]
    finally:
        reader.close()
