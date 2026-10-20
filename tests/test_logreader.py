from types import SimpleNamespace

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
