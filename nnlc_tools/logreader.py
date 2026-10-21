#!/usr/bin/env python3
"""Minimal standalone LogReader for reading openpilot rlog files.

Replaces the dependency on openpilot's LogReader by bundling the cereal
capnp schemas in nnlc_tools/cereal/ and providing a simple iterator interface.
"""

import bz2
import os
import shutil
import sys
import tempfile

import capnp
import zstandard as zstd

CEREAL_DIR = os.path.join(os.path.dirname(__file__), "cereal")
capnp_log = capnp.load(os.path.join(CEREAL_DIR, "log.capnp"), imports=[CEREAL_DIR])

ZSTD_MAGIC = b"\x28\xB5\x2F\xFD"
BZ2_MAGIC = b"BZ"
COPY_CHUNK_BYTES = 1024 * 1024
# pycapnp's file reader needs a real fileno(), so large payloads still spill
# to an anonymous temporary file. Smaller compressed rlogs stay in memory.
IN_MEMORY_MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
IN_MEMORY_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


def _runtime_directory():
    """Return the directory containing the executable or source project."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _temporary_file():
    """Open a temporary file beside the program, falling back to the OS temp dir.

    ``NNLC_TEMP_DIR`` can override the preferred location.  The fallback keeps
    installations under read-only directories (for example ``Program Files``)
    usable.
    """
    preferred_dir = os.environ.get("NNLC_TEMP_DIR") or _runtime_directory()
    try:
        os.makedirs(preferred_dir, exist_ok=True)
        return tempfile.TemporaryFile(dir=preferred_dir)
    except (OSError, ValueError):
        return tempfile.TemporaryFile()


def _is_truncated_capnp_stream(exc):
    """Return True when pycapnp hit EOF or a truncated trailing message."""
    if not isinstance(exc, capnp.KjException):
        return False
    message = str(exc)
    return "EOF" in message or "Message ends prematurely" in message


def _decode_compressed_stream(reader, prefer_memory):
    """Decompress into memory when small enough, otherwise a seekable temp file.

    Returns ``(buffer, temp_file)``. Exactly one of the two is not None.
    """
    if not prefer_memory:
        target = _temporary_file()
        shutil.copyfileobj(reader, target, length=COPY_CHUNK_BYTES)
        return None, target

    chunks = []
    total = 0
    while True:
        chunk = reader.read(COPY_CHUNK_BYTES)
        if not chunk:
            break
        next_total = total + len(chunk)
        if next_total > IN_MEMORY_MAX_UNCOMPRESSED_BYTES:
            target = _temporary_file()
            for previous in chunks:
                target.write(previous)
            target.write(chunk)
            shutil.copyfileobj(reader, target, length=COPY_CHUNK_BYTES)
            return None, target
        chunks.append(chunk)
        total = next_total
    return b"".join(chunks), None


class LogReader:
    """Read and iterate over messages in an rlog file.

    ``sort_by_time`` is retained for compatibility with the former API. When
    enabled, all events from this single rlog are read and stably sorted by
    ``logMonoTime`` before iteration. This keeps event ordering deterministic
    without imposing an arbitrary reorder window or dropping late events.
    ``check_time_order`` is kept as a compatibility option for older callers;
    when enabled without sorting, it raises if the source order is invalid.
    """

    def __init__(self, fn, sort_by_time=False, check_time_order=None):
        self._file = None
        self._buffer = None
        self._temporary_file = False
        self._filename = fn
        if not isinstance(sort_by_time, bool):
            raise ValueError("sort_by_time must be a boolean")
        if check_time_order is not None and not isinstance(check_time_order, bool):
            raise ValueError("check_time_order must be a boolean or None")
        self._sort_by_time = sort_by_time
        self._check_time_order = bool(check_time_order)
        source = None
        target = None

        try:
            source = open(fn, "rb")
            magic = source.read(4)
            source.seek(0)

            if magic == ZSTD_MAGIC or magic[:2] == BZ2_MAGIC:
                # Small compressed rlogs decompress in RAM and use
                # read_multiple_bytes(). Larger ones still need a real
                # fileno() for pycapnp's file reader.
                prefer_memory = (
                    os.path.getsize(fn) <= IN_MEMORY_MAX_COMPRESSED_BYTES
                )
                if magic == ZSTD_MAGIC:
                    reader_cm = zstd.ZstdDecompressor().stream_reader(source)
                else:
                    reader_cm = bz2.BZ2File(source, "rb")
                with reader_cm as reader:
                    buffer, target = _decode_compressed_stream(
                        reader, prefer_memory,
                    )
                source.close()
                source = None
                if buffer is not None:
                    if not buffer:
                        raise ValueError(f"rlog file is empty: {fn}")
                    self._buffer = buffer
                    return
                self._temporary_file = True
            else:
                target = source
                source = None

            # pycapnp's read_multiple(file) consumes this file one message
            # at a time, so the uncompressed payload stays on disk instead
            # of becoming a second full-size Python bytes object.
            target.flush()
            size = target.seek(0, os.SEEK_END)
            if size == 0:
                raise ValueError(f"rlog file is empty: {fn}")
            target.seek(0)
            self._file = target
            target = None
        except Exception:
            # Decompression or stream setup may fail before ``source`` is
            # transferred to ``self._file``.  Close both handles explicitly;
            # otherwise repeated corrupt rlogs can exhaust file descriptors.
            for handle in (target, source):
                if handle is not None and handle is not self._file:
                    try:
                        handle.close()
                    except OSError:
                        pass
            self.close()
            raise

    def iter_events(self, *, skip_copy=False):
        """Iterate source-order events, optionally borrowing Cap'n Proto data.

        When ``skip_copy`` is true, callers must copy every value they need
        before requesting the next event. This is intended for streaming
        consumers that immediately convert selected fields to Python values.
        In-memory compressed logs already borrow from a pinned buffer, so
        ``skip_copy`` does not create extra copies there.
        """
        if not isinstance(skip_copy, bool):
            raise ValueError("skip_copy must be a boolean")
        if self._buffer is None and self._file is None:
            return

        if self._buffer is not None:
            events = iter(capnp_log.Event.read_multiple_bytes(self._buffer))
        else:
            # read_multiple requires a real file object and advances its
            # cursor as each framed Cap'n Proto message is yielded.
            self._file.seek(0)
            events = iter(
                capnp_log.Event.read_multiple(self._file, skip_copy=skip_copy)
            )

        yielded_events = False
        previous_time = None
        while True:
            try:
                event = next(events)
            except StopIteration:
                break
            except capnp.KjException as exc:
                # Some real rlogs have a truncated Cap'n Proto trailer.
                # The file reader treats that as EOF; the bytes reader
                # raises "Message ends prematurely" after the last valid
                # event. Swallow it only after at least one event so a
                # corrupt empty payload still surfaces.
                if yielded_events and _is_truncated_capnp_stream(exc):
                    break
                raise
            yielded_events = True
            if self._check_time_order:
                current_time = event.logMonoTime
                if previous_time is not None and current_time < previous_time:
                    raise ValueError(
                        f"rlog events are not ordered by logMonoTime: "
                        f"{current_time} < {previous_time} ({self._filename})"
                    )
                previous_time = current_time
            yield event
            # Borrowed readers must be released before pycapnp advances the
            # underlying file buffer for the next message.
            del event
        if not yielded_events:
            raise ValueError(f"rlog contains no events: {self._filename}")

    def __iter__(self):
        events = self.iter_events(skip_copy=False)
        if self._sort_by_time:
            # Stable sorting preserves source order for equal timestamps.
            buffered_events = list(events)
            buffered_events.sort(key=lambda event: event.logMonoTime)
            yield from buffered_events
            return
        yield from events

    def close(self):
        """Release the source and any temporary decompression file."""
        self._buffer = None
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        self.close()
