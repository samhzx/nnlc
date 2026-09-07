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

            # Decompress directly into a temporary file.  pycapnp's
            # read_multiple(file) consumes this file one message at a time,
            # so the uncompressed payload stays on disk instead of becoming a
            # second full-size Python bytes object.
            if magic == b"\x28\xB5\x2F\xFD":  # zstd magic
                target = _temporary_file()
                with zstd.ZstdDecompressor().stream_reader(source) as reader:
                    shutil.copyfileobj(reader, target, length=1024 * 1024)
                source.close()
                self._temporary_file = True
            elif magic[:2] == b"BZ":
                target = _temporary_file()
                with bz2.BZ2File(source, "rb") as reader:
                    shutil.copyfileobj(reader, target, length=1024 * 1024)
                source.close()
                self._temporary_file = True
            else:
                target = source

            target.flush()
            size = target.seek(0, os.SEEK_END)
            if size == 0:
                raise ValueError(f"rlog file is empty: {fn}")
            target.seek(0)
            self._file = target
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

    def __iter__(self):
        if self._file is None:
            return

        # read_multiple requires a real file object and advances its cursor as
        # each framed Cap'n Proto message is yielded.  Readers are copied by
        # pycapnp's default ``skip_copy=False`` behavior, which is important
        # because extract_segment retains references to the latest state
        # messages while reading the next event.
        self._file.seek(0)
        yielded_events = False
        events = capnp_log.Event.read_multiple(self._file, skip_copy=False)
        if self._sort_by_time:
            # The caller explicitly accepts per-rlog memory usage. Sorting is
            # stable, so events with equal timestamps retain their file order.
            buffered_events = list(events)
            buffered_events.sort(key=lambda event: event.logMonoTime)
            events = iter(buffered_events)
        previous_time = None
        for event in events:
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
        if not yielded_events:
            raise ValueError(f"rlog contains no events: {self._filename}")

    def close(self):
        """Release the source and any temporary decompression file."""
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
