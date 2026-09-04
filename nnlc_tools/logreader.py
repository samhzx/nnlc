#!/usr/bin/env python3
"""Minimal standalone LogReader for reading openpilot rlog files.

Replaces the dependency on openpilot's LogReader by bundling the cereal
capnp schemas in nnlc_tools/cereal/ and providing a simple iterator interface.
"""

import bz2
import mmap
import os
import shutil
import tempfile

import capnp
import zstandard as zstd

CEREAL_DIR = os.path.join(os.path.dirname(__file__), "cereal")
capnp_log = capnp.load(os.path.join(CEREAL_DIR, "log.capnp"), imports=[CEREAL_DIR])


class LogReader:
    """Read and iterate over messages in an rlog file."""

    def __init__(self, fn, sort_by_time=False):
        self._file = None
        self._mapping = None
        self._temporary_file = False
        self._ents = []
        source = None
        target = None

        try:
            source = open(fn, "rb")
            magic = source.read(4)
            source.seek(0)

            # Decompress directly into a temporary file.  Keeping the parsed
            # messages backed by mmap avoids a second full-size Python bytes
            # object and bounds the peak memory used by large rlogs.
            if magic == b"\x28\xB5\x2F\xFD":  # zstd magic
                target = tempfile.TemporaryFile()
                with zstd.ZstdDecompressor().stream_reader(source) as reader:
                    shutil.copyfileobj(reader, target, length=1024 * 1024)
                source.close()
                self._temporary_file = True
            elif magic[:2] == b"BZ":
                target = tempfile.TemporaryFile()
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
            self._mapping = mmap.mmap(target.fileno(), 0, access=mmap.ACCESS_READ)
            self._ents = capnp_log.Event.read_multiple_bytes(self._mapping)

            if sort_by_time:
                self._ents = sorted(self._ents, key=lambda e: e.logMonoTime)
        except Exception:
            # Decompression or mmap setup may fail before ``source`` is
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
        return iter(self._ents)

    def close(self):
        """Release the mmap and temporary decompression file."""
        self._ents = []
        if self._mapping is not None:
            try:
                self._mapping.close()
            except (BufferError, OSError):
                # capnp may retain a short-lived view; the file is still
                # released by the owning process if that happens.
                pass
            self._mapping = None
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
