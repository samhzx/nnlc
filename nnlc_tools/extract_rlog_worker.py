"""Persistent isolated worker for extracting individual rlog files.

The parent process communicates with this module using UTF-8 JSON lines.  A
native crash in pycapnp or a decompression library therefore terminates only
this worker, allowing the parent extractor to identify the current rlog and
apply strict or tolerant error handling.
"""

import gc
import json
import os
import sys

from nnlc_tools.extract_lateral_data import (
    _remove_with_retry,
    _StreamingCsvWriter,
    extract_segment,
)


def _open_windows_standard_handle(handle_id, mode):
    """Open a binary stream for an inherited handle in a windowed EXE."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel32.GetStdHandle.restype = wintypes.HANDLE
    kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.DuplicateHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    source_handle = kernel32.GetStdHandle(handle_id & 0xFFFFFFFF)
    if source_handle in (None, 0, wintypes.HANDLE(-1).value):
        raise OSError(f"standard handle {handle_id} is unavailable")

    current_process = kernel32.GetCurrentProcess()
    duplicate_handle = wintypes.HANDLE()
    if not kernel32.DuplicateHandle(
        current_process,
        source_handle,
        current_process,
        ctypes.byref(duplicate_handle),
        0,
        False,
        0x00000002,  # DUPLICATE_SAME_ACCESS
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    flags = os.O_BINARY | (os.O_RDONLY if mode == "rb" else os.O_WRONLY)
    try:
        descriptor = msvcrt.open_osfhandle(duplicate_handle.value, flags)
    except Exception:
        kernel32.CloseHandle(duplicate_handle)
        raise
    return os.fdopen(descriptor, mode, buffering=0)


def _protocol_stream(stream, descriptor, windows_handle, mode):
    """Return a binary protocol stream and whether this module owns it."""
    if stream is not None:
        return getattr(stream, "buffer", stream), False
    if os.name == "nt":
        return _open_windows_standard_handle(windows_handle, mode), True
    return os.fdopen(os.dup(descriptor), mode, buffering=0), True


def _send_response(response, output_stream):
    payload = json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n"
    output_stream.write(payload)
    output_stream.flush()


def _process_request(request):
    rlog_path = os.fspath(request["rlog_path"])
    output_path = os.fspath(request["output_path"])
    writer = None
    try:
        writer = _StreamingCsvWriter(
            output_path,
            temporal=bool(request.get("temporal", False)),
            filter_overrides=bool(request.get("filter_overrides", False)),
            route_id=request.get("route_id", "unknown"),
        )
        extract_segment(rlog_path, row_callback=writer.accept)
        writer.finish()
        result = {
            "status": "ok",
            "rows_written": writer.rows_written,
            "rows_seen": writer.rows_seen,
            "rows_filtered": writer.rows_filtered,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        if writer is not None:
            writer.close()

    if result["status"] != "ok":
        _remove_with_retry(output_path)

    # Force pycapnp readers to be released before acknowledging completion.
    # If native cleanup itself fails, the parent sees a worker crash instead
    # of incorrectly accepting a partially processed segment.
    gc.collect()
    return result


def main():
    input_stream, close_input = _protocol_stream(sys.stdin, 0, -10, "rb")
    output_stream, close_output = _protocol_stream(sys.stdout, 1, -11, "wb")
    try:
        while True:
            request_line = input_stream.readline()
            if not request_line:
                return
            try:
                request = json.loads(request_line.decode("utf-8"))
                response = _process_request(request)
            except Exception as exc:
                response = {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            _send_response(response, output_stream)
    finally:
        if close_input:
            input_stream.close()
        if close_output:
            output_stream.close()


if __name__ == "__main__":
    main()
