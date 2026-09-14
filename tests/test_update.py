import hashlib
import io
import json
import threading
import time
from pathlib import Path

import pytest

from nnlc_update import (
    UpdateError,
    check_for_update,
    download_update,
    existing_update_path,
    is_newer_version,
    parse_manifest,
)


def make_manifest(**overrides):
    payload = {
        "version": "1.0.1",
        "size": 4,
        "filename": "NNLC_Trainer-1.0.1-windows-x64.exe",
        "notes": "修复训练流程问题。",
        "sha256": hashlib.sha256(b"data").hexdigest(),
        "ignored": "extra",
    }
    payload.update(overrides)
    return payload


def test_parse_manifest_accepts_required_fields_and_ignores_extras():
    manifest = parse_manifest(make_manifest())
    assert manifest.version == "1.0.1"
    assert manifest.filename == "NNLC_Trainer-1.0.1-windows-x64.exe"
    assert manifest.url.endswith("/NNLC_Trainer-1.0.1-windows-x64.exe")


def test_parse_manifest_rejects_invalid_values():
    with pytest.raises(UpdateError, match="缺少字段"):
        parse_manifest({"version": "1.0.1"})
    with pytest.raises(UpdateError, match="文件名"):
        parse_manifest(make_manifest(filename="NNLC_Trainer.exe"))
    with pytest.raises(UpdateError, match="文件名"):
        parse_manifest(make_manifest(filename="NNLC_Trainer-01.0.1-windows-x64.exe"))
    with pytest.raises(UpdateError, match="不一致"):
        parse_manifest(make_manifest(filename="NNLC_Trainer-1.0.2-windows-x64.exe"))
    with pytest.raises(UpdateError, match="大小"):
        parse_manifest(make_manifest(size=0))
    with pytest.raises(UpdateError, match="大小"):
        parse_manifest(make_manifest(size="4"))
    with pytest.raises(UpdateError, match="大小"):
        parse_manifest(make_manifest(size=True))
    with pytest.raises(UpdateError, match="校验"):
        parse_manifest(make_manifest(sha256="abc"))


def test_is_newer_version():
    assert is_newer_version("1.0.1", "1.0.0")
    assert not is_newer_version("1.0.0", "1.0.0")
    assert not is_newer_version("0.9.9", "1.0.0")
    assert is_newer_version("1.10.0", "1.9.9")


class FakeResponse:
    def __init__(self, payload: bytes, chunks: list[bytes] | None = None):
        self.payload = payload
        self.chunks = list(chunks) if chunks is not None else None
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1):
        if self.chunks is not None:
            if not self.chunks:
                return b""
            return self.chunks.pop(0)
        return self._buffer.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_check_for_update_only_returns_newer_manifest(monkeypatch):
    payload = json.dumps(make_manifest()).encode("utf-8")
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(payload))

    assert check_for_update("1.0.0") is not None
    assert check_for_update("1.0.1") is None
    assert check_for_update("1.2.0") is None


def test_download_update_writes_versioned_file_and_reports_progress(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(content, chunks=[b"da", b"ta"]))
    monkeypatch.setattr("nnlc_update.PROGRESS_INTERVAL_SECONDS", 0)

    progress = []
    path = download_update(manifest, dest_dir=tmp_path, progress_callback=lambda *item: progress.append(item))

    assert path == tmp_path / manifest.filename
    assert path.read_bytes() == content
    assert [item.name for item in tmp_path.iterdir()] == [manifest.filename]
    assert progress[-1] == (len(content), len(content), 100)


def test_existing_valid_file_skips_download(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    target = tmp_path / manifest.filename
    target.write_bytes(content)
    leftover = tmp_path / (manifest.filename + ".part")
    leftover.write_bytes(b"partial")

    def fail_open(url):
        raise AssertionError("should not download")

    monkeypatch.setattr("nnlc_update._urlopen", fail_open)
    path = download_update(manifest, dest_dir=tmp_path)
    assert path == target
    assert existing_update_path(manifest, tmp_path) == target
    assert not leftover.exists()


def test_failed_download_deletes_partial_files(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(b"datx"))

    with pytest.raises(UpdateError, match="校验失败"):
        download_update(manifest, dest_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_cancel_deletes_partial_files(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    cancel_event = threading.Event()
    cancel_event.set()
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(content, chunks=[b"da", b"ta"]))

    with pytest.raises(UpdateError, match="取消"):
        download_update(manifest, dest_dir=tmp_path, cancel_event=cancel_event)

    assert list(tmp_path.iterdir()) == []


def test_short_download_is_rejected(tmp_path, monkeypatch):
    manifest = parse_manifest(make_manifest(size=4, sha256=hashlib.sha256(b"data").hexdigest()))
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(b"da"))

    with pytest.raises(UpdateError, match="大小"):
        download_update(manifest, dest_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_oversized_download_is_rejected(tmp_path, monkeypatch):
    manifest = parse_manifest(make_manifest(size=2, sha256=hashlib.sha256(b"ab").hexdigest()))
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(b"abcd"))

    with pytest.raises(UpdateError, match="大于"):
        download_update(manifest, dest_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_failed_download_keeps_existing_unrelated_file(tmp_path, monkeypatch):
    existing = b"old-file"
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))
    target = tmp_path / manifest.filename
    target.write_bytes(existing)
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FakeResponse(b"datx"))

    with pytest.raises(UpdateError, match="校验失败"):
        download_update(manifest, dest_dir=tmp_path)

    assert target.read_bytes() == existing
    assert not (tmp_path / (manifest.filename + ".part")).exists()


def test_write_probe_is_removed_after_unwritable_check(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))

    def fail_open(self, *args, **kwargs):
        raise OSError("read only")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(UpdateError, match="不可写"):
        download_update(manifest, dest_dir=tmp_path)

    leftover = [path.name for path in tmp_path.iterdir() if path.name.startswith(".nnlc-write-test-")]
    assert leftover == []


def test_interrupted_download_deletes_partial_files(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest()))

    class FailingResponse(FakeResponse):
        def read(self, size: int = -1):
            chunk = super().read(size)
            if chunk:
                raise OSError("network down")
            return chunk

    monkeypatch.setattr("nnlc_update._urlopen", lambda url: FailingResponse(content, chunks=[b"da", b"ta"]))

    with pytest.raises(UpdateError, match="下载更新失败"):
        download_update(manifest, dest_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_cancel_closes_blocked_response_and_cleans_partial_file(tmp_path, monkeypatch):
    content = b"data"
    manifest = parse_manifest(
        make_manifest(size=len(content), sha256=hashlib.sha256(content).hexdigest())
    )
    cancel_event = threading.Event()

    class BlockingResponse(FakeResponse):
        def __init__(self):
            super().__init__(content)
            self.closed = threading.Event()

        def read(self, size: int = -1):
            self.closed.wait(2)
            if not self.closed.is_set():
                raise TimeoutError("test response remained blocked")
            raise OSError("response closed")

        def close(self):
            self.closed.set()

    response = BlockingResponse()
    monkeypatch.setattr("nnlc_update._urlopen", lambda url: response)
    errors = []

    def run_download():
        try:
            download_update(manifest, dest_dir=tmp_path, cancel_event=cancel_event)
        except UpdateError as exc:
            errors.append(exc)

    thread = threading.Thread(target=run_download)
    thread.start()
    time.sleep(0.05)
    cancel_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert response.closed.is_set()
    assert errors and "取消" in str(errors[0])
    assert list(tmp_path.iterdir()) == []
