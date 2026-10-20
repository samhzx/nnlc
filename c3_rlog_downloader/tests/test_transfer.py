from __future__ import annotations

import io
import json
import stat
import sys
import threading
import time
from datetime import date
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from c3_rlog_downloader.settings import default_destination, load_settings, save_settings
from c3_rlog_downloader.transfer import (
    PREFETCH_CONCURRENT_REQUESTS,
    REMOTE_CONFIG_DIR,
    REMOTE_ROOT,
    CleanupPreview,
    DownloadFailure,
    DownloadSummary,
    InvalidKeyPassphrase,
    KeyPassphraseRequired,
    TransferCancelled,
    TransferError,
    _ProgressRate,
    _start_prefetch,
    cleanup_snapshot,
    connect_sftp,
    date_folder,
    destination_path,
    download_files,
    preview_cleanup,
    run_download,
    scan_remote_tree,
    upload_config_file,
)

EXPORT_DATE = date(2026, 10, 16)
EXPORT_FOLDER = "20261016"


class Attr:
    def __init__(self, filename, mode, size=0, mtime=0):
        self.filename = filename
        self.st_mode = mode
        self.st_size = size
        self.st_mtime = mtime


class FailingBytesIO(io.BytesIO):
    def __init__(self, value, fail_after_first_chunk=False):
        super().__init__(value)
        self.fail_after_first_chunk = fail_after_first_chunk
        self.read_calls = 0
        self.prefetch_calls = []
        self.seek_calls = []

    def prefetch(self, file_size=None, max_concurrent_requests=None):
        self.prefetch_calls.append((file_size, max_concurrent_requests))

    def seek(self, offset, whence=0):
        self.seek_calls.append((offset, whence))
        return super().seek(offset, whence)

    def read(self, size=-1):
        if self.fail_after_first_chunk and self.read_calls == 0:
            self.read_calls += 1
            return super().read(min(size, 5))
        if self.fail_after_first_chunk and self.read_calls == 1:
            self.read_calls += 1
            raise OSError("simulated network drop")
        self.read_calls += 1
        return super().read(size)


class MemorySFTP:
    def __init__(self, now=2_000_000_000):
        self.now = now
        self.nodes = {REMOTE_ROOT: (stat.S_IFDIR | 0o755, b"", now - 1000)}
        self.read_count = {}
        self.stat_count = {}
        self.fail_first_read = set()
        self.removed = []
        self.opened_files = []

    def add_dir(self, path, mtime=None):
        mtime = self.now - 1000 if mtime is None else mtime
        self.nodes[path] = (stat.S_IFDIR | 0o755, b"", mtime)
        parent = path.rsplit("/", 1)[0]
        if parent and parent not in self.nodes:
            self.add_dir(parent, mtime)

    def add_file(self, path, data, mtime=None):
        mtime = self.now - 1000 if mtime is None else mtime
        parent = path.rsplit("/", 1)[0]
        if parent not in self.nodes:
            self.add_dir(parent)
        self.nodes[path] = (stat.S_IFREG | 0o644, data, mtime)

    def add_symlink(self, path, mtime=None):
        mtime = self.now - 1000 if mtime is None else mtime
        parent = path.rsplit("/", 1)[0]
        if parent not in self.nodes:
            self.add_dir(parent)
        self.nodes[path] = (stat.S_IFLNK | 0o777, b"", mtime)

    def _attr(self, path, filename=""):
        mode, data, mtime = self.nodes[path]
        return Attr(filename or path.rsplit("/", 1)[-1], mode, len(data), mtime)

    def lstat(self, path):
        if path not in self.nodes:
            raise FileNotFoundError(path)
        return self._attr(path)

    def stat(self, path):
        self.stat_count[path] = self.stat_count.get(path, 0) + 1
        return self.lstat(path)

    def listdir_attr(self, path):
        if path not in self.nodes or not stat.S_ISDIR(self.nodes[path][0]):
            raise FileNotFoundError(path)
        prefix = path.rstrip("/") + "/"
        values = []
        for child in self.nodes:
            if not child.startswith(prefix):
                continue
            rest = child[len(prefix) :]
            if "/" not in rest:
                values.append(self._attr(child, rest))
        return values

    def open(self, path, mode):
        self.read_count[path] = self.read_count.get(path, 0) + 1
        handle = FailingBytesIO(
            self.nodes[path][1],
            fail_after_first_chunk=path in self.fail_first_read
            and self.read_count[path] == 1,
        )
        self.opened_files.append(handle)
        return handle

    def remove(self, path):
        if path not in self.nodes:
            raise FileNotFoundError(path)
        del self.nodes[path]
        self.removed.append(path)

    def put(self, localpath, remotepath):
        parent = remotepath.rsplit("/", 1)[0]
        if parent not in self.nodes or not stat.S_ISDIR(self.nodes[parent][0]):
            raise FileNotFoundError(parent)
        data = Path(localpath).read_bytes()
        self.nodes[remotepath] = (stat.S_IFREG | 0o644, data, self.now)

    def rmdir(self, path):
        if path not in self.nodes:
            raise FileNotFoundError(path)
        mode, _data, _mtime = self.nodes[path]
        if not stat.S_ISDIR(mode):
            raise OSError("not a directory")
        prefix = path.rstrip("/") + "/"
        if any(child.startswith(prefix) for child in self.nodes):
            raise OSError("directory not empty")
        del self.nodes[path]
        self.removed.append(path)

    def close(self):
        return None


def test_scan_selects_only_stable_exact_rlogs_and_never_follows_symlinks():
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    route = f"{REMOTE_ROOT}/2025-04-03--12-30-00--0"
    sftp.add_file(route + "/rlog.zst", b"rlog", mtime=now - 400)
    sftp.add_file(route + "/nested/rlog.zst", b"nested", mtime=now - 400)
    sftp.add_file(route + "/rlog.bz2", b"other", mtime=now - 400)
    sftp.add_file(route + "/recent/rlog.zst", b"new", mtime=now - 60)
    sftp.add_symlink(route + "/link", mtime=now - 400)

    result = scan_remote_tree(sftp, now=now)

    assert {item.relative_path.as_posix() for item in result.rlog_files} == {
        "2025-04-03--12-30-00--0/rlog.zst",
        "2025-04-03--12-30-00--0/nested/rlog.zst",
    }
    assert [item.relative_path.as_posix() for item in result.recent_rlogs] == [
        "2025-04-03--12-30-00--0/recent/rlog.zst"
    ]
    assert all(
        not item.relative_path.as_posix().endswith("/link")
        for item in result.regular_files
    )


def test_date_folder_uses_local_export_date_and_preserves_route_path(tmp_path):
    from c3_rlog_downloader.transfer import RemoteFile

    route_log = RemoteFile(
        REMOTE_ROOT,
        PurePosixPath("2024-02-29--12-30-00--0/rlog.zst"),
        4,
        0,
        stat.S_IFREG,
    )
    assert date_folder(EXPORT_DATE) == EXPORT_FOLDER
    assert destination_path(tmp_path, route_log, EXPORT_DATE) == (
        tmp_path.resolve() / EXPORT_FOLDER / "2024-02-29--12-30-00--0" / "rlog.zst"
    )


def test_download_is_incremental_and_writes_manifest(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    sftp.add_file(path, b"compressed-log", mtime=now - 400)
    inventory = scan_remote_tree(sftp, now=now)

    first = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)
    local = tmp_path / EXPORT_FOLDER / "2025-04-03--route" / "rlog.zst"
    assert first.all_succeeded
    assert first.downloaded_files == 1
    assert first.downloaded_bytes == len(b"compressed-log")
    assert local.read_bytes() == b"compressed-log"
    assert json.loads((tmp_path / EXPORT_FOLDER / ".c3_rlog_manifest.json").read_text())

    second = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)
    assert second.all_succeeded
    assert second.skipped_existing == 1
    assert second.downloaded_files == 0
    assert sftp.read_count[path] == 1
    assert sftp.stat_count[path] == 2
    assert sftp.opened_files[0].prefetch_calls == [
        (len(b"compressed-log"), PREFETCH_CONCURRENT_REQUESTS)
    ]


def test_same_destination_does_not_silently_overwrite_another_device(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    sftp.add_file(path, b"device-one", mtime=now - 400)
    inventory = scan_remote_tree(sftp, now=now)
    first = download_files(
        sftp,
        inventory,
        tmp_path,
        now=now,
        device_id="192.0.2.1",
        export_date=EXPORT_DATE,
    )
    local = tmp_path / EXPORT_FOLDER / "2025-04-03--route" / "rlog.zst"
    original = local.read_bytes()

    sftp.nodes[path] = (stat.S_IFREG | 0o644, b"device-two", now - 400)
    second_inventory = scan_remote_tree(sftp, now=now)
    second = download_files(
        sftp,
        second_inventory,
        tmp_path,
        now=now,
        device_id="192.0.2.2",
        export_date=EXPORT_DATE,
    )

    assert first.all_succeeded
    assert second.failures
    assert "另一台设备" in second.failures[0].message
    assert local.read_bytes() == original


def test_interrupted_transfer_resumes_matching_partial_file(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    payload = b"abcdefghijklmnopqrstuvwxyz"
    sftp.add_file(path, payload, mtime=now - 400)
    sftp.fail_first_read.add(path)
    inventory = scan_remote_tree(sftp, now=now)

    first = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)
    assert not first.all_succeeded
    part = tmp_path / EXPORT_FOLDER / "2025-04-03--route" / "rlog.zst.part"
    assert part.read_bytes() == payload[:5]

    second = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)
    final = part.with_name("rlog.zst")
    assert second.all_succeeded
    assert final.read_bytes() == payload
    assert not part.exists()


def test_cancel_preserves_partial_file_and_reports_progress(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    sftp.add_file(path, b"a" * (300 * 1024), mtime=now - 400)
    inventory = scan_remote_tree(sftp, now=now)
    cancel = threading.Event()
    progress = []

    def stop_after_first_chunk(value):
        progress.append(value)
        if value.current_bytes:
            cancel.set()

    with pytest.raises(TransferCancelled):
        download_files(
            sftp,
            inventory,
            tmp_path,
            cancel,
            stop_after_first_chunk,
            now=now,
            export_date=EXPORT_DATE,
        )

    part = tmp_path / EXPORT_FOLDER / "2025-04-03--route" / "rlog.zst.part"
    assert part.exists()
    assert progress


def test_cleanup_preview_and_snapshot_skip_recent_changed_and_symlink_entries():
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    route = REMOTE_ROOT + "/2025-04-03--route"
    sftp.add_file(route + "/rlog.zst", b"downloaded", mtime=now - 400)
    sftp.add_file(route + "/qlog", b"other-log", mtime=now - 400)
    sftp.add_file(route + "/recent.mp4", b"keep", mtime=now - 10)
    sftp.add_symlink(route + "/link", mtime=now - 400)
    preview = preview_cleanup(sftp, now=now)
    assert {entry.relative_path.name for entry in preview.entries} == {
        "rlog.zst",
        "qlog",
    }
    assert preview.recent_count == 1

    changed_path = route + "/qlog"
    sftp.add_file(changed_path, b"changed", mtime=now - 200)
    result = cleanup_snapshot(sftp, preview, now_fn=lambda: now)

    assert result.deleted_files == 1
    assert result.skipped_changed == 1
    assert result.deleted_dirs == 0
    assert sftp.nodes[route + "/recent.mp4"]
    assert sftp.nodes[route + "/link"]
    assert changed_path in sftp.nodes
    assert route in sftp.nodes


def test_cleanup_removes_empty_route_directories_after_deleting_files():
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    route = REMOTE_ROOT + "/0000007e--857bed606b--14"
    nested = REMOTE_ROOT + "/2025-04-03--route/0"
    keep = REMOTE_ROOT + "/0000007e--857bed606b--15"
    sftp.add_file(route + "/rlog.zst", b"rlog", mtime=now - 400)
    sftp.add_file(route + "/qlog.zst", b"qlog", mtime=now - 400)
    sftp.add_file(route + "/qcamera.ts", b"cam", mtime=now - 400)
    sftp.add_file(nested + "/rlog.zst", b"nested", mtime=now - 400)
    sftp.add_file(keep + "/rlog.zst", b"keep", mtime=now - 10)
    preview = preview_cleanup(sftp, now=now)
    result = cleanup_snapshot(sftp, preview, now_fn=lambda: now)

    assert result.deleted_files == 4
    assert result.deleted_dirs == 3
    assert route not in sftp.nodes
    assert nested not in sftp.nodes
    assert REMOTE_ROOT + "/2025-04-03--route" not in sftp.nodes
    assert keep in sftp.nodes
    assert keep + "/rlog.zst" in sftp.nodes
    assert REMOTE_ROOT in sftp.nodes


def test_cleanup_rejects_a_root_outside_fixed_realdata():
    with pytest.raises(TransferError, match="固定的 C3 realdata"):
        cleanup_snapshot(MemorySFTP(), CleanupPreview("/tmp", (), 0))


def test_summary_never_reports_success_with_a_failed_file():
    result = DownloadSummary(
        planned_files=2,
        completed_files=1,
        downloaded_files=1,
        skipped_existing=0,
        skipped_recent=0,
        total_bytes=20,
        downloaded_bytes=10,
        completed_bytes=10,
        failures=(DownloadFailure("route/rlog.zst", "network"),),
    )
    assert not result.all_succeeded


def test_any_download_failure_prevents_cleanup_preview(monkeypatch, tmp_path):
    now = int(time.time())
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    sftp.add_file(path, b"partial-transfer", mtime=now - 400)
    sftp.fail_first_read.add(path)
    preview_calls = []

    class Client:
        def close(self):
            return None

    monkeypatch.setattr(
        "c3_rlog_downloader.transfer.connect_sftp",
        lambda *args, **kwargs: (Client(), sftp),
    )
    monkeypatch.setattr(
        "c3_rlog_downloader.transfer.preview_cleanup",
        lambda *args, **kwargs: preview_calls.append(True),
    )

    summary, preview, _inventory = run_download(
        "192.0.2.5", "key", tmp_path, tmp_path / "settings", None, threading.Event()
    )

    assert summary.failures
    assert preview is None
    assert preview_calls == []


def test_supplied_connection_stays_open_for_gui_lifecycle(monkeypatch, tmp_path):
    now = int(time.time())
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    sftp.add_file(path, b"downloaded", mtime=now - 400)
    close_calls = []

    class Client:
        def close(self):
            close_calls.append(True)

    monkeypatch.setattr(
        "c3_rlog_downloader.transfer.connect_sftp",
        lambda *args, **kwargs: pytest.fail("supplied connection should be reused"),
    )

    summary, _preview, _inventory = run_download(
        "192.0.2.5",
        "key",
        tmp_path,
        tmp_path / "settings",
        None,
        threading.Event(),
        connection=(Client(), sftp),
    )

    assert summary.all_succeeded
    assert close_calls == []


def test_start_prefetch_uses_bounded_requests_and_ignores_missing_prefetch():
    class Compatible:
        def __init__(self):
            self.calls = []

        def prefetch(self, file_size=None, max_concurrent_requests=None):
            self.calls.append((file_size, max_concurrent_requests))

    compatible = Compatible()
    _start_prefetch(compatible, 1024)
    _start_prefetch(object(), 512)

    assert compatible.calls == [(1024, PREFETCH_CONCURRENT_REQUESTS)]


def test_existing_manifest_skips_without_remote_stat(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    payload = b"already-downloaded"
    sftp.add_file(path, payload, mtime=now - 400)
    inventory = scan_remote_tree(sftp, now=now)
    first = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)
    sftp.stat_count.clear()
    sftp.read_count.clear()
    sftp.opened_files.clear()

    second = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)

    assert first.all_succeeded
    assert second.skipped_existing == 1
    assert path not in sftp.stat_count
    assert path not in sftp.read_count
    assert sftp.opened_files == []


def test_download_prefetches_after_resume_seek(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    path = REMOTE_ROOT + "/2025-04-03--route/rlog.zst"
    payload = b"abcdefghijklmnopqrstuvwxyz"
    sftp.add_file(path, payload, mtime=now - 400)
    inventory = scan_remote_tree(sftp, now=now)
    local_dir = tmp_path / EXPORT_FOLDER / "2025-04-03--route"
    local_dir.mkdir(parents=True)
    (local_dir / "rlog.zst.part").write_bytes(payload[:8])
    (local_dir / "rlog.zst.part.json").write_text(
        json.dumps(
            {
                "remote_path": "2025-04-03--route/rlog.zst",
                "device_id": "",
                "size": len(payload),
                "mtime": now - 400,
            }
        ),
        encoding="utf-8",
    )

    result = download_files(sftp, inventory, tmp_path, now=now, export_date=EXPORT_DATE)

    assert result.all_succeeded
    assert sftp.opened_files[0].seek_calls == [(8, 0)]
    assert sftp.opened_files[0].prefetch_calls == [
        (len(payload), PREFETCH_CONCURRENT_REQUESTS)
    ]


def test_progress_rate_reports_smoothed_speed_and_eta(monkeypatch):
    ticks = iter((100.5,))
    monkeypatch.setattr("c3_rlog_downloader.transfer.time.monotonic", lambda: next(ticks))
    rate = _ProgressRate(last_sample_at=100.0)

    speed, eta = rate.update(measured_bytes=100, completed_bytes=100, total_bytes=300)

    assert speed == pytest.approx(200.0)
    assert eta == pytest.approx(1.0)


def test_settings_store_ip_key_path_destination_and_config(tmp_path):
    save_settings(
        "192.0.2.5",
        "/keys/c3",
        str(tmp_path / "C3_Rlogs"),
        str(tmp_path / "model.json"),
        tmp_path,
    )
    settings = load_settings(tmp_path)
    raw = (tmp_path / "settings.json").read_text(encoding="utf-8")

    assert settings == {
        "host": "192.0.2.5",
        "key_path": "/keys/c3",
        "destination": str(tmp_path / "C3_Rlogs"),
        "config_path": str(tmp_path / "model.json"),
    }
    assert "passphrase" not in raw
    assert "private_key" not in raw


def test_settings_keep_existing_destination_when_only_connection_is_saved(tmp_path):
    keep = str(tmp_path / "keep")
    config = str(tmp_path / "model.json")
    save_settings("192.0.2.5", "/keys/c3", keep, config, tmp_path)
    save_settings("192.0.2.8", "/keys/other", "", "", tmp_path)
    settings = load_settings(tmp_path)

    assert settings["host"] == "192.0.2.8"
    assert settings["key_path"] == "/keys/other"
    assert settings["destination"] == keep
    assert settings["config_path"] == config


def test_missing_destination_falls_back_to_default(tmp_path):
    (tmp_path / "settings.json").write_text(
        '{"host": "192.0.2.5", "key_path": "/keys/c3"}\n',
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)

    assert settings["destination"] == ""
    assert default_destination().endswith("C3_Rlogs")


def test_upload_config_overwrites_existing_json(tmp_path):
    now = 2_000_000_000
    sftp = MemorySFTP(now)
    sftp.add_dir(REMOTE_CONFIG_DIR)
    remote_path = REMOTE_CONFIG_DIR + "/policy.json"
    sftp.add_file(remote_path, b'{"old": true}', mtime=now - 400)
    local = tmp_path / "policy.json"
    local.write_text('{"new": true}', encoding="utf-8")

    uploaded = upload_config_file(sftp, local)

    assert uploaded == remote_path
    assert sftp.nodes[remote_path][1] == b'{"new": true}'


def test_upload_config_rejects_non_json_and_unknown_directory(tmp_path):
    sftp = MemorySFTP()
    sftp.add_dir(REMOTE_CONFIG_DIR)
    local = tmp_path / "policy.txt"
    local.write_text("not json", encoding="utf-8")
    with pytest.raises(TransferError, match="必须是 JSON"):
        upload_config_file(sftp, local)
    json_file = tmp_path / "policy.json"
    json_file.write_text("{}", encoding="utf-8")
    with pytest.raises(TransferError, match="固定的 NNLC 配置目录"):
        upload_config_file(sftp, json_file, remote_dir="/tmp")


def test_encrypted_key_errors_request_or_reject_a_passphrase(monkeypatch, tmp_path):
    key_path = tmp_path / "encrypted-key"
    key_path.write_text("test placeholder", encoding="utf-8")

    class PasswordRequired(Exception):
        pass

    class SSHError(Exception):
        pass

    class FakeKey:
        @staticmethod
        def from_path(_path, passphrase=None):
            if passphrase is None:
                raise TypeError(
                    "Key is password-protected, but password was not provided."
                )
            raise ValueError("bad decrypt")

    fake_paramiko = SimpleNamespace(
        PKey=FakeKey,
        PasswordRequiredException=PasswordRequired,
        SSHException=SSHError,
    )
    monkeypatch.setitem(sys.modules, "paramiko", fake_paramiko)

    with pytest.raises(KeyPassphraseRequired):
        connect_sftp("192.0.2.5", str(key_path), tmp_path / "settings")
    with pytest.raises(InvalidKeyPassphrase):
        connect_sftp("192.0.2.5", str(key_path), tmp_path / "settings", "wrong")
