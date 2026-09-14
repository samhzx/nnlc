"""Check for a newer Windows EXE and download it beside the running program."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

from nnlc_runtime import application_directory
from nnlc_version import VersionError, parse_version


MANIFEST_URL = "https://file.897242746.xyz/data/NNLC_Trainer/update.json"
DOWNLOAD_BASE_URL = "https://file.897242746.xyz/data/NNLC_Trainer/"
FILENAME_RE = re.compile(r"^NNLC_Trainer-((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))-windows-x64\.exe$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_MAX_BYTES = 64 * 1024
REQUEST_TIMEOUT_SECONDS = 30
PROGRESS_INTERVAL_SECONDS = 0.2
DOWNLOAD_CHUNK_SIZE = 64 * 1024
CANCEL_POLL_INTERVAL_SECONDS = 0.05


class UpdateError(RuntimeError):
    """Raised when the update manifest or download cannot be used."""


@dataclass(frozen=True)
class UpdateManifest:
    version: str
    size: int
    filename: str
    notes: str
    sha256: str

    @property
    def url(self) -> str:
        return urljoin(DOWNLOAD_BASE_URL, self.filename)


ProgressCallback = Callable[[int, int, int], None]


def parse_manifest(value: object) -> UpdateManifest:
    if not isinstance(value, dict):
        raise UpdateError("更新清单不是有效的 JSON 对象。")

    missing = [name for name in ("version", "size", "filename", "notes", "sha256") if name not in value]
    if missing:
        raise UpdateError("更新清单缺少字段: " + ", ".join(missing))

    version = value["version"]
    try:
        parse_version(version)
    except VersionError as exc:
        raise UpdateError(str(exc)) from exc
    if not isinstance(value["notes"], str):
        raise UpdateError("更新说明必须是字符串。")
    if not isinstance(value["filename"], str) or Path(value["filename"]).name != value["filename"]:
        raise UpdateError("更新文件名无效。")
    match = FILENAME_RE.fullmatch(value["filename"])
    if match is None:
        raise UpdateError("更新文件名格式无效。")
    if match.group(1) != version:
        raise UpdateError("更新文件名与版本号不一致。")
    size = value["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise UpdateError("更新文件大小必须是正整数。")
    sha256 = value["sha256"]
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        raise UpdateError("更新文件校验值无效。")
    return UpdateManifest(
        version=str(version),
        size=size,
        filename=value["filename"],
        notes=value["notes"],
        sha256=sha256,
    )


def _urlopen(url: str):
    request = urllib.request.Request(
        url,
        headers={"Cache-Control": "no-cache", "User-Agent": "NNLC_Trainer"},
    )
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)


def fetch_manifest(url: str = MANIFEST_URL) -> UpdateManifest:
    try:
        with _urlopen(url) as response:
            payload = response.read(MANIFEST_MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise UpdateError(f"无法获取更新清单: {exc}") from exc
    if len(payload) > MANIFEST_MAX_BYTES:
        raise UpdateError("更新清单过大。")
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise UpdateError("更新清单不是有效的 JSON。") from exc
    return parse_manifest(data)


def is_newer_version(remote_version: str, current_version: str) -> bool:
    try:
        return parse_version(remote_version) > parse_version(current_version)
    except VersionError as exc:
        raise UpdateError(str(exc)) from exc


def check_for_update(current_version: str, url: str = MANIFEST_URL) -> UpdateManifest | None:
    manifest = fetch_manifest(url)
    if is_newer_version(manifest.version, current_version):
        return manifest
    return None


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _file_matches_manifest(path: Path, manifest: UpdateManifest) -> bool:
    try:
        if not path.is_file() or path.stat().st_size != manifest.size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest() == manifest.sha256
    except OSError:
        return False


def _remove_if_exists(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
    except OSError:
        pass


def _cleanup_incomplete_download(part_path: Path) -> None:
    _remove_if_exists(part_path)


def existing_update_path(
    manifest: UpdateManifest,
    dest_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    destination = Path(dest_dir) if dest_dir is not None else application_directory()
    target_path = destination / manifest.filename
    if not _file_matches_manifest(target_path, manifest):
        return None
    _remove_if_exists(destination / (manifest.filename + ".part"))
    return target_path


def download_update(
    manifest: UpdateManifest,
    dest_dir: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_event=None,
) -> Path:
    destination = Path(dest_dir) if dest_dir is not None else application_directory()
    target_path = destination / manifest.filename
    part_path = destination / (manifest.filename + ".part")

    existing = existing_update_path(manifest, destination)
    if existing is not None:
        if progress_callback is not None:
            progress_callback(manifest.size, manifest.size, 100)
        return existing

    probe = destination / f".nnlc-write-test-{os.getpid()}"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with probe.open("wb") as handle:
            handle.write(b"ok")
    except OSError as exc:
        raise UpdateError(f"程序目录不可写: {destination}\n{exc}") from exc
    finally:
        _remove_if_exists(probe)

    _remove_if_exists(part_path)
    digest = hashlib.sha256()
    downloaded = 0
    last_progress = 0.0
    completed = False
    stop_cancel_watcher = threading.Event()
    cancel_watcher = None

    def emit_progress(force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < PROGRESS_INTERVAL_SECONDS:
            return
        last_progress = now
        percent = min(100, int(downloaded * 100 / manifest.size)) if manifest.size else 0
        if progress_callback is not None:
            progress_callback(downloaded, manifest.size, percent)

    try:
        with _urlopen(manifest.url) as response, part_path.open("wb") as handle:
            if cancel_event is not None:
                def watch_for_cancel() -> None:
                    while not stop_cancel_watcher.wait(CANCEL_POLL_INTERVAL_SECONDS):
                        if not cancel_event.is_set():
                            continue
                        close = getattr(response, "close", None)
                        if close is not None:
                            try:
                                close()
                            except OSError:
                                pass
                        return

                cancel_watcher = threading.Thread(
                    target=watch_for_cancel,
                    name="nnlc-update-cancel-watcher",
                    daemon=True,
                )
                cancel_watcher.start()
            emit_progress(force=True)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise UpdateError("已取消下载。")
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > manifest.size:
                    raise UpdateError("下载文件大于更新清单中的大小。")
                handle.write(chunk)
                digest.update(chunk)
                emit_progress()
        emit_progress(force=True)
        if downloaded != manifest.size:
            raise UpdateError("下载文件大小与更新清单不一致。")
        if digest.hexdigest() != manifest.sha256:
            raise UpdateError("下载文件校验失败。")
        os.replace(part_path, target_path)
        completed = True
        return target_path
    except UpdateError:
        raise
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if cancel_event is not None and cancel_event.is_set():
            raise UpdateError("已取消下载。") from exc
        raise UpdateError(f"下载更新失败: {exc}") from exc
    finally:
        stop_cancel_watcher.set()
        if cancel_watcher is not None and cancel_watcher.is_alive():
            cancel_watcher.join(timeout=0.2)
        if not completed:
            _cleanup_incomplete_download(part_path)
