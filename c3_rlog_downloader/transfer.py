"""SSH/SFTP transfer operations for the standalone C3 downloader."""

from __future__ import annotations

import json
import os
import posixpath
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath

REMOTE_ROOT = "/data/media/0/realdata"
REMOTE_CONFIG_DIR = (
    "/data/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control"
)
RECENT_WINDOW_SECONDS = 5 * 60
CHUNK_SIZE = 256 * 1024
SFTP_MAX_REQUEST_SIZE = 255 * 1024  # OpenSSH SSH_FXP_READ cap; 256KB triggers short reads
PREFETCH_CONCURRENT_REQUESTS = 64
MANIFEST_NAME = ".c3_rlog_manifest.json"
_PART_META_SUFFIX = ".part.json"


class TransferError(RuntimeError):
    """A user-displayable transfer failure."""


class TransferCancelled(TransferError):
    """Raised when a user cancels an active transfer."""


class KeyPassphraseRequired(TransferError):
    """Raised when an encrypted private key needs a passphrase."""


class InvalidKeyPassphrase(TransferError):
    """Raised when a supplied passphrase cannot unlock the private key."""


class HostKeyMismatch(TransferError):
    """Raised when a previously trusted host presents a different key."""


@dataclass(frozen=True)
class RemoteFile:
    root: str
    relative_path: PurePosixPath
    size: int
    mtime: int
    mode: int

    @property
    def remote_path(self) -> str:
        return posixpath.join(self.root.rstrip("/"), self.relative_path.as_posix())


@dataclass(frozen=True)
class RemoteInventory:
    root: str
    rlog_files: tuple[RemoteFile, ...]
    recent_rlogs: tuple[RemoteFile, ...]
    regular_files: tuple[RemoteFile, ...]
    recent_files: tuple[RemoteFile, ...]
    scanned_at: int


@dataclass(frozen=True)
class TransferProgress:
    completed_bytes: int
    total_bytes: int
    current_path: str
    current_bytes: int
    current_size: int
    completed_files: int
    total_files: int
    phase: str = "download"
    speed_bytes_per_second: float = 0.0
    eta_seconds: float | None = None


@dataclass(frozen=True)
class DownloadFailure:
    remote_path: str
    message: str


@dataclass(frozen=True)
class DownloadSummary:
    planned_files: int
    completed_files: int
    downloaded_files: int
    skipped_existing: int
    skipped_recent: int
    total_bytes: int
    downloaded_bytes: int
    completed_bytes: int
    failures: tuple[DownloadFailure, ...] = field(default_factory=tuple)

    @property
    def all_succeeded(self) -> bool:
        return (
            self.planned_files > 0
            and not self.failures
            and self.completed_files == self.planned_files
        )


@dataclass(frozen=True)
class CleanupPreview:
    root: str
    entries: tuple[RemoteFile, ...]
    recent_count: int

    @property
    def total_bytes(self) -> int:
        return sum(entry.size for entry in self.entries)


@dataclass(frozen=True)
class CleanupSummary:
    deleted_files: int
    deleted_bytes: int
    skipped_changed: int
    failures: tuple[DownloadFailure, ...] = field(default_factory=tuple)
    deleted_dirs: int = 0


ProgressCallback = Callable[[TransferProgress], None] | None


@dataclass
class _ProgressRate:
    """Estimate transfer speed from actual bytes read, with light smoothing."""

    last_bytes: int = 0
    last_sample_at: float = field(default_factory=time.monotonic)
    speed_bytes_per_second: float = 0.0

    def update(
        self, measured_bytes: int, completed_bytes: int, total_bytes: int
    ) -> tuple[float, float | None]:
        now = time.monotonic()
        elapsed = now - self.last_sample_at
        if measured_bytes > self.last_bytes and elapsed > 0.01:
            if elapsed >= 0.25 or completed_bytes >= total_bytes:
                sample = (measured_bytes - self.last_bytes) / elapsed
                if self.speed_bytes_per_second <= 0:
                    self.speed_bytes_per_second = sample
                else:
                    self.speed_bytes_per_second = (
                        self.speed_bytes_per_second * 0.7 + sample * 0.3
                    )
                self.last_bytes = measured_bytes
                self.last_sample_at = now

        if total_bytes <= 0 or self.speed_bytes_per_second <= 0:
            eta = None
        elif completed_bytes >= total_bytes:
            eta = 0.0
        else:
            eta = max(0.0, (total_bytes - completed_bytes) / self.speed_bytes_per_second)
        return self.speed_bytes_per_second, eta


def is_recent(mtime: int, now: int | None = None) -> bool:
    """Return whether a remote entry changed within the five-minute window."""
    current = int(time.time()) if now is None else int(now)
    return int(mtime) >= current - RECENT_WINDOW_SECONDS


def date_folder(export_date: date | None = None) -> str:
    """Return the local export date as YYYYMMDD, independent of remote dates."""
    return (export_date or date.today()).strftime("%Y%m%d")


def destination_path(
    destination: Path, remote_file: RemoteFile, export_date: date | None = None
) -> Path:
    """Map a remote rlog to its export-date path without escaping the root."""
    root = Path(destination).expanduser().resolve()
    relative = Path(date_folder(export_date), *remote_file.relative_path.parts)
    target = (root / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TransferError("远端路径转换后超出目标目录，已拒绝下载。") from exc
    return target


def _safe_relative_path(value: PurePosixPath | str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise TransferError("远端返回了不安全的相对路径，已拒绝操作。")
    if any("/" in part or "\\" in part or "\x00" in part for part in relative.parts):
        raise TransferError("远端返回了不安全的文件名，已拒绝操作。")
    return relative


def _entry_from_attr(root: str, relative: PurePosixPath, attr) -> RemoteFile:
    return RemoteFile(
        root=root,
        relative_path=_safe_relative_path(relative),
        size=max(0, int(getattr(attr, "st_size", 0) or 0)),
        mtime=int(getattr(attr, "st_mtime", 0) or 0),
        mode=int(getattr(attr, "st_mode", 0) or 0),
    )


def scan_remote_tree(
    sftp, root: str = REMOTE_ROOT, now: int | None = None
) -> RemoteInventory:
    """Recursively list regular files without following symbolic links."""
    scanned_at = int(time.time()) if now is None else int(now)
    try:
        root_attr = sftp.lstat(root)
    except OSError as exc:
        raise TransferError(
            "无法访问设备 realdata 目录，请确认设备已连接且目录存在。"
        ) from exc
    if not stat.S_ISDIR(int(getattr(root_attr, "st_mode", 0) or 0)):
        raise TransferError("设备 realdata 路径不是目录。")

    regular_files: list[RemoteFile] = []

    def walk(remote_dir: str, parents: tuple[str, ...]) -> None:
        try:
            entries = sftp.listdir_attr(remote_dir)
        except OSError as exc:
            raise TransferError("读取设备日志目录失败：" + remote_dir) from exc
        for attr in entries:
            name = getattr(attr, "filename", "")
            if (
                not name
                or name in (".", "..")
                or "/" in name
                or "\\" in name
                or "\x00" in name
            ):
                continue
            relative = PurePosixPath(*(parents + (name,)))
            mode = int(getattr(attr, "st_mode", 0) or 0)
            if stat.S_ISLNK(mode):
                continue
            full_path = posixpath.join(remote_dir, name)
            if stat.S_ISDIR(mode):
                walk(full_path, parents + (name,))
            elif stat.S_ISREG(mode):
                regular_files.append(_entry_from_attr(root, relative, attr))

    walk(root, ())
    regular_files.sort(key=lambda item: item.relative_path.as_posix().casefold())
    recent_files = tuple(
        item for item in regular_files if is_recent(item.mtime, scanned_at)
    )
    rlogs = tuple(
        item for item in regular_files if item.relative_path.name == "rlog.zst"
    )
    recent_rlogs = tuple(item for item in rlogs if is_recent(item.mtime, scanned_at))
    return RemoteInventory(
        root=root,
        rlog_files=tuple(
            item for item in rlogs if not is_recent(item.mtime, scanned_at)
        ),
        recent_rlogs=recent_rlogs,
        regular_files=tuple(regular_files),
        recent_files=recent_files,
        scanned_at=scanned_at,
    )


def _load_manifest(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": 1, "files": {}}
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, dict):
        files = {}
    return {"version": 1, "files": files}


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _manifest_entry_matches(
    entry: object, item: RemoteFile, local_path: Path, destination: Path, device_id: str
) -> bool:
    if not isinstance(entry, dict) or not local_path.is_file():
        return False
    try:
        relative = local_path.relative_to(destination.resolve()).as_posix()
        return (
            entry.get("destination") == relative
            and entry.get("device_id", "") == device_id
            and int(entry.get("size", -1)) == item.size
            and int(entry.get("mtime", -1)) == item.mtime
            and local_path.stat().st_size == item.size
        )
    except (OSError, ValueError, TypeError):
        return False


def _part_metadata_matches(path: Path, item: RemoteFile, device_id: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value.get("remote_path") == item.relative_path.as_posix()
            and value.get("device_id", "") == device_id
            and int(value.get("size", -1)) == item.size
            and int(value.get("mtime", -1)) == item.mtime
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _emit_progress(callback: ProgressCallback, progress: TransferProgress) -> None:
    if callback is not None:
        callback(progress)


def _check_cancel(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise TransferCancelled(
            "下载已取消。已完成文件保留，未完成文件可在下次运行时续传。"
        )


def _start_prefetch(remote_file, file_size: int) -> None:
    """Pipeline remaining SFTP reads when the backend supports Paramiko prefetch."""
    prefetch = getattr(remote_file, "prefetch", None)
    if callable(prefetch):
        prefetch(
            file_size,
            max_concurrent_requests=PREFETCH_CONCURRENT_REQUESTS,
        )


def _download_one(
    sftp,
    item: RemoteFile,
    remote_attr,
    part: Path,
    metadata_path: Path,
    cancel_event: threading.Event,
    on_bytes: Callable[[int], None],
    device_id: str,
) -> int:
    if item.size <= 0:
        raise TransferError("远端文件为空，未将其视为有效日志。")
    if int(remote_attr.st_size) != item.size or int(remote_attr.st_mtime) != item.mtime:
        raise TransferError("远端文件在扫描后发生变化；请重新扫描后再下载。")

    for candidate in (part, metadata_path):
        if candidate.is_symlink():
            candidate.unlink()
        elif candidate.exists() and not candidate.is_file():
            raise TransferError("临时下载路径不是普通文件，已拒绝写入。")
    if not _part_metadata_matches(metadata_path, item, device_id):
        try:
            part.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
        except OSError as exc:
            raise TransferError("无法清理不匹配的临时下载文件。") from exc
    _write_json_atomic(
        metadata_path,
        {
            "remote_path": item.relative_path.as_posix(),
            "device_id": device_id,
            "size": item.size,
            "mtime": item.mtime,
        },
    )

    offset = part.stat().st_size if part.exists() else 0
    if offset > item.size:
        part.unlink(missing_ok=True)
        offset = 0
    on_bytes(offset)

    try:
        with sftp.open(item.remote_path, "rb") as remote_file:
            if offset:
                remote_file.seek(offset)
            _start_prefetch(remote_file, item.size)
            mode = "ab" if offset else "wb"
            with part.open(mode) as local_file:
                position = offset
                while position < item.size:
                    _check_cancel(cancel_event)
                    chunk = remote_file.read(min(CHUNK_SIZE, item.size - position))
                    if not chunk:
                        raise TransferError("传输提前结束，临时文件已保留以便续传。")
                    local_file.write(chunk)
                    local_file.flush()
                    position += len(chunk)
                    on_bytes(position)
    except TransferCancelled:
        raise
    except TransferError:
        raise
    except Exception as exc:
        raise TransferError("读取远端文件失败；临时文件已保留以便续传。") from exc

    try:
        after = sftp.stat(item.remote_path)
    except OSError as exc:
        raise TransferError("下载后无法重新验证远端文件；临时文件已保留。") from exc
    if int(after.st_size) != item.size or int(after.st_mtime) != item.mtime:
        part.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise TransferError("远端文件在下载过程中发生变化，已丢弃该临时文件。")
    if part.stat().st_size != item.size:
        raise TransferError("下载文件大小校验失败；临时文件已保留以便续传。")
    return item.size


def download_files(
    sftp,
    inventory: RemoteInventory,
    destination: Path,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback = None,
    now: int | None = None,
    device_id: str = "",
    export_date: date | None = None,
) -> DownloadSummary:
    """Download stable rlogs, atomically publish them, and persist resume state."""
    cancel_event = cancel_event or threading.Event()
    export_date = export_date or date.today()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    day_dir = destination / date_folder(export_date)
    day_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = day_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)
    manifest_files = manifest["files"]
    total_bytes = sum(item.size for item in inventory.rlog_files)
    completed_bytes = 0
    completed_files = 0
    downloaded_files = 0
    skipped_existing = 0
    skipped_recent = len(inventory.recent_rlogs)
    downloaded_bytes = 0
    measured_bytes = 0
    rate = _ProgressRate()
    failures: list[DownloadFailure] = []
    total_files = len(inventory.rlog_files)

    for item in inventory.rlog_files:
        _check_cancel(cancel_event)
        relative_key = item.relative_path.as_posix()
        key = f"{device_id}|{relative_key}"
        local_path = destination_path(destination, item, export_date)
        entry = manifest_files.get(key)
        if _manifest_entry_matches(entry, item, local_path, destination, device_id):
            completed_files += 1
            skipped_existing += 1
            completed_bytes += item.size
            speed, eta = rate.update(measured_bytes, completed_bytes, total_bytes)
            _emit_progress(
                progress_callback,
                TransferProgress(
                    completed_bytes,
                    total_bytes,
                    key,
                    item.size,
                    item.size,
                    completed_files,
                    total_files,
                    speed_bytes_per_second=speed,
                    eta_seconds=eta,
                ),
            )
            continue

        try:
            current = sftp.stat(item.remote_path)
        except OSError as exc:
            failures.append(DownloadFailure(key, "读取远端文件信息失败：" + str(exc)))
            continue
        if is_recent(current.st_mtime, now):
            skipped_recent += 1
            completed_files += 1
            completed_bytes += item.size
            speed, eta = rate.update(measured_bytes, completed_bytes, total_bytes)
            _emit_progress(
                progress_callback,
                TransferProgress(
                    completed_bytes,
                    total_bytes,
                    key,
                    item.size,
                    item.size,
                    completed_files,
                    total_files,
                    speed_bytes_per_second=speed,
                    eta_seconds=eta,
                ),
            )
            continue
        if int(current.st_size) != item.size or int(current.st_mtime) != item.mtime:
            failures.append(
                DownloadFailure(key, "远端文件在扫描后发生变化；请重新扫描后再下载。")
            )
            continue

        local_relative = local_path.relative_to(destination).as_posix()
        owned_by_other_device = any(
            isinstance(existing, dict)
            and existing.get("destination") == local_relative
            and existing.get("device_id", "") != device_id
            for existing in manifest_files.values()
        )
        if owned_by_other_device and local_path.exists():
            failures.append(
                DownloadFailure(
                    relative_key,
                    "目标目录已有另一台设备的同名日志；请选择独立目录以避免覆盖。",
                )
            )
            continue

        part = local_path.with_name(local_path.name + ".part")
        metadata_path = local_path.with_name(local_path.name + _PART_META_SUFFIX)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_start_position: list[int | None] = [None]

        def report_file_progress(
            position: int,
            base_bytes: int = completed_bytes,
            current_key: str = key,
            current_size: int = item.size,
            files_done: int = completed_files,
        ) -> None:
            nonlocal measured_bytes
            if file_start_position[0] is None:
                file_start_position[0] = position
            current_measured_bytes = measured_bytes + max(
                0, position - file_start_position[0]
            )
            speed, eta = rate.update(
                current_measured_bytes, base_bytes + position, total_bytes
            )
            _emit_progress(
                progress_callback,
                TransferProgress(
                    base_bytes + position,
                    total_bytes,
                    current_key,
                    position,
                    current_size,
                    files_done,
                    total_files,
                    speed_bytes_per_second=speed,
                    eta_seconds=eta,
                ),
            )

        try:
            _download_one(
                sftp,
                item,
                current,
                part,
                metadata_path,
                cancel_event,
                report_file_progress,
                device_id,
            )
            os.replace(part, local_path)
            metadata_path.unlink(missing_ok=True)
            relative_destination = local_path.relative_to(destination).as_posix()
            manifest_files[key] = {
                "destination": relative_destination,
                "device_id": device_id,
                "size": item.size,
                "mtime": item.mtime,
            }
            _write_json_atomic(manifest_path, manifest)
            measured_bytes += item.size - (file_start_position[0] or 0)
            completed_files += 1
            downloaded_files += 1
            downloaded_bytes += item.size
            completed_bytes += item.size
        except TransferCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - SFTP backends expose different per-file errors.
            failures.append(DownloadFailure(key, str(exc)))

    return DownloadSummary(
        planned_files=total_files,
        completed_files=completed_files,
        downloaded_files=downloaded_files,
        skipped_existing=skipped_existing,
        skipped_recent=skipped_recent,
        total_bytes=total_bytes,
        downloaded_bytes=downloaded_bytes,
        completed_bytes=completed_bytes,
        failures=tuple(failures),
    )


def preview_cleanup(
    sftp, root: str = REMOTE_ROOT, now: int | None = None
) -> CleanupPreview:
    """Snapshot deletable regular files while excluding recent files and symlinks."""
    if posixpath.normpath(root) != REMOTE_ROOT:
        raise TransferError("清理范围不是固定的 C3 realdata 目录，已拒绝操作。")
    inventory = scan_remote_tree(sftp, root=root, now=now)
    entries = tuple(
        item
        for item in inventory.regular_files
        if not is_recent(item.mtime, inventory.scanned_at)
    )
    return CleanupPreview(
        root=root, entries=entries, recent_count=len(inventory.recent_files)
    )


def _directory_is_empty(sftp, path: str) -> bool:
    try:
        entries = sftp.listdir_attr(path)
    except OSError:
        return False
    return not any(
        getattr(entry, "filename", "") not in ("", ".", "..") for entry in entries
    )


def _remove_empty_ancestor_dirs(
    sftp,
    root: str,
    relative_files: list[PurePosixPath],
    cancel_event: threading.Event,
) -> int:
    """Remove empty route directories left behind after deleting files."""
    root = posixpath.normpath(root)
    candidates: set[str] = set()
    for relative in relative_files:
        parts = relative.parts[:-1]
        for depth in range(len(parts), 0, -1):
            candidates.add(posixpath.join(root, *parts[:depth]))
    removed = 0
    for path in sorted(candidates, key=lambda value: value.count("/"), reverse=True):
        _check_cancel(cancel_event)
        normalized = posixpath.normpath(path)
        if normalized == root or not normalized.startswith(root.rstrip("/") + "/"):
            continue
        try:
            current = sftp.lstat(path)
            mode = int(getattr(current, "st_mode", 0) or 0)
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                continue
            if not _directory_is_empty(sftp, path):
                continue
            sftp.rmdir(path)
            removed += 1
        except TransferCancelled:
            raise
        except Exception:  # noqa: BLE001 - leftover empty dirs must not fail the cleanup.
            continue
    return removed


def cleanup_snapshot(
    sftp,
    preview: CleanupPreview,
    cancel_event: threading.Event | None = None,
    progress_callback: ProgressCallback = None,
    now_fn: Callable[[], int] = lambda: int(time.time()),
) -> CleanupSummary:
    """Delete only unchanged regular files captured by a cleanup preview."""
    if posixpath.normpath(preview.root) != REMOTE_ROOT:
        raise TransferError("清理范围不是固定的 C3 realdata 目录，已拒绝操作。")
    cancel_event = cancel_event or threading.Event()
    deleted_files = 0
    deleted_bytes = 0
    skipped_changed = 0
    failures: list[DownloadFailure] = []
    deleted_relatives: list[PurePosixPath] = []
    total_bytes = preview.total_bytes
    total_files = len(preview.entries)
    rate = _ProgressRate()

    for index, item in enumerate(preview.entries, start=1):
        _check_cancel(cancel_event)
        relative = _safe_relative_path(item.relative_path)
        remote_path = posixpath.join(preview.root.rstrip("/"), relative.as_posix())
        try:
            current = sftp.lstat(remote_path)
            mode = int(getattr(current, "st_mode", 0) or 0)
            if (
                not stat.S_ISREG(mode)
                or int(current.st_size) != item.size
                or int(current.st_mtime) != item.mtime
                or is_recent(current.st_mtime, now_fn())
            ):
                skipped_changed += 1
                continue
            sftp.remove(remote_path)
            deleted_files += 1
            deleted_bytes += item.size
            deleted_relatives.append(relative)
            speed, eta = rate.update(deleted_bytes, deleted_bytes, total_bytes)
            _emit_progress(
                progress_callback,
                TransferProgress(
                    deleted_bytes,
                    total_bytes,
                    relative.as_posix(),
                    item.size,
                    item.size,
                    index,
                    total_files,
                    speed_bytes_per_second=speed,
                    eta_seconds=eta,
                    phase="cleanup",
                ),
            )
        except TransferCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - continue cleanup after individual SFTP failures.
            failures.append(DownloadFailure(relative.as_posix(), str(exc)))
    deleted_dirs = _remove_empty_ancestor_dirs(
        sftp, preview.root, deleted_relatives, cancel_event
    )
    return CleanupSummary(
        deleted_files,
        deleted_bytes,
        skipped_changed,
        tuple(failures),
        deleted_dirs,
    )


def connect_sftp(
    host: str,
    key_filename: str,
    settings_dir: Path,
    passphrase: str | None = None,
    timeout: float = 15.0,
):
    """Create a pinned-on-first-use SSH connection and return client plus SFTP."""
    try:
        import paramiko
    except ImportError as exc:
        raise TransferError("SSH 组件未打包，请重新构建独立下载客户端。") from exc

    import ipaddress

    try:
        ipaddress.ip_address(host)
    except ValueError as exc:
        raise TransferError("请输入有效的 C3 IP 地址。") from exc

    key_path = Path(key_filename).expanduser()
    if not key_path.is_file():
        raise TransferError("所选私钥文件不存在。")
    encoded_passphrase = (
        passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
    )
    try:
        private_key = paramiko.PKey.from_path(
            str(key_path), passphrase=encoded_passphrase
        )
    except paramiko.PasswordRequiredException as exc:
        raise KeyPassphraseRequired("该私钥需要口令。") from exc
    except Exception as exc:
        message = str(exc).casefold()
        missing_passphrase = any(
            marker in message
            for marker in (
                "password was not given",
                "password was not provided",
                "password-protected",
                "private key is encrypted",
            )
        )
        if not passphrase and missing_passphrase:
            raise KeyPassphraseRequired("该私钥需要口令。") from exc
        if passphrase:
            raise InvalidKeyPassphrase(
                "无法使用该口令解锁私钥；请检查口令或密钥格式。"
            ) from exc
        raise TransferError("私钥格式不受支持或文件已损坏。") from exc

    paramiko.SFTPFile.MAX_REQUEST_SIZE = SFTP_MAX_REQUEST_SIZE

    settings_dir = Path(settings_dir)
    settings_dir.mkdir(parents=True, exist_ok=True)
    known_hosts = settings_dir / "known_hosts"
    known_hosts.touch(exist_ok=True)

    client = paramiko.SSHClient()
    try:
        client.load_host_keys(str(known_hosts))
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=host,
            port=22,
            username="comma",
            pkey=private_key,
            allow_agent=False,
            look_for_keys=False,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )
        client.save_host_keys(str(known_hosts))
        sftp = client.open_sftp()
        channel = sftp.get_channel()
        channel.settimeout(30.0)
        return client, sftp
    except paramiko.BadHostKeyException as exc:
        client.close()
        raise HostKeyMismatch(
            "设备 SSH 主机指纹与本机记录不一致，已拒绝连接。"
        ) from exc
    except paramiko.AuthenticationException as exc:
        client.close()
        raise TransferError("SSH 身份验证失败，请确认私钥已授权给这台 C3。") from exc
    except Exception as exc:
        client.close()
        if isinstance(exc, paramiko.SSHException):
            raise TransferError("SSH 连接失败：" + str(exc)) from exc
        raise TransferError("无法连接 C3：" + str(exc)) from exc


def upload_config_file(sftp, local_path: Path, remote_dir: str = REMOTE_CONFIG_DIR) -> str:
    """Overwrite a local JSON file onto the device NNLC config directory."""
    source = Path(local_path).expanduser()
    if not source.is_file():
        raise TransferError("请选择存在的模型参数文件。")
    if source.suffix.casefold() != ".json":
        raise TransferError("模型参数文件必须是 JSON。")
    remote_dir = posixpath.normpath(remote_dir)
    if remote_dir != REMOTE_CONFIG_DIR:
        raise TransferError("上传目录不是固定的 NNLC 配置目录，已拒绝操作。")
    try:
        attr = sftp.lstat(remote_dir)
    except OSError as exc:
        raise TransferError("无法访问设备配置目录，请确认设备已连接且目录存在。") from exc
    mode = int(getattr(attr, "st_mode", 0) or 0)
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise TransferError("设备配置路径不是目录。")
    remote_path = posixpath.join(remote_dir, source.name)
    try:
        sftp.put(str(source), remote_path)
    except Exception as exc:  # noqa: BLE001 - surface SFTP backend failures to the UI.
        raise TransferError("上传模型参数失败：" + str(exc)) from exc
    return remote_path


def close_sftp(client, sftp) -> None:
    """Close both Paramiko handles explicitly."""
    try:
        if sftp is not None:
            sftp.close()
    finally:
        if client is not None:
            client.close()


def run_download(
    host: str,
    key_filename: str,
    destination: Path,
    settings_dir: Path,
    passphrase: str | None,
    cancel_event: threading.Event,
    progress_callback: ProgressCallback = None,
    connection: tuple[object, object] | None = None,
):
    """Scan, download, and preview cleanup entries on a supplied or new connection."""
    export_date = date.today()
    client = sftp = None
    owns_connection = connection is None
    try:
        if connection is None:
            client, sftp = connect_sftp(host, key_filename, settings_dir, passphrase)
        else:
            client, sftp = connection
        inventory = scan_remote_tree(sftp)
        result = download_files(
            sftp,
            inventory,
            destination,
            cancel_event,
            progress_callback,
            device_id=host,
            export_date=export_date,
        )
        cleanup = preview_cleanup(sftp) if result.all_succeeded else None
        return result, cleanup, inventory
    finally:
        if owns_connection:
            close_sftp(client, sftp)


def run_cleanup(
    host: str,
    key_filename: str,
    settings_dir: Path,
    passphrase: str | None,
    preview: CleanupPreview,
    cancel_event: threading.Event,
    progress_callback: ProgressCallback = None,
    connection: tuple[object, object] | None = None,
) -> CleanupSummary:
    """Remove only the previously previewed unchanged files on a supplied or new connection."""
    client = sftp = None
    owns_connection = connection is None
    try:
        if connection is None:
            client, sftp = connect_sftp(host, key_filename, settings_dir, passphrase)
        else:
            client, sftp = connection
        return cleanup_snapshot(sftp, preview, cancel_event, progress_callback)
    finally:
        if owns_connection:
            close_sftp(client, sftp)
