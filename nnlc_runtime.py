"""Bootstrap the reusable Windows Julia runtime beside the packaged EXE."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


CONFIG_FILENAME = "windows_runtime.json"
MANIFEST_FILENAME = "runtime-manifest.json"
COPY_CHUNK_SIZE = 1024 * 1024
STAGING_DIRECTORY_PREFIX = ".nnlc-"
STAGING_TOKEN_LENGTH = 8
ProgressCallback = Callable[[int, int, str], None]


class RuntimeBootstrapError(RuntimeError):
    """A user-actionable failure while locating or extracting Julia."""


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    runtime_version: str
    platform: str
    julia_version: str
    runtime_directory: str
    archive_name: str
    release_tag: str
    release_url: str

    @classmethod
    def from_mapping(cls, value: object) -> "RuntimeConfig":
        if not isinstance(value, dict):
            raise RuntimeBootstrapError("Windows 运行环境配置不是有效的 JSON 对象。")
        try:
            config = cls(
                schema_version=int(value["schema_version"]),
                runtime_version=str(value["runtime_version"]),
                platform=str(value["platform"]),
                julia_version=str(value["julia_version"]),
                runtime_directory=str(value["runtime_directory"]),
                archive_name=str(value["archive_name"]),
                release_tag=str(value["release_tag"]),
                release_url=str(value["release_url"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeBootstrapError(f"Windows 运行环境配置缺少必要字段: {exc}") from exc

        string_values = (
            config.runtime_version,
            config.platform,
            config.julia_version,
            config.runtime_directory,
            config.archive_name,
            config.release_tag,
            config.release_url,
        )
        if config.schema_version <= 0 or any(not item.strip() for item in string_values):
            raise RuntimeBootstrapError("Windows 运行环境配置包含空值或无效版本号。")
        if Path(config.runtime_directory).name != config.runtime_directory:
            raise RuntimeBootstrapError("runtime_directory 必须是单层目录名。")
        if Path(config.archive_name).name != config.archive_name:
            raise RuntimeBootstrapError("archive_name 必须是单层文件名。")
        return config


@dataclass(frozen=True)
class RuntimePaths:
    base_dir: Path
    runtime_dir: Path
    archive_path: Path
    julia_exe: Path
    julia_depot: Path
    manifest_path: Path


def resource_root() -> Path:
    """Return the source root or PyInstaller's extracted resource directory."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def application_directory() -> Path:
    """Return the directory users placed the executable in."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def load_runtime_config(path: str | os.PathLike[str] | None = None) -> RuntimeConfig:
    config_path = Path(path) if path is not None else resource_root() / CONFIG_FILENAME
    try:
        with config_path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeBootstrapError(f"无法读取 Windows 运行环境配置: {config_path}\n{exc}") from exc
    return RuntimeConfig.from_mapping(value)


def runtime_paths(
    config: RuntimeConfig | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> RuntimePaths:
    config = config or load_runtime_config()
    base = Path(base_dir) if base_dir is not None else application_directory()
    runtime_dir = base / config.runtime_directory
    return RuntimePaths(
        base_dir=base,
        runtime_dir=runtime_dir,
        archive_path=base / config.archive_name,
        julia_exe=runtime_dir / "julia-runtime" / "bin" / "julia.exe",
        julia_depot=runtime_dir / "julia-depot",
        manifest_path=runtime_dir / MANIFEST_FILENAME,
    )


def _read_manifest(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeBootstrapError(f"Julia 环境清单无法读取: {path}\n{exc}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeBootstrapError(f"Julia 环境清单格式错误: {path}")
    return manifest


def validate_runtime_directory(
    runtime_dir: str | os.PathLike[str],
    config: RuntimeConfig,
    *,
    check_julia: bool = True,
) -> None:
    """Validate the marker, layout, and Julia executable for one runtime."""
    runtime_dir = Path(runtime_dir)
    manifest_path = runtime_dir / MANIFEST_FILENAME
    manifest = _read_manifest(manifest_path)
    expected = {
        "schema_version": config.schema_version,
        "runtime_version": config.runtime_version,
        "platform": config.platform,
        "julia_version": config.julia_version,
    }
    mismatches = [
        f"{key}={manifest.get(key)!r}（需要 {expected_value!r}）"
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    ]
    if mismatches:
        raise RuntimeBootstrapError("Julia 环境版本不匹配: " + "，".join(mismatches))

    julia_exe = runtime_dir / "julia-runtime" / "bin" / "julia.exe"
    julia_depot = runtime_dir / "julia-depot"
    if not julia_exe.is_file():
        raise RuntimeBootstrapError(f"Julia 可执行文件缺失: {julia_exe}")
    if not julia_depot.is_dir():
        raise RuntimeBootstrapError(f"Julia depot 缺失: {julia_depot}")
    if not check_julia:
        return

    try:
        result = subprocess.run(
            [str(julia_exe), "--startup-file=no", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeBootstrapError(f"Julia 无法启动: {julia_exe}\n{exc}") from exc
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeBootstrapError(f"Julia 版本检查失败（退出码 {result.returncode}）: {output}")
    if config.julia_version not in output:
        raise RuntimeBootstrapError(
            f"Julia 版本不匹配: {output or '未返回版本'}（需要 {config.julia_version}）"
        )


def _safe_member_path(staging_dir: Path, member: zipfile.ZipInfo, runtime_name: str) -> Path:
    normalized = member.filename.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != runtime_name
        or any(":" in part for part in relative.parts)
    ):
        raise RuntimeBootstrapError(f"环境压缩包包含不安全路径: {member.filename}")
    file_mode = (member.external_attr >> 16) & 0o170000
    if file_mode == stat.S_IFLNK:
        raise RuntimeBootstrapError(f"环境压缩包不允许包含符号链接: {member.filename}")

    destination = staging_dir.joinpath(*relative.parts)
    staging_resolved = staging_dir.resolve()
    destination_resolved = destination.resolve()
    try:
        destination_resolved.relative_to(staging_resolved)
    except ValueError as exc:
        raise RuntimeBootstrapError(f"环境压缩包路径越界: {member.filename}") from exc
    return destination


def _extract_archive(
    archive_path: Path,
    staging_dir: Path,
    config: RuntimeConfig,
    progress_callback: ProgressCallback | None,
) -> Path:
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RuntimeBootstrapError(
            f"Julia 环境压缩包损坏或无法读取: {archive_path}\n{exc}"
        ) from exc

    with archive:
        members = archive.infolist()
        destinations = [
            _safe_member_path(staging_dir, member, config.runtime_directory)
            for member in members
        ]
        total_bytes = sum(member.file_size for member in members if not member.is_dir())
        try:
            free_bytes = shutil.disk_usage(staging_dir.parent).free
        except OSError as exc:
            raise RuntimeBootstrapError(f"无法检查环境解压磁盘空间: {exc}") from exc
        if free_bytes < total_bytes:
            raise RuntimeBootstrapError(
                "磁盘空间不足，Julia 环境解压至少还需要 "
                f"{total_bytes / 1024 ** 3:.2f} GB 可用空间。"
            )

        completed = 0
        if progress_callback is not None:
            progress_callback(0, total_bytes, "正在解压 Julia 运行环境...")
        try:
            for member, destination in zip(members, destinations):
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    while True:
                        chunk = source.read(COPY_CHUNK_SIZE)
                        if not chunk:
                            break
                        target.write(chunk)
                        completed += len(chunk)
                        if progress_callback is not None:
                            progress_callback(completed, total_bytes, member.filename)
        except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
            raise RuntimeBootstrapError(f"Julia 环境解压失败: {exc}") from exc

    extracted_runtime = staging_dir / config.runtime_directory
    if not extracted_runtime.is_dir():
        raise RuntimeBootstrapError(
            f"环境压缩包缺少顶层目录 {config.runtime_directory}。"
        )
    return extracted_runtime


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _create_staging_directory(base_dir: Path) -> Path:
    """Create a short-lived staging directory with a Windows-friendly name."""
    for _ in range(10):
        staging_dir = base_dir / (
            f"{STAGING_DIRECTORY_PREFIX}{uuid.uuid4().hex[:STAGING_TOKEN_LENGTH]}"
        )
        try:
            staging_dir.mkdir()
        except FileExistsError:
            continue
        return staging_dir
    raise RuntimeBootstrapError(
        f"无法创建 Julia 环境临时目录: {base_dir}\n请关闭其他正在准备环境的程序后重试。"
    )


def ensure_windows_runtime(
    *,
    config: RuntimeConfig | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    check_julia: bool = True,
) -> RuntimePaths:
    """Reuse a valid runtime or atomically extract it from the sibling ZIP."""
    config = config or load_runtime_config()
    paths = runtime_paths(config, base_dir)
    existing_error: RuntimeBootstrapError | None = None
    try:
        validate_runtime_directory(paths.runtime_dir, config, check_julia=check_julia)
        return paths
    except RuntimeBootstrapError as exc:
        existing_error = exc

    if not paths.archive_path.is_file():
        detail = f"\n当前环境问题: {existing_error}" if paths.runtime_dir.exists() else ""
        raise RuntimeBootstrapError(
            f"没有可用的 Julia 运行环境。{detail}\n\n"
            f"请下载 {config.archive_name}，放到 NNLC_Trainer.exe 同一目录后重新运行。\n"
            f"下载页面: {config.release_url}"
        )

    try:
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = _create_staging_directory(paths.base_dir)
    except OSError as exc:
        raise RuntimeBootstrapError(
            f"EXE 所在目录不可写，无法准备 Julia 环境: {paths.base_dir}\n{exc}"
        ) from exc

    try:
        extracted_runtime = _extract_archive(
            paths.archive_path, staging_dir, config, progress_callback
        )
        validate_runtime_directory(
            extracted_runtime, config, check_julia=check_julia
        )
        backup_dir = paths.base_dir / (
            f".{config.runtime_directory}.backup-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            if paths.runtime_dir.exists() or paths.runtime_dir.is_symlink():
                os.replace(paths.runtime_dir, backup_dir)
            os.replace(extracted_runtime, paths.runtime_dir)
        except OSError as exc:
            if backup_dir.exists() and not paths.runtime_dir.exists():
                try:
                    os.replace(backup_dir, paths.runtime_dir)
                except OSError:
                    pass
            raise RuntimeBootstrapError(
                f"无法安装 Julia 环境到 {paths.runtime_dir}: {exc}"
            ) from exc
        try:
            _remove_path(backup_dir)
        except OSError:
            pass
        if progress_callback is not None:
            progress_callback(1, 1, "Julia 运行环境准备完成")
        return paths
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
