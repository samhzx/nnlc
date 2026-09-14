"""Application version helpers for source builds and the frozen Windows EXE."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
WINDOWS_VERSION_LIMIT = 65535
BUILD_INFO_NAME = "build_info.json"


class VersionError(RuntimeError):
    """Raised when the application version cannot be read or validated."""


def parse_version(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value.strip()):
        raise VersionError(f"版本号无效: {value!r}")
    major, minor, patch = (int(part) for part in value.strip().split("."))
    for part in (major, minor, patch):
        if part > WINDOWS_VERSION_LIMIT:
            raise VersionError(f"版本号超出 Windows 文件版本范围: {value}")
    return major, minor, patch


def format_windows_file_version(version: str) -> str:
    major, minor, patch = parse_version(version)
    return f"{major}.{minor}.{patch}.0"


def windows_file_version_tuple(version: str) -> tuple[int, int, int, int]:
    major, minor, patch = parse_version(version)
    return major, minor, patch, 0


def read_pyproject_version(path: str | Path) -> str:
    try:
        import tomllib
    except ImportError as exc:  # pragma: no cover - Python 3.11 is required
        raise VersionError("读取 pyproject.toml 需要 Python 3.11+") from exc

    pyproject_path = Path(path)
    try:
        with pyproject_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError) as exc:
        raise VersionError(f"无法读取版本文件: {pyproject_path}\n{exc}") from exc

    try:
        version = data["project"]["version"]
    except (KeyError, TypeError) as exc:
        raise VersionError(f"{pyproject_path} 缺少 project.version") from exc
    parse_version(version)
    return str(version)


def read_build_info_version(path: str | Path) -> str:
    build_info_path = Path(path)
    try:
        with build_info_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise VersionError(f"无法读取版本信息: {build_info_path}\n{exc}") from exc
    if not isinstance(data, dict) or "version" not in data:
        raise VersionError(f"版本信息缺少 version 字段: {build_info_path}")
    version = data["version"]
    parse_version(version)
    return str(version)


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent


def get_version() -> str:
    if getattr(sys, "frozen", False):
        return read_build_info_version(resource_root() / BUILD_INFO_NAME)
    return read_pyproject_version(Path(__file__).resolve().parent / "pyproject.toml")
