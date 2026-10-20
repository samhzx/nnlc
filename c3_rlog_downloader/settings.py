"""Per-user settings isolated from the NNLC application."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_DIRECTORY_NAME = "C3_Rlog_Downloader"


def settings_directory() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / APP_DIRECTORY_NAME


def load_settings(directory: Path | None = None) -> dict:
    path = (directory or settings_directory()) / "settings.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        "host": value.get("host", "") if isinstance(value.get("host", ""), str) else "",
        "key_path": value.get("key_path", "")
        if isinstance(value.get("key_path", ""), str)
        else "",
        "destination": value.get("destination", "")
        if isinstance(value.get("destination", ""), str)
        else "",
    }


def default_destination() -> str:
    return str(Path.home() / "Documents" / "C3_Rlogs")


def save_settings(
    host: str,
    key_path: str,
    destination: str = "",
    directory: Path | None = None,
) -> None:
    target_dir = directory or settings_directory()
    existing = load_settings(target_dir)
    destination_value = destination.strip() or existing.get("destination", "")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "settings.json"
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "host": host.strip(),
                "key_path": key_path.strip(),
                "destination": destination_value,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
