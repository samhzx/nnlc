"""Create the five-field Windows update manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nnlc_update import parse_manifest
from nnlc_version import parse_version


def create_update_manifest(
    exe_path: str | Path,
    version: str,
    notes: str,
    output_path: str | Path,
) -> dict:
    parse_version(version)
    source = Path(exe_path)
    if not source.is_file():
        raise SystemExit(f"missing release executable: {source}")
    expected_name = f"NNLC_Trainer-{version}-windows-x64.exe"
    if source.name != expected_name:
        raise SystemExit(f"release executable name must be {expected_name}, got {source.name}")

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    payload = {
        "version": version,
        "size": source.stat().st_size,
        "filename": f"NNLC_Trainer-{version}-windows-x64.exe",
        "notes": notes,
        "sha256": digest.hexdigest(),
    }
    parse_manifest(payload)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create NNLC update.json")
    parser.add_argument("--exe", required=True, help="Path to the versioned EXE")
    parser.add_argument("--version", required=True, help="Application version")
    parser.add_argument("--notes", default="", help="Update notes")
    parser.add_argument("--output", required=True, help="Path to write update.json")
    args = parser.parse_args(argv)
    payload = create_update_manifest(args.exe, args.version, args.notes, args.output)
    print(payload["filename"])
    print(payload["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
