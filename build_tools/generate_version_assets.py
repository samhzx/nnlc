"""Generate frozen version resources from pyproject.toml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from nnlc_version import (
    format_windows_file_version,
    parse_version,
    read_pyproject_version,
    windows_file_version_tuple,
)


def version_info_text(version: str) -> str:
    file_version = format_windows_file_version(version)
    filevers = windows_file_version_tuple(version)
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={filevers},
    prodvers={filevers},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'FileDescription', u'NNLC Trainer'),
        StringStruct(u'FileVersion', u'{file_version}'),
        StringStruct(u'InternalName', u'NNLC_Trainer'),
        StringStruct(u'OriginalFilename', u'NNLC_Trainer.exe'),
        StringStruct(u'ProductName', u'NNLC Trainer'),
        StringStruct(u'ProductVersion', u'{version}')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""


def generate_version_assets(pyproject: str | Path, output_dir: str | Path) -> dict[str, Path]:
    version = read_pyproject_version(pyproject)
    parse_version(version)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    build_info_path = destination / "build_info.json"
    version_txt_path = destination / "version.txt"
    version_info_path = destination / "version_info.txt"

    build_info_path.write_text(
        json.dumps({"version": version}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    version_txt_path.write_text(version + "\n", encoding="utf-8")
    version_info_path.write_text(version_info_text(version), encoding="utf-8")
    return {
        "build_info": build_info_path,
        "version_txt": version_txt_path,
        "version_info": version_info_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate NNLC version build assets")
    parser.add_argument(
        "--pyproject",
        default=str(PROJECT_DIR / "pyproject.toml"),
        help="Path to pyproject.toml",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_DIR / "build" / "version-assets"),
        help="Directory for generated version files",
    )
    args = parser.parse_args(argv)
    paths = generate_version_assets(args.pyproject, args.output_dir)
    print(f"Wrote {paths['version_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
