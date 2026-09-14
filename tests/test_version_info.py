import hashlib
import json
import sys
from pathlib import Path

import pytest

from build_tools.create_update_manifest import create_update_manifest
from build_tools.generate_version_assets import generate_version_assets
from nnlc_version import (
    VersionError,
    format_windows_file_version,
    get_version,
    parse_version,
    read_build_info_version,
    read_pyproject_version,
)


def test_parse_version_accepts_semver():
    assert parse_version("1.0.0") == (1, 0, 0)
    assert parse_version("1.10.2") == (1, 10, 2)


def test_parse_version_rejects_invalid_values():
    for value in ("1.0.0.0", "v1.0.0", "1.0", "01.0.0", "1.0.0-rc.1", ""):
        with pytest.raises(VersionError):
            parse_version(value)


def test_version_comparison_uses_numeric_parts():
    assert parse_version("1.10.0") > parse_version("1.9.9")


def test_windows_file_version_mapping():
    assert format_windows_file_version("1.2.3") == "1.2.3.0"


def test_read_pyproject_and_build_info(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "nnlc-tools"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    build_info = tmp_path / "build_info.json"
    build_info.write_text(json.dumps({"version": "1.2.3"}), encoding="utf-8")

    assert read_pyproject_version(pyproject) == "1.2.3"
    assert read_build_info_version(build_info) == "1.2.3"


def test_missing_build_info_fails(tmp_path):
    with pytest.raises(VersionError):
        read_build_info_version(tmp_path / "missing.json")


def test_generate_version_assets(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "nnlc-tools"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "assets"
    paths = generate_version_assets(pyproject, output_dir)

    assert paths["version_txt"].read_text(encoding="utf-8").strip() == "1.2.3"
    assert json.loads(paths["build_info"].read_text(encoding="utf-8")) == {"version": "1.2.3"}
    version_info = paths["version_info"].read_text(encoding="utf-8")
    assert "filevers=(1, 2, 3, 0)" in version_info
    assert "FileVersion', u'1.2.3.0'" in version_info
    assert "ProductVersion', u'1.2.3'" in version_info


def test_get_version_from_source_tree():
    assert get_version() == read_pyproject_version(
        Path(__file__).resolve().parents[1] / "pyproject.toml"
    )


def test_create_update_manifest_writes_fixed_fields(tmp_path):
    exe = tmp_path / "NNLC_Trainer-1.2.3-windows-x64.exe"
    payload_bytes = b"exe-bytes"
    exe.write_bytes(payload_bytes)
    output = tmp_path / "update.json"
    payload = create_update_manifest(exe, "1.2.3", "修复问题", output)
    raw = output.read_text(encoding="utf-8")
    assert raw.startswith('{"version":"1.2.3","size":')
    assert payload["filename"] == "NNLC_Trainer-1.2.3-windows-x64.exe"
    assert payload["size"] == len(payload_bytes)
    assert payload["notes"] == "修复问题"
    assert payload["sha256"] == hashlib.sha256(payload_bytes).hexdigest()


def test_create_update_manifest_rejects_mismatched_filename(tmp_path):
    exe = tmp_path / "NNLC_Trainer.exe"
    exe.write_bytes(b"exe-bytes")
    with pytest.raises(SystemExit, match="name must be"):
        create_update_manifest(exe, "1.2.3", "", tmp_path / "update.json")


def test_version_and_help_skip_runtime_bootstrap(monkeypatch, capsys):
    import nnlc_auto_train

    def fail_bootstrap(**kwargs):
        raise AssertionError("should not prepare Julia runtime")

    monkeypatch.setattr(nnlc_auto_train, "_prepare_bundled_runtime", fail_bootstrap)

    monkeypatch.setattr(sys, "argv", ["nnlc_auto_train.py", "--version"])
    with pytest.raises(SystemExit) as exc:
        nnlc_auto_train.main()
    assert exc.value.code in (0, None)
    assert capsys.readouterr().out.strip() == get_version()

    monkeypatch.setattr(sys, "argv", ["nnlc_auto_train.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        nnlc_auto_train.main()
    assert exc.value.code in (0, None)
    captured = capsys.readouterr()
    assert "NNLC 横向控制模型一键训练" in captured.out
