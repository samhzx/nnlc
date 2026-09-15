import json
import os
import zipfile
from pathlib import Path

import pytest

from nnlc_runtime import (
    MANIFEST_FILENAME,
    RuntimeBootstrapError,
    RuntimeConfig,
    _create_staging_directory,
    ensure_windows_runtime,
    validate_runtime_directory,
)


CONFIG = RuntimeConfig(
    schema_version=1,
    runtime_version="1",
    platform="windows-x64",
    julia_version="1.10.11",
    runtime_directory="NNLC_Runtime",
    archive_name="NNLC_Julia_Runtime-windows-x64-v1.zip",
    release_tag="julia-runtime-v1",
    release_url="https://example.test/runtime",
)


def runtime_manifest(**overrides):
    manifest = {
        "schema_version": CONFIG.schema_version,
        "runtime_version": CONFIG.runtime_version,
        "platform": CONFIG.platform,
        "julia_version": CONFIG.julia_version,
    }
    manifest.update(overrides)
    return manifest


def create_runtime(root: Path, **manifest_overrides) -> Path:
    runtime = root / CONFIG.runtime_directory
    (runtime / "julia-runtime" / "bin").mkdir(parents=True)
    (runtime / "julia-runtime" / "bin" / "julia.exe").write_bytes(b"fake")
    (runtime / "julia-depot").mkdir()
    (runtime / MANIFEST_FILENAME).write_text(
        json.dumps(runtime_manifest(**manifest_overrides)),
        encoding="utf-8",
    )
    return runtime


def create_archive(base_dir: Path, source_root: Path) -> Path:
    archive_path = base_dir / CONFIG.archive_name
    runtime = source_root / CONFIG.runtime_directory
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in runtime.rglob("*"):
            archive.write(path, path.relative_to(source_root))
    return archive_path


def extraction_staging_dirs(base_dir: Path):
    return list(base_dir.glob(".nnlc-*"))


def test_extraction_staging_directory_has_a_short_unique_name(tmp_path):
    first = _create_staging_directory(tmp_path)
    second = _create_staging_directory(tmp_path)

    assert first != second
    assert first.name.startswith(".nnlc-")
    assert len(first.name) == len(".nnlc-") + 8

    first.rmdir()
    second.rmdir()


def test_valid_runtime_does_not_require_archive(tmp_path):
    runtime = create_runtime(tmp_path)

    paths = ensure_windows_runtime(
        config=CONFIG, base_dir=tmp_path, check_julia=False
    )

    assert paths.runtime_dir == runtime
    assert not paths.archive_path.exists()


def test_missing_runtime_extracts_archive_atomically(tmp_path):
    source_root = tmp_path / "archive-source"
    create_runtime(source_root)
    create_archive(tmp_path, source_root)

    paths = ensure_windows_runtime(
        config=CONFIG, base_dir=tmp_path, check_julia=False
    )

    assert paths.julia_exe.read_bytes() == b"fake"
    assert paths.manifest_path.is_file()
    assert not extraction_staging_dirs(tmp_path)


def test_missing_runtime_and_archive_reports_download(tmp_path):
    with pytest.raises(RuntimeBootstrapError) as exc_info:
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )

    message = str(exc_info.value)
    assert CONFIG.archive_name in message
    assert CONFIG.release_url in message


def test_mismatched_existing_runtime_is_replaced_by_matching_archive(tmp_path):
    create_runtime(tmp_path, runtime_version="old")
    source_root = tmp_path / "archive-source"
    create_runtime(source_root)
    create_archive(tmp_path, source_root)

    paths = ensure_windows_runtime(
        config=CONFIG, base_dir=tmp_path, check_julia=False
    )

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["runtime_version"] == CONFIG.runtime_version


def test_mismatched_runtime_without_archive_is_rejected(tmp_path):
    create_runtime(tmp_path, runtime_version="old")

    with pytest.raises(RuntimeBootstrapError, match="版本不匹配"):
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )


def test_julia_version_output_must_match_manifest(tmp_path, monkeypatch):
    runtime = create_runtime(tmp_path)

    class Result:
        returncode = 0
        stdout = "julia version 1.10.10"
        stderr = ""

    monkeypatch.setattr("nnlc_runtime.subprocess.run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeBootstrapError, match="版本不匹配"):
        validate_runtime_directory(runtime, CONFIG)


def test_corrupt_archive_leaves_no_accepted_runtime(tmp_path):
    (tmp_path / CONFIG.archive_name).write_bytes(b"not a zip")

    with pytest.raises(RuntimeBootstrapError, match="损坏"):
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )

    assert not (tmp_path / CONFIG.runtime_directory).exists()
    assert not extraction_staging_dirs(tmp_path)


def test_archive_path_traversal_is_rejected(tmp_path):
    archive_path = tmp_path / CONFIG.archive_name
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.txt", "bad")

    with pytest.raises(RuntimeBootstrapError, match="不安全路径"):
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )

    assert not (tmp_path / "outside.txt").exists()
    assert not extraction_staging_dirs(tmp_path)


def test_failed_install_restores_previous_runtime(tmp_path, monkeypatch):
    existing = create_runtime(tmp_path, runtime_version="old")
    source_root = tmp_path / "archive-source"
    create_runtime(source_root)
    create_archive(tmp_path, source_root)

    real_replace = os.replace
    replace_calls = 0

    def fail_new_runtime(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise PermissionError("destination locked")
        return real_replace(source, destination)

    monkeypatch.setattr("nnlc_runtime.os.replace", fail_new_runtime)

    with pytest.raises(RuntimeBootstrapError, match="无法安装"):
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )

    manifest = json.loads((existing / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["runtime_version"] == "old"
    assert not extraction_staging_dirs(tmp_path)


def test_interrupted_extraction_does_not_replace_existing_runtime(tmp_path, monkeypatch):
    existing = create_runtime(tmp_path, runtime_version="old")
    source_root = tmp_path / "archive-source"
    create_runtime(source_root)
    create_archive(tmp_path, source_root)

    def interrupt(_completed, _total, filename):
        if filename.endswith("julia.exe"):
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeBootstrapError, match="解压失败"):
        ensure_windows_runtime(
            config=CONFIG,
            base_dir=tmp_path,
            progress_callback=interrupt,
            check_julia=False,
        )

    manifest = json.loads((existing / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["runtime_version"] == "old"
    assert not extraction_staging_dirs(tmp_path)


def test_unwritable_base_directory_reports_clear_error(tmp_path, monkeypatch):
    def fail_mkdir(self, *args, **kwargs):
        if self.name.startswith(".nnlc-"):
            raise PermissionError("read only")
        return original_mkdir(self, *args, **kwargs)

    source_root = tmp_path / "archive-source"
    create_runtime(source_root)
    create_archive(tmp_path, source_root)
    original_mkdir = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(RuntimeBootstrapError, match="不可写"):
        ensure_windows_runtime(
            config=CONFIG, base_dir=tmp_path, check_julia=False
        )
