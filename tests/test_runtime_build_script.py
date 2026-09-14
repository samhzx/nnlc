from pathlib import Path


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build_julia_runtime.ps1"


def test_runtime_build_preserves_julia_package_trees():
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Remove-JuliaPackageDevelopmentFiles" not in script
    for directory_name in ("example", "examples", "test", "tests"):
        assert f"-xr!{directory_name}" not in script
