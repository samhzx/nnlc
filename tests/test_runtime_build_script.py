from pathlib import Path


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build_julia_runtime.ps1"


def test_runtime_build_preserves_runtime_package_trees():
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Remove-JuliaPackageOptionalDirectories" in script
    assert "Remove-UnreadableJuliaDepotItems" in script
    for directory_name in ("example", "examples", "test", "tests"):
        assert f"-xr!{directory_name}" not in script
        assert f'"{directory_name}"' not in script
