import ast
from pathlib import Path


BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build_julia_runtime.ps1"


def test_runtime_build_preserves_runtime_package_trees():
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert "Remove-JuliaPackageOptionalDirectories" in script
    assert "Remove-UnreadableJuliaDepotItems" in script
    for directory_name in ("example", "examples", "test", "tests"):
        assert f"-xr!{directory_name}" not in script
        assert f'"{directory_name}"' not in script


def test_windows_bundle_excludes_scipy():
    spec_path = BUILD_SCRIPT.parent / "nnlc_windows.spec"
    tree = ast.parse(spec_path.read_text(encoding="utf-8"))
    analysis = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Analysis"
    )
    excludes = next(keyword.value for keyword in analysis.keywords if keyword.arg == "excludes")
    assert "scipy" in ast.literal_eval(excludes)
