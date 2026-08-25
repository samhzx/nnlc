"""PyInstaller spec for the self-contained Windows NNLC trainer.

The build script stages a Windows Julia distribution and its package depot in
the project root before invoking this file.  The resulting one-dir bundle is
large (Julia + Flux/Plots artifacts are intentionally included) but does not
require Python, Julia, or any package installation on the target computer.
"""

from pathlib import Path

from PyInstaller.building.build_main import Analysis, EXE, PYZ, COLLECT
from PyInstaller.utils.hooks import collect_all, collect_submodules


PROJECT_DIR = Path(SPECPATH).resolve()
JULIA_RUNTIME = PROJECT_DIR / "julia-runtime"
JULIA_DEPOT = PROJECT_DIR / "julia-depot"
if not JULIA_RUNTIME.is_dir():
    raise SystemExit(f"Missing {JULIA_RUNTIME}; run build_windows.ps1 first")
if not JULIA_DEPOT.is_dir():
    raise SystemExit(f"Missing {JULIA_DEPOT}; run build_windows.ps1 first")

def directory_datas(source_dir: Path, target_dir: str):
    """Return runtime files using Analysis(datas=...)'s 2-tuples.

    The second item in each tuple is a destination directory, not a complete
    destination filename.  Including the filename there creates an extra
    directory layer such as ``bin/julia.exe/julia.exe``.

    Julia artifacts contain CMake build metadata under ``lib/cmake``.  Those
    files are not read when loading the precompiled Julia packages, and their
    very deep names can exceed Windows' extraction path limit in a one-file
    PyInstaller executable.
    """
    def is_runtime_file(path: Path) -> bool:
        relative_path = path.relative_to(source_dir)
        relative_parts = {part.lower() for part in relative_path.parts}
        target_parts = {part.lower() for part in Path(target_dir).parts}
        if "cmake" in relative_parts:
            return False
        if path.name.lower() == "cmakelists.txt":
            return False
        if path.name.lower().endswith((".cmake", ".cmake.in")):
            return False
        # Julia's runtime test suite and documentation are not needed by the
        # trainer and account for many unnecessary files in the bundle.
        if target_parts == {"julia-runtime"} and relative_parts.intersection(
            {"test", "tests", "doc", "docs", "man"}
        ):
            return False
        # Package tests/examples/benchmarks are development-only content.
        if target_parts == {"julia-depot"} and relative_path.parts:
            top_level = relative_path.parts[0].lower()
            if top_level in {"scratchspaces", "logs", "clones"}:
                return False
            if top_level == "packages" and relative_parts.intersection(
                {"test", "tests", "docs", "examples", "benchmark", "benchmarks"}
            ):
                return False
        return True

    return [
        (
            str(path),
            str(Path(target_dir) / path.relative_to(source_dir).parent),
        )
        for path in source_dir.rglob("*")
        if path.is_file() and is_runtime_file(path)
    ]


datas = [
    *directory_datas(PROJECT_DIR / "training", "training"),
    *directory_datas(PROJECT_DIR / "nnlc_tools" / "cereal", "nnlc_tools/cereal"),
    *directory_datas(JULIA_RUNTIME, "julia-runtime"),
    *directory_datas(JULIA_DEPOT, "julia-depot"),
]
binaries = []
hiddenimports = [
    "nnlc_gui",
    "nnlc_tools",
    "nnlc_tools.logreader",
    "nnlc_tools.extract_lateral_data",
    "nnlc_tools.score_routes",
    "nnlc_tools.prune_routes",
    "nnlc_tools.analyze_interventions",
    "nnlc_tools.visualize_coverage",
    "nnlc_tools.visualize_model",
    "nnlc_tools.steering_classifier",
]

for package in ("numpy", "pandas", "matplotlib", "scipy", "zstandard", "capnp", "tqdm"):
    try:
        d, b, h = collect_all(package)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Some package names differ between import name and distribution name;
        # PyInstaller's normal analysis will still report a useful error.
        hiddenimports += collect_submodules(package)

hiddenimports += collect_submodules("nnlc_tools")

a = Analysis(
    [str(PROJECT_DIR / "nnlc_auto_train.py")],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NNLC_Trainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="NNLC_Trainer",
)
