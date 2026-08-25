"""PyInstaller spec for the self-contained Windows NNLC trainer.

The build script stages a Windows Julia distribution and its package depot in
the project root before invoking this file.  The resulting one-file exe is
large (Julia + Flux/Plots artifacts are intentionally included) but does not
require Python, Julia, or any package installation on the target computer.
"""

from pathlib import Path

from PyInstaller.building.build_main import Analysis, EXE, PYZ, COLLECT
from PyInstaller.building.datastruct import Tree
from PyInstaller.utils.hooks import collect_all, collect_submodules


PROJECT_DIR = Path(SPECPATH).resolve()
JULIA_RUNTIME = PROJECT_DIR / "julia-runtime"
JULIA_DEPOT = PROJECT_DIR / "julia-depot"
if not JULIA_RUNTIME.is_dir():
    raise SystemExit(f"Missing {JULIA_RUNTIME}; run build_windows.ps1 first")
if not JULIA_DEPOT.is_dir():
    raise SystemExit(f"Missing {JULIA_DEPOT}; run build_windows.ps1 first")

datas = [
    *Tree(str(PROJECT_DIR / "training"), prefix="training"),
    *Tree(str(PROJECT_DIR / "nnlc_tools" / "cereal"), prefix="nnlc_tools/cereal"),
    *Tree(str(JULIA_RUNTIME), prefix="julia-runtime"),
    *Tree(str(JULIA_DEPOT), prefix="julia-depot"),
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
    a.binaries,
    a.datas,
    [],
    name="NNLC_Trainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=None,
)
