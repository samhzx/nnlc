"""PyInstaller spec for the one-file Windows NNLC trainer.

Python, project code, training scripts, and schemas live in the EXE. The large
Julia runtime and depot are distributed once as a separate reusable ZIP.
"""

from pathlib import Path
import sys
import sysconfig

from PyInstaller.building.build_main import Analysis, EXE, PYZ
from PyInstaller.utils.hooks import collect_all, collect_submodules


PROJECT_DIR = Path(SPECPATH).resolve()

def directory_datas(source_dir: Path, target_dir: str):
    """Return resource files using Analysis(datas=...)'s 2-tuples.

    The second item in each tuple is a destination directory, not a complete
    destination filename.  Including the filename there creates an extra
    directory layer such as ``bin/julia.exe/julia.exe``.
    """
    def is_resource_file(path: Path) -> bool:
        relative_path = path.relative_to(source_dir)
        relative_parts = {part.lower() for part in relative_path.parts}

        if "cmake" in relative_parts:
            return False
        if path.name.lower() == "cmakelists.txt":
            return False
        if path.name.lower().endswith((".cmake", ".cmake.in")):
            return False
        return True

    return [
        (
            str(path),
            str(Path(target_dir) / path.relative_to(source_dir).parent),
        )
        for path in source_dir.rglob("*")
        if path.is_file() and is_resource_file(path)
    ]


def standard_library_extensions(module_name: str):
    """Collect Windows stdlib extension modules used by runtime hooks.

    PyInstaller normally discovers these through imports, but the
    multiprocessing runtime hook can execute before the regular import graph
    is restored. Explicitly bundling ``_socket.pyd`` prevents the frozen app
    from starting with ``No module named '_socket'``.
    """
    roots = set()
    dest_shared = sysconfig.get_config_var("DESTSHARED")
    platstdlib = sysconfig.get_path("platstdlib")
    if dest_shared:
        roots.add(Path(dest_shared))
    if platstdlib:
        roots.add(Path(platstdlib) / "lib-dynload")
    roots.add(Path(sys.base_prefix) / "DLLs")
    roots.add(Path(sys.executable).parent)
    roots.add(Path(sys.executable).parent / "DLLs")
    binaries = []
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob(f"{module_name}*.pyd"):
            resolved = path.resolve()
            if resolved not in seen:
                binaries.append((str(path), "."))
                seen.add(resolved)
    if not binaries:
        raise SystemExit(
            f"Could not locate {module_name}.pyd in the Python standard library; "
            "the Windows bundle would be incomplete"
        )
    return binaries


datas = [
    *directory_datas(PROJECT_DIR / "training", "training"),
    *directory_datas(PROJECT_DIR / "nnlc_tools" / "cereal", "nnlc_tools/cereal"),
    (str(PROJECT_DIR / "windows_runtime.json"), "."),
]
binaries = standard_library_extensions("_socket")
hiddenimports = [
    "_socket",
    "socket",
    "multiprocessing",
    "multiprocessing.context",
    "multiprocessing.reduction",
    "nnlc_gui",
    "nnlc_runtime",
    "nnlc_tools",
    "nnlc_tools.logreader",
    "nnlc_tools.extract_lateral_data",
    "nnlc_tools.extract_rlog_worker",
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
