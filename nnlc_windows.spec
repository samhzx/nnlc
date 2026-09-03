"""PyInstaller spec for the self-contained Windows NNLC trainer.

The build script stages a Windows Julia distribution and its package depot in
the project root before invoking this file.  The resulting one-dir bundle is
large (Julia + Flux/Plots artifacts are intentionally included) but does not
require Python, Julia, or any package installation on the target computer.
"""

from pathlib import Path
import sys
import sysconfig

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

    Julia runtime and depot files stay complete because package source files
    outside the usual test directories can still be required during Julia
    precompilation.
    """
    def is_runtime_file(path: Path) -> bool:
        relative_path = path.relative_to(source_dir)
        relative_parts = {part.lower() for part in relative_path.parts}
        target_parts = {part.lower() for part in Path(target_dir).parts}

        # Debug symbols are not loaded by the bundled Julia process.  They are
        # only useful when debugging Julia itself.
        if path.suffix.lower() == ".pdb":
            return False

        # Keep the Julia runtime and package depot intact.  Only transient
        # depot caches are removed by the build script before this runs.
        if target_parts == {"julia-runtime"}:
            return not path.name.lower().endswith((".cmake", ".cmake.in")) and "cmake" not in relative_parts
        if target_parts == {"julia-depot"}:
            top_level = relative_path.parts[0].lower() if relative_path.parts else ""
            if top_level in {"scratchspaces", "logs", "clones"}:
                return False
            # Static archives in package artifacts are not loaded at runtime
            # and can be omitted from the depot copy.
            if path.suffix.lower() in {".a", ".la"}:
                return False

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
        if path.is_file() and is_runtime_file(path)
    ]


def filter_python_dev_files(entries):
    """Drop package tests/examples from PyInstaller-collected files.

    ``collect_all`` intentionally collects broad package data.  The trainer
    never imports these development-only trees, so excluding them keeps the
    Windows bundle smaller without affecting runtime modules or plotting data.
    """
    excluded_parts = {
        "test", "tests", "testing", "example", "examples",
        "benchmark", "benchmarks",
    }
    filtered = []
    for entry in entries:
        source = Path(str(entry[0]))
        if any(part.lower() in excluded_parts for part in source.parts):
            continue
        filtered.append(entry)
    return filtered


def standard_library_extensions(module_name: str):
    """Collect Windows stdlib extension modules used by runtime hooks.

    PyInstaller normally discovers these through imports, but the
    multiprocessing runtime hook can execute before the regular import graph
    is restored. Explicitly bundling ``_socket.pyd`` prevents a one-dir build
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
    *directory_datas(JULIA_RUNTIME, "julia-runtime"),
    *directory_datas(JULIA_DEPOT, "julia-depot"),
]
binaries = standard_library_extensions("_socket")
hiddenimports = [
    "_socket",
    "socket",
    "multiprocessing",
    "multiprocessing.context",
    "multiprocessing.reduction",
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
        datas += filter_python_dev_files(d)
        binaries += filter_python_dev_files(b)
        hiddenimports += [
            module for module in h
            if not any(part.lower() in {"test", "tests", "testing", "example", "examples", "benchmark", "benchmarks"}
                       for part in module.split("."))
        ]
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
