"""PyInstaller spec for the one-file Windows NNLC trainer.

Python, project code, training scripts, and schemas live in the EXE. The large
Julia runtime and depot are distributed once as a separate reusable ZIP.
"""

from pathlib import Path
import sys
import sysconfig

from PyInstaller.building.build_main import Analysis, EXE, PYZ


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
    extension_patterns = (
        [f"{module_name}*.pyd"]
        if sys.platform == "win32"
        else [f"{module_name}*.so", f"{module_name}*.dylib"]
    )
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in extension_patterns:
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved not in seen:
                    binaries.append((str(path), "."))
                    seen.add(resolved)
    if not binaries:
        raise SystemExit(
            f"Could not locate {module_name} extension in the Python standard library; "
            "the Windows bundle would be incomplete"
        )
    return binaries


VERSION_ASSETS_DIR = PROJECT_DIR / "build" / "version-assets"
BUILD_INFO_PATH = VERSION_ASSETS_DIR / "build_info.json"
VERSION_INFO_PATH = VERSION_ASSETS_DIR / "version_info.txt"
if not BUILD_INFO_PATH.is_file() or not VERSION_INFO_PATH.is_file():
    raise SystemExit(
        "Missing generated version assets. Run "
        "`python -m build_tools.generate_version_assets` before PyInstaller."
    )

datas = [
    *directory_datas(PROJECT_DIR / "training", "training"),
    *directory_datas(PROJECT_DIR / "nnlc_tools" / "cereal", "nnlc_tools/cereal"),
    (str(PROJECT_DIR / "windows_runtime.json"), "."),
    (str(BUILD_INFO_PATH), "."),
]
binaries = standard_library_extensions("_socket")
hiddenimports = [
    "_socket",
    "socket",
    "numpy.testing",
    "multiprocessing",
    "multiprocessing.context",
    "multiprocessing.reduction",
    "nnlc_gui",
    "nnlc_runtime",
    "nnlc_update",
    "nnlc_version",
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
    "nnlc_tools.steering_classifier.cascade",
    "nnlc_tools.steering_classifier.config",
    "nnlc_tools.steering_classifier.features",
    "nnlc_tools.steering_classifier.filters",
    "nnlc_tools.steering_classifier.types",
]

a = Analysis(
    [str(PROJECT_DIR / "nnlc_auto_train.py")],
    pathex=[str(PROJECT_DIR)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PROJECT_DIR / "pyinstaller_hooks")],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[],
    excludes=[
        "tkinter.test",
        "numpy.tests",
        "pandas.tests",
        "matplotlib.tests",
        "matplotlib.testing",
        "matplotlib.backends._backend_gtk",
        "matplotlib.backends.backend_gtk3",
        "matplotlib.backends.backend_gtk3agg",
        "matplotlib.backends.backend_macosx",
        "matplotlib.backends.backend_qt",
        "matplotlib.backends.backend_qt5",
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_wx",
        "matplotlib.backends.backend_wxagg",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.backends._backend_tk",
    ],
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
    version=str(VERSION_INFO_PATH),
)
