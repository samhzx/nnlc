"""PyInstaller one-file build for the independent C3 downloader."""

from pathlib import Path
import sys

from PyInstaller.building.build_main import Analysis, EXE, PYZ
from PyInstaller.utils.hooks import collect_data_files, collect_submodules


SOURCE_DIR = Path(SPECPATH).resolve()
REPO_ROOT = SOURCE_DIR.parent
ICON_PATH = SOURCE_DIR / "C3_Rlog_Downloader.ico"
PAYLOAD_DIR = SOURCE_DIR / "nnlc_payloads"

hiddenimports = collect_submodules("paramiko")
datas = [
    *collect_data_files("paramiko"),
    (str(ICON_PATH), "."),
    (str(PAYLOAD_DIR / "nnlc_18d.py"), "nnlc_payloads"),
    (str(PAYLOAD_DIR / "nnlc_23d.py"), "nnlc_payloads"),
    (str(PAYLOAD_DIR / "nnlc.py"), "nnlc_payloads"),
]

a = Analysis(
    [str(SOURCE_DIR / "main.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "numpy",
        "pandas",
        "matplotlib",
        "pycapnp",
        "zstandard",
        "julia",
        "nnlc_auto_train",
        "nnlc_gui",
        "nnlc_runtime",
        "nnlc_tools",
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
    name="C3_Rlog_Downloader",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ICON_PATH) if sys.platform == "win32" else None,
)
