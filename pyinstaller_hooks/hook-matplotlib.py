"""Trim Matplotlib's bundled data to files used by the trainer."""

import os
from pathlib import Path

from PyInstaller import compat, isolated
from PyInstaller.utils import hooks as hookutils


@isolated.decorate
def mpl_data_dir():
    import matplotlib

    return matplotlib.get_data_path()


_excluded_parts = {"sample_data", "plot_directive", "tests", "testing"}
_data_root = Path(mpl_data_dir())
datas = []
for source in _data_root.rglob("*"):
    if not source.is_file():
        continue
    relative = source.relative_to(_data_root)
    if (
        source.suffix.lower() in {".pyi", ".pyc"}
        or "__pycache__" in relative.parts
        or any(part in _excluded_parts for part in relative.parts)
    ):
        continue
    datas.append((str(source), str(Path("matplotlib/mpl-data") / relative.parent)))

binaries = []
if compat.is_win and hookutils.check_requirement("matplotlib >= 3.7.0"):
    delvewheel_datas, delvewheel_binaries = hookutils.collect_delvewheel_libs_directory(
        "matplotlib"
    )
    datas += delvewheel_datas
    binaries += delvewheel_binaries
