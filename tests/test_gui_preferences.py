import json
import os
from pathlib import Path

from nnlc_gui import NNLCApp


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def test_saved_path_normalizes_strings_and_ignores_invalid_values():
    assert NNLCApp._saved_path("  ~/nnlc  ") == os.path.normpath(os.path.expanduser("~/nnlc"))
    assert NNLCApp._saved_path("   ") == ""
    assert NNLCApp._saved_path(None) == ""
    assert NNLCApp._saved_path(123) == ""


def test_load_preferences_reads_directory_paths(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({
            "car": " BYD_TANG_DMI_24 ",
            "training_mode": "CPU 流式低内存模式",
            "data_dir": str(tmp_path / "rlogs"),
            "output_dir": str(tmp_path / "out"),
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(NNLCApp, "_preferences_path", staticmethod(lambda: settings))

    preferences = NNLCApp._load_preferences()

    assert preferences["car"] == "BYD_TANG_DMI_24"
    assert preferences["data_dir"] == os.path.normpath(str(tmp_path / "rlogs"))
    assert preferences["output_dir"] == os.path.normpath(str(tmp_path / "out"))


def test_load_preferences_ignores_non_string_directories(tmp_path, monkeypatch):
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps({"data_dir": ["not", "a", "path"], "output_dir": 12}),
        encoding="utf-8",
    )
    monkeypatch.setattr(NNLCApp, "_preferences_path", staticmethod(lambda: settings))

    preferences = NNLCApp._load_preferences()

    assert preferences["data_dir"] == ""
    assert preferences["output_dir"] == ""


def test_save_preferences_writes_directory_paths(tmp_path, monkeypatch):
    settings = tmp_path / "NNLC" / "settings.json"
    monkeypatch.setattr(NNLCApp, "_preferences_path", staticmethod(lambda: settings))
    app = NNLCApp.__new__(NNLCApp)
    app.car_var = _Var(" BYD_TANG_DMI_24 ")
    app.training_mode_var = _Var("CPU 流式低内存模式")
    app.data_var = _Var(str(tmp_path / "rlogs"))
    app.output_var = _Var(str(tmp_path / "out"))
    app.keep_intermediates_var = _Var(True)

    app._save_preferences()

    saved = json.loads(settings.read_text(encoding="utf-8"))
    assert saved["car"] == "BYD_TANG_DMI_24"
    assert saved["training_mode"] == "CPU 流式低内存模式"
    assert saved["data_dir"] == os.path.normpath(str(tmp_path / "rlogs"))
    assert saved["output_dir"] == os.path.normpath(str(tmp_path / "out"))
    assert saved["keep_intermediates"] is True
