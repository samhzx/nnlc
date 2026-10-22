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
    excludes = ast.literal_eval(
        next(keyword.value for keyword in analysis.keywords if keyword.arg == "excludes")
    )
    for name in ("scipy", "pytest", "py", "pygments"):
        assert name in excludes


def test_windows_build_uses_clean_packaging_venv():
    script = Path(__file__).resolve().parents[1] / "build_windows.ps1"
    contents = script.read_text(encoding="utf-8")
    assert ".packaging-venv" in contents
    assert "pip install -e . pyinstaller" in contents
    assert "pip install -e . pytest" not in contents
    clean_at = contents.find("Cleaning previous PyInstaller output")
    venv_at = contents.find("Creating a clean packaging virtual environment")
    assert 0 <= clean_at < venv_at

def test_windows_child_creationflags_hides_console(monkeypatch):
    import nnlc_auto_train
    import subprocess

    monkeypatch.setattr(nnlc_auto_train.os, "name", "nt")
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    flags = nnlc_auto_train._windows_child_creationflags()
    assert flags & 0x08000000
    assert flags & 0x00000200


def test_julia_training_emits_epoch_progress():
    script = Path(__file__).resolve().parents[1] / "training" / "latmodel_temporal.jl"
    contents = script.read_text(encoding="utf-8")
    assert 'println(out_streams, "Training progress: 0/$epoch_max")' in contents
    assert 'println(out_streams, "Training progress: $epoch/$epoch_max")' in contents
    assert "flush(out_streams)" in contents


def test_forward_process_output_emits_compiling_hint(monkeypatch):
    import threading
    import nnlc_auto_train

    messages = []
    monkeypatch.setattr(nnlc_auto_train, "print_info", messages.append)
    monkeypatch.setattr(nnlc_auto_train, "_emit", lambda message: None)
    monkeypatch.setattr(nnlc_auto_train, "_terminate_process_tree", lambda process: None)

    class FakeStdout:
        def __iter__(self):
            if False:
                yield None

    class FakeProcess:
        def __init__(self):
            self.stdout = FakeStdout()
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def wait(self):
            self._alive = False
            return 0

    process = FakeProcess()
    original_thread = threading.Thread

    def fake_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        original_start = thread.start

        def start():
            process._alive = False
            original_start()

        thread.start = start
        return thread

    monkeypatch.setattr(nnlc_auto_train.threading, "Thread", fake_thread)
    assert nnlc_auto_train._forward_process_output(
        process,
        idle_message="Julia 正在编译依赖，请继续等待",
        idle_timeout=30,
    ) == 0
    assert messages[0] == "Julia 正在编译依赖，请继续等待"
