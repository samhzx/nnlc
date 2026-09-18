import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from nnlc_update import UpdateError
from nnlc_update_ui import GuiUpdateController


class FakeVar:
    def __init__(self, value=""):
        self.value = value

    def set(self, value) -> None:
        self.value = value

    def get(self):
        return self.value


class FakeRoot:
    def __init__(self) -> None:
        self.after_calls = []

    def after(self, ms, fn):
        self.after_calls.append((ms, fn))
        return None


class FakeWorker:
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class FakeHost:
    def __init__(self) -> None:
        self.root = FakeRoot()
        self.app_version = "1.0.4"
        self.messages: queue.Queue = queue.Queue()
        self.status_var = FakeVar("就绪")
        self.worker = None
        self._closing = False
        self.set_running_calls = []
        self.events = []
        self.opened_paths = []

    def _set_running(self, running: bool) -> None:
        self.set_running_calls.append(running)

    def _append_event(self, text: str, tag: str) -> None:
        self.events.append((text, tag))

    def _open_path(self, path: Path) -> None:
        self.opened_paths.append(path)


def make_manifest(**overrides):
    manifest = SimpleNamespace(
        version="1.0.5",
        size=4,
        filename="NNLC_Trainer-1.0.5-windows-x64.exe",
        notes="修复训练流程问题。",
        sha256="ab",
    )
    for key, value in overrides.items():
        setattr(manifest, key, value)
    return manifest


def drain_update_messages(controller: GuiUpdateController, host: FakeHost) -> list[str]:
    kinds = []
    while True:
        try:
            kind, payload = host.messages.get_nowait()
        except queue.Empty:
            break
        assert controller.handle_message(kind, payload)
        kinds.append(kind)
    return kinds


def test_defers_prompt_while_training():
    host = FakeHost()
    host.worker = FakeWorker(True)
    controller = GuiUpdateController(host)
    prompted = []
    controller._prompt_pending_update = lambda: prompted.append(True)

    handled = controller.handle_message("update_available", make_manifest())

    assert handled is True
    assert prompted == []
    assert controller.update_prompt_shown is False
    assert controller.pending_update.version == "1.0.5"
    assert host.events == [("发现新版本 v1.0.5，将在训练结束后提示。", "info")]


def test_prompts_after_training_finishes():
    host = FakeHost()
    controller = GuiUpdateController(host)
    controller.pending_update = make_manifest()

    controller.on_training_finished()

    assert host.root.after_calls == [(200, controller._prompt_pending_update)]


def test_handle_message_consumes_update_kinds():
    host = FakeHost()
    controller = GuiUpdateController(host)
    prompted = []
    controller._prompt_pending_update = lambda: prompted.append(True)

    assert controller.handle_message("update_available", make_manifest()) is True
    assert prompted == [True]
    assert controller.handle_message("update_download_progress", (1, 2, 50)) is True

    controller.update_in_progress = True
    host._closing = True
    assert controller.handle_message("update_downloaded", Path("x")) is True
    assert controller.in_progress is False

    controller.update_in_progress = True
    assert controller.handle_message("update_failed", UpdateError("失败")) is True
    assert controller.in_progress is False
    assert controller.handle_message("log", "x") is False
    assert controller.handle_message("done", "x") is False


def _run_download(controller, host, monkeypatch, download_impl):
    monkeypatch.setattr("nnlc_update_ui.download_update", download_impl)
    monkeypatch.setattr(controller, "_show_download_window", lambda manifest: None)
    monkeypatch.setattr(
        "nnlc_update_ui.messagebox",
        SimpleNamespace(
            askyesno=lambda *args, **kwargs: False,
            showinfo=lambda *args, **kwargs: None,
            showerror=lambda *args, **kwargs: None,
        ),
    )
    controller._start_update_download(make_manifest())
    assert controller.in_progress is True
    thread = controller.update_thread
    assert thread is not None
    thread.join(timeout=2)
    assert not thread.is_alive()
    drain_update_messages(controller, host)
    assert controller.in_progress is False


def test_download_success_resets_in_progress(monkeypatch, tmp_path):
    host = FakeHost()
    controller = GuiUpdateController(host)
    downloaded = tmp_path / "NNLC_Trainer-1.0.5-windows-x64.exe"
    _run_download(controller, host, monkeypatch, lambda *args, **kwargs: downloaded)
    assert host.status_var.get() == "更新已下载"
    assert host.events[-1][1] == "success"


def test_download_failure_resets_in_progress(monkeypatch):
    host = FakeHost()
    controller = GuiUpdateController(host)

    def fail(*args, **kwargs):
        raise UpdateError("下载更新失败: boom")

    _run_download(controller, host, monkeypatch, fail)
    assert host.status_var.get() == "更新下载失败"
    assert host.events[-1][1] == "error"


def test_download_cancel_resets_in_progress(monkeypatch):
    host = FakeHost()
    controller = GuiUpdateController(host)

    def cancel(*args, **kwargs):
        raise UpdateError("已取消下载。")

    _run_download(controller, host, monkeypatch, cancel)
    assert host.status_var.get() == "更新下载失败"
    assert host.events[-1] == ("已取消更新下载。", "warning")


def test_start_background_check_skips_when_not_frozen(monkeypatch):
    host = FakeHost()
    controller = GuiUpdateController(host)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("should not start a check thread")

    monkeypatch.setattr(threading, "Thread", fail)
    controller.start_background_check()
