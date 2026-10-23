import queue
from datetime import datetime
from pathlib import Path

from nnlc_gui import (
    NNLCApp,
    PIPELINE_STEP_COUNT,
    ellipsize_to_width,
    format_progress_title,
    latest_visible_log_line,
    parse_pipeline_step,
    create_training_log_file,
    sanitize_log_car_name,
    training_log_filename,
    training_log_path,
)


class _Var:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def __init__(self):
        self.options = {}

    def configure(self, **options):
        self.options.update(options)


def test_sanitize_log_car_name_replaces_path_characters():
    assert sanitize_log_car_name("BYD_QZJ_05_DMI") == "BYD_QZJ_05_DMI"
    assert sanitize_log_car_name("  BYD/QZJ:05  ") == "BYD_QZJ_05"
    assert sanitize_log_car_name("..") == "model"


def test_training_log_filename_uses_timestamp_and_car():
    when = datetime(2026, 10, 23, 12, 48, 9)
    assert training_log_filename("BYD_QZJ_05_DMI", when) == "20261023_124809_BYD_QZJ_05_DMI.log"
    assert training_log_filename("BYD/QZJ", when) == "20261023_124809_BYD_QZJ.log"


def test_training_log_path_lives_under_program_logs(tmp_path):
    when = datetime(2026, 10, 23, 12, 48, 9)
    path = training_log_path("BYD_QZJ_05_DMI", when=when, base_dir=tmp_path)
    assert path == tmp_path / "logs" / "20261023_124809_BYD_QZJ_05_DMI.log"


def test_same_second_log_creations_get_distinct_exclusive_files(tmp_path):
    when = datetime(2026, 10, 23, 12, 48, 9)
    first_path, first_file = create_training_log_file(
        "BYD_QZJ_05_DMI", when=when, base_dir=tmp_path
    )
    second_path, second_file = create_training_log_file(
        "BYD_QZJ_05_DMI", when=when, base_dir=tmp_path
    )
    first_file.write("first run\n")
    second_file.write("second run\n")
    first_file.close()
    second_file.close()

    assert first_path.name == "20261023_124809_BYD_QZJ_05_DMI.log"
    assert second_path.name == "20261023_124809_BYD_QZJ_05_DMI_2.log"
    assert first_path.read_text(encoding="utf-8") == "first run\n"
    assert second_path.read_text(encoding="utf-8") == "second run\n"


def test_parse_pipeline_step_reads_last_valid_heading():
    text = (
        "Training progress: 15/1000\n"
        "\x1b[1m\x1b[36m[5/7] 训练 BYD_QZJ_05_DMI 模型\x1b[0m\n"
        "============================================================\n"
    )
    assert parse_pipeline_step(text) == (5, 7, "训练 BYD_QZJ_05_DMI 模型")
    assert parse_pipeline_step("[1/6] 提取横向控制数据") is None
    assert parse_pipeline_step("Training progress: 15/1000") is None
    assert parse_pipeline_step("") is None
    chunk = "[1/7] 提取横向控制数据\n[5/7] 训练 BYD_QZJ_05_DMI 模型\n"
    assert parse_pipeline_step(chunk) == (5, 7, "训练 BYD_QZJ_05_DMI 模型")


def test_latest_visible_log_line_strips_ansi_and_keeps_full_line():
    text = "\x1b[36m[5/7] 训练 BYD_QZJ_05_DMI 模型\x1b[0m\n    正在编译训练代码，请继续等待\n"
    assert latest_visible_log_line(text) == "正在编译训练代码，请继续等待"
    assert latest_visible_log_line("   \n  ") == ""


class _FakeFont:
    def measure(self, text):
        return len(text) * 10


def test_ellipsize_to_width_uses_label_pixel_width():
    font = _FakeFont()
    assert ellipsize_to_width("abcdef", font, 100) == "abcdef"
    truncated = ellipsize_to_width("abcdefghijklmnopqrstuvwxyz", font, 80)
    assert truncated.endswith("…")
    assert font.measure(truncated) <= 80


def test_format_progress_title_uses_waiting_state():
    assert format_progress_title(0, PIPELINE_STEP_COUNT, "等待开始") == "0 / 7  等待开始"
    assert format_progress_title(5, 7, "训练 BYD_QZJ_05_DMI 模型") == "5 / 7  训练 BYD_QZJ_05_DMI 模型"


def test_progress_highlights_all_steps_up_to_current_in_button_blue():
    from nnlc_gui import C_ACCENT, C_MUTED

    app = NNLCApp.__new__(NNLCApp)
    app.progress_current = 0
    app.progress_total = PIPELINE_STEP_COUNT
    app.progress_heading = "等待开始"
    app.progress_title_var = _Var()
    app.latest_log_var = _Var()
    app.progress_bar = None
    app.progress_step_labels = [_Widget() for _ in range(PIPELINE_STEP_COUNT)]

    app._set_progress(5, 7, "训练 BYD_QZJ_05_DMI 模型")

    assert [label.options["fg"] for label in app.progress_step_labels] == [
        C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT, C_ACCENT, C_MUTED, C_MUTED,
    ]
    app._reset_progress_card()
    assert all(label.options["fg"] == C_MUTED for label in app.progress_step_labels)


def test_append_log_writes_immediately_and_keeps_file_after_close(tmp_path, monkeypatch):
    log_path = tmp_path / "logs" / "20261023_124809_BYD_QZJ_05_DMI.log"
    monkeypatch.setattr(
        "nnlc_gui.training_log_path",
        lambda car, when=None, base_dir=None: log_path,
    )

    app = NNLCApp.__new__(NNLCApp)
    app.progress_bar = None
    app.progress_title_var = _Var()
    app.latest_log_var = _Var()
    app._log_line_buffer = ""
    app.log_file = None
    app.current_log_path = None
    app._log_write_failed = False
    app.progress_step_labels = []

    app._open_training_log("BYD_QZJ_05_DMI")
    app._append_log("[5/7] 训练 BYD_QZJ_05_DMI 模型\n正在编译训练代码\n")

    assert log_path.exists()
    assert "正在编译训练代码" in log_path.read_text(encoding="utf-8")
    assert app.progress_current == 5
    assert app.latest_log_var.get() == "正在编译训练代码"

    app._append_log("[10:20:31] 训练失败：cancelled\n")
    app._close_training_log()
    content = log_path.read_text(encoding="utf-8")
    assert "训练失败：cancelled" in content
    assert log_path.exists()


def test_worker_sends_exception_traceback_to_gui_queue(monkeypatch):
    app = NNLCApp.__new__(NNLCApp)
    app.messages = queue.Queue()

    def failing_train(*args, **kwargs):
        raise RuntimeError("CSV header is invalid")

    monkeypatch.setattr("nnlc_gui.auto_train", failing_train)
    app._run_worker(None, None, None, None, False, 1, False, True, None, {})

    kind, payload = app.messages.get_nowait()
    assert kind == "error"
    exc, traceback_text = payload
    assert str(exc) == "CSV header is invalid"
    assert "Traceback (most recent call last)" in traceback_text
    assert "failing_train" in traceback_text


def test_training_error_appends_traceback_to_run_log(tmp_path):
    log_path = tmp_path / "training.log"
    app = NNLCApp.__new__(NNLCApp)
    app.log_file = log_path.open("w", encoding="utf-8")
    app._log_write_failed = False
    app._log_line_buffer = ""
    app.latest_log_var = _Var()

    app._append_training_error(
        RuntimeError("CSV header is invalid"),
        "Traceback (most recent call last):\n  File \"trainer.py\", line 42, in train\nRuntimeError: CSV header is invalid\n",
    )
    app._close_training_log()

    content = log_path.read_text(encoding="utf-8")
    assert "训练失败：CSV header is invalid" in content
    assert "完整异常堆栈：" in content
    assert 'File "trainer.py", line 42' in content
    assert "RuntimeError: CSV header is invalid" in content


def test_log_write_failure_is_reported_once_and_training_can_continue(monkeypatch, tmp_path):
    class _FailingLogFile:
        closed = False

        def write(self, text):
            raise OSError("disk is full")

        def flush(self):
            pass

        def close(self):
            self.closed = True

    errors = []
    log_file = _FailingLogFile()
    app = NNLCApp.__new__(NNLCApp)
    app.log_file = log_file
    app.current_log_path = tmp_path / "logs" / "training.log"
    app._log_write_failed = False
    app.status_var = _Var()
    monkeypatch.setattr("nnlc_gui.messagebox.showerror", lambda title, message: errors.append((title, message)))

    app._write_training_log("first output\n")
    app._write_training_log("second output\n")

    assert app._log_write_failed is True
    assert app.log_file is None
    assert log_file.closed is True
    assert app.status_var.get() == "日志保存失败，训练仍继续"
    assert len(errors) == 1
    assert errors[0][0] == "训练日志写入失败"
    assert str(app.current_log_path) in errors[0][1]
    assert "disk is full" in errors[0][1]
    assert "训练仍会继续" in errors[0][1]
