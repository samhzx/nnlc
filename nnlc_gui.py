"""Small Tkinter front-end for the NNLC one-click training pipeline."""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
except ImportError:  # Some minimal Python builds (including CI macOS) omit Tk.
    tk = None
    filedialog = messagebox = ttk = None
    tkfont = None

from nnlc_auto_train import _terminate_process_tree, auto_train
from nnlc_runtime import application_directory
from nnlc_tools.model_inputs import DEFAULT_INPUT_DIM, INPUT_DIMENSIONS
from nnlc_update_ui import GuiUpdateController
from nnlc_version import get_version


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PIPELINE_STEP_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+)$")
PIPELINE_STEP_COUNT = 7
PIPELINE_STEP_LABELS = ("提取", "评分", "修剪", "覆盖度", "训练", "验证", "部署")
TRAINING_MODES = {
    "标准模式": 16384,
    "流式模式": 4096,
}
INPUT_DIM_OPTIONS = {f"{input_dim} 维": input_dim for input_dim in INPUT_DIMENSIONS}
INPUT_DIM_LABELS = {value: label for label, value in INPUT_DIM_OPTIONS.items()}


def normalize_input_dim(value) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value in INPUT_DIMENSIONS
        else DEFAULT_INPUT_DIM
    )


LEGACY_TRAINING_MODES = {
    "CPU 标准模式": "标准模式",
    "CPU 流式低内存模式": "流式模式",
    "低内存模式": "流式模式",
    "CPU 低内存模式": "流式模式",
}


class _QueueWriter(io.TextIOBase):
    def __init__(self, messages: queue.Queue):
        self.messages = messages

    def write(self, text: str) -> int:
        if text:
            self.messages.put(("log", text))
        return len(text)

    def flush(self) -> None:
        return None


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from console output."""
    return _ANSI_ESCAPE_RE.sub("", text)


def sanitize_log_car_name(car_name: str) -> str:
    """Keep a car name safe for a single log filename."""
    sanitized = re.sub(r"[\\/:*?\"<>|]+", "_", str(car_name).strip())
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "model"


def training_log_filename(
    car_name: str, when: datetime | None = None, input_dim: int = DEFAULT_INPUT_DIM
) -> str:
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{sanitize_log_car_name(car_name)}_{input_dim}d.log"


def training_log_directory(base_dir: str | os.PathLike[str] | None = None) -> Path:
    root = Path(base_dir) if base_dir is not None else application_directory()
    return root / "logs"


def training_log_path(car_name: str, when: datetime | None = None,
                      base_dir: str | os.PathLike[str] | None = None,
                      input_dim: int = DEFAULT_INPUT_DIM) -> Path:
    return training_log_directory(base_dir) / training_log_filename(
        car_name, when, input_dim
    )


def create_training_log_file(
    car_name: str,
    when: datetime | None = None,
    base_dir: str | os.PathLike[str] | None = None,
    input_dim: int = DEFAULT_INPUT_DIM,
):
    """Create a new log without overwriting or appending to another run."""
    base_path = training_log_path(
        car_name, when=when, base_dir=base_dir, input_dim=input_dim
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = base_path
    suffix = 2
    while True:
        try:
            handle = candidate.open("x", encoding="utf-8", newline="\n")
            return candidate, handle
        except FileExistsError:
            candidate = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
            suffix += 1


def parse_pipeline_step(text: str) -> tuple[int, int, str] | None:
    """Parse a pipeline heading such as ``[5/7] 训练 BYD_QZJ_05_DMI 模型``."""
    found = None
    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        if not line.startswith("["):
            continue
        match = _PIPELINE_STEP_RE.fullmatch(line)
        if match is None:
            continue
        current = int(match.group(1))
        total = int(match.group(2))
        title = match.group(3).strip()
        if current < 1 or total != PIPELINE_STEP_COUNT or current > total or not title:
            continue
        found = (current, total, title)
    return found


def latest_visible_log_line(text: str) -> str:
    """Return the last non-empty console line for the progress card."""
    visible = ""
    for raw_line in strip_ansi(text).replace("\r", "\n").splitlines():
        line = raw_line.strip()
        if line:
            visible = line
    return visible


def ellipsize_to_width(text: str, font, max_width: int) -> str:
    """Fit one line into ``max_width`` pixels, appending an ellipsis when needed."""
    if max_width <= 0 or not text:
        return text
    if font.measure(text) <= max_width:
        return text
    ellipsis = "…"
    ellipsis_width = font.measure(ellipsis)
    if ellipsis_width >= max_width:
        return ellipsis
    available = max_width - ellipsis_width
    low = 0
    high = len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if font.measure(text[:mid]) <= available:
            low = mid
        else:
            high = mid - 1
    return text[:low] + ellipsis


def format_progress_title(current: int, total: int, title: str) -> str:
    heading = title.strip() or "等待开始"
    return f"{current} / {total}  {heading}"


def progress_step_color(index: int, current: int, *, completed: bool = False) -> str:
    """Color pipeline labels: done, current, and pending."""
    if completed:
        return C_STEP_DONE
    if current <= 0:
        return C_STEP_TODO
    if index < current - 1:
        return C_STEP_DONE
    if index == current - 1:
        return C_STEP_CURRENT
    return C_STEP_TODO


APP_TITLE = "NNLC 横向控制模型训练"
APP_SUBTITLE = "从 reallog 提取数据，完成评分、训练与模型部署"
C_HEADER_BG = "#16324f"
C_HEADER_SUB = "#8fb0d1"
C_BADGE_BG = "#274d72"
C_BADGE_FG = "#d5e4f4"
C_PAGE_BG = "#f2f4f8"
C_CARD_BG = "#ffffff"
C_BORDER = "#e2e7ef"
C_ACCENT = "#1890ff"
C_ACCENT_HOVER = "#40a9ff"
C_ACCENT_DISABLED = "#b3d8ff"
C_TEXT = "#2b333d"
C_MUTED = "#7d8590"
C_BTN_TEXT = "#606266"
C_BTN_TEXT_HOVER = "#303133"
C_BTN_BORDER = "#dcdfe6"
C_BTN_BORDER_HOVER = "#c0c4cc"
C_BTN_DISABLED_BG = "#f5f7fa"
C_BTN_DISABLED_FG = "#c0c4cc"
C_BTN_DISABLED_BORDER = "#e4e7ed"
C_STATUS_BG = "#e8ecf3"
C_DISABLED = "#f5f6f8"
C_STEP_DONE = "#2b333d"
C_STEP_CURRENT = C_ACCENT
C_STEP_TODO = C_MUTED
WINDOW_ASPECT_WIDTH = 16
WINDOW_ASPECT_HEIGHT = 9
F_TITLE = ("Microsoft YaHei UI", 16, "bold")
F_VERSION = ("Microsoft YaHei UI", 9, "bold")
F_SUB = ("Microsoft YaHei UI", 9)
F_SECTION = ("Microsoft YaHei UI", 11, "bold")
F_BASE = ("Microsoft YaHei UI", 10)
F_BOLD = ("Microsoft YaHei UI", 10, "bold")
PAD_PAGE = 20
PAD_CARD = 18
PAD_ROW = 7
INPUT_H = 34
LABEL_W = 72
MODE_BOX_W = 150


def window_dimensions_for_height(height: int) -> tuple[int, int]:
    """Return a fixed 16:9 window size for the laid-out content height."""
    return round(height * WINDOW_ASPECT_WIDTH / WINDOW_ASPECT_HEIGHT), height


if tk is not None:
    class PathPicker(tk.Frame):
        """Path entry plus a compact browse control."""

        def __init__(self, parent, variable, browse_command):
            super().__init__(
                parent,
                bg=C_CARD_BG,
                height=INPUT_H,
                highlightthickness=1,
                highlightbackground=C_BORDER,
                highlightcolor=C_ACCENT,
            )
            self.pack_propagate(False)
            self.grid_propagate(False)
            self.variable = variable
            self.browse_command = browse_command
            self._enabled = True

            self.entry = tk.Entry(
                self,
                textvariable=variable,
                font=F_BASE,
                bd=0,
                highlightthickness=0,
                fg=C_TEXT,
                bg=C_CARD_BG,
                disabledbackground=C_DISABLED,
                disabledforeground=C_MUTED,
            )
            self.entry.pack(side="left", fill="both", expand=True, padx=(10, 0))
            self.entry.bind("<FocusIn>", lambda _e: self.configure(highlightbackground=C_ACCENT))
            self.entry.bind("<FocusOut>", lambda _e: self.configure(highlightbackground=C_BORDER))

            tk.Frame(self, bg=C_BORDER, width=1).pack(side="left", fill="y", pady=8)
            self.btn = tk.Label(
                self,
                text="···",
                font=("Microsoft YaHei UI", 11, "bold"),
                bg=C_CARD_BG,
                fg=C_MUTED,
                cursor="hand2",
                padx=10,
                anchor="center",
            )
            self.btn.pack(side="left", fill="both")
            self.btn.bind("<Button-1>", self._on_browse)
            self.btn.bind("<Enter>", lambda _e: self.btn.configure(fg=C_ACCENT if self._enabled else C_MUTED))
            self.btn.bind("<Leave>", lambda _e: self.btn.configure(fg=C_MUTED))

        def _on_browse(self, _event=None):
            if self._enabled:
                self.browse_command()

        def set_enabled(self, enabled: bool) -> None:
            self._enabled = enabled
            fill = C_CARD_BG if enabled else C_DISABLED
            self.entry.configure(state="normal" if enabled else "disabled", bg=fill)
            self.configure(bg=fill)
            self.btn.configure(bg=fill, cursor="hand2" if enabled else "arrow")


class NNLCApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.geometry("1200x640")
        self.root.resizable(False, False)
        try:
            self.root.attributes("-zoomed", False)
        except tk.TclError:
            pass
        # macOS still shows the green zoom button unless it is explicitly disabled.
        if sys.platform == "darwin":
            try:
                self.root.createcommand("tk::mac::ZoomWindow", lambda: None)
            except tk.TclError:
                pass
        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.started_at: float | None = None
        self.config_widgets = []
        self.path_pickers = []
        self.cancel_event = threading.Event()
        self.process_holder = {}
        self.log_file = None
        self.current_log_path: Path | None = None
        self._log_write_failed = False
        self._log_line_buffer = ""
        self.progress_current = 0
        self.progress_total = PIPELINE_STEP_COUNT
        self.progress_heading = "等待开始"
        self.progress_step_labels = []

        saved_preferences = self._load_preferences()
        saved_mode = saved_preferences.get("training_mode")
        saved_mode = LEGACY_TRAINING_MODES.get(saved_mode, saved_mode)
        if not isinstance(saved_mode, str) or saved_mode not in TRAINING_MODES:
            saved_mode = "标准模式"
        saved_input_dim = normalize_input_dim(
            saved_preferences.get("input_dim", DEFAULT_INPUT_DIM)
        )

        saved_car = saved_preferences.get("car")
        if not isinstance(saved_car, str) or not saved_car.strip():
            saved_car = "BYD_TANG_DMI_24"

        saved_data_dir = saved_preferences.get("data_dir", "")
        saved_output_dir = saved_preferences.get("output_dir", "")
        self.data_var = tk.StringVar(value=saved_data_dir)
        self.output_var = tk.StringVar(value=saved_output_dir or self._default_output())
        self.car_var = tk.StringVar(value=saved_car)
        self.threshold_var = tk.StringVar()
        self.training_mode_var = tk.StringVar(value=saved_mode)
        self.input_dim_var = tk.StringVar(value=INPUT_DIM_LABELS[saved_input_dim])
        self.auto_threshold_var = tk.BooleanVar(value=True)
        self.skip_viz_var = tk.BooleanVar(value=True)
        self.keep_intermediates_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.elapsed_var = tk.StringVar(value="耗时 00:00")
        self.progress_title_var = tk.StringVar(value=format_progress_title(0, PIPELINE_STEP_COUNT, "等待开始"))
        self.latest_log_var = tk.StringVar(value="等待开始训练")
        self._latest_log_text = "等待开始训练"
        try:
            self.app_version = get_version()
        except Exception:
            self.app_version = "unknown"
        self.root.title(f"{APP_TITLE} v{self.app_version}")
        self._closing = False
        self.updater = GuiUpdateController(self)
        self._build_widgets()
        self._lock_window_to_content()
        self.root.after_idle(self._refresh_latest_log_display)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_messages()
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            self.root.after(1000, self.updater.start_background_check)

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize user-visible paths using the current OS convention."""
        path = path.strip()
        return os.path.normpath(os.path.expanduser(path)) if path else ""

    @classmethod
    def _default_output(cls) -> str:
        documents = Path.home() / "Documents"
        default_path = documents / "NNLC_Output" if documents.exists() else Path.cwd() / "NNLC_Output"
        return cls._normalize_path(str(default_path))

    @staticmethod
    def _preferences_path() -> Path:
        """Return a per-user settings path that also works in a bundled exe."""
        if os.name == "nt":
            config_root = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
        elif sys.platform == "darwin":
            config_root = Path.home() / "Library" / "Application Support"
        else:
            config_root = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
        return Path(config_root) / "NNLC" / "settings.json"

    @classmethod
    def _load_preferences(cls) -> dict:
        try:
            with cls._preferences_path().open("r", encoding="utf-8") as handle:
                preferences = json.load(handle)
            if not isinstance(preferences, dict):
                return {}
            car = preferences.get("car")
            if isinstance(car, str):
                preferences["car"] = car.strip()
            preferences["data_dir"] = cls._saved_path(preferences.get("data_dir"))
            preferences["output_dir"] = cls._saved_path(preferences.get("output_dir"))
            return preferences
        except (OSError, ValueError, TypeError):
            return {}

    @classmethod
    def _saved_path(cls, value) -> str:
        """Normalize a persisted directory path, ignoring invalid values."""
        if not isinstance(value, str):
            return ""
        return cls._normalize_path(value)

    def _save_preferences(self) -> None:
        car = self.car_var.get().strip()
        training_mode = self.training_mode_var.get()
        training_mode = LEGACY_TRAINING_MODES.get(training_mode, training_mode)
        if training_mode not in TRAINING_MODES:
            training_mode = "标准模式"
        input_dim = INPUT_DIM_OPTIONS.get(
            self.input_dim_var.get(), DEFAULT_INPUT_DIM
        )
        path = self._preferences_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(".tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "car": car,
                        "training_mode": training_mode,
                        "input_dim": input_dim,
                        "data_dir": self._saved_path(self.data_var.get()),
                        "output_dir": self._saved_path(self.output_var.get()),
                        "keep_intermediates": bool(self.keep_intermediates_var.get()),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
            os.replace(temporary_path, path)
        except OSError:
            # A read-only profile must not prevent the training task from starting.
            pass

    def _build_widgets(self) -> None:
        self._build_style()
        self.root.configure(bg=C_PAGE_BG)
        self._build_header()
        self._build_body()
        self._build_log()
        self._build_statusbar()

    def _lock_window_to_content(self) -> None:
        """Keep the original content height and size the width to 16:9."""
        self.root.update_idletasks()
        height = self.root.winfo_reqheight()
        width, height = window_dimensions_for_height(height)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(width, height)
        self.root.maxsize(width, height)
        self.root.resizable(False, False)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
        style.configure("TCombobox", padding=(8, 2))
        style.configure(
            "NNLC.Horizontal.TProgressbar",
            troughcolor="#e8edf3",
            background=C_ACCENT,
            lightcolor=C_ACCENT,
            darkcolor=C_ACCENT,
            bordercolor="#e8edf3",
            thickness=10,
        )

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=C_HEADER_BG, height=84)
        header.pack(fill="x")
        header.pack_propagate(False)

        left = tk.Frame(header, bg=C_HEADER_BG)
        left.pack(side="left", padx=24, pady=13, fill="y")
        tk.Label(left, text=APP_TITLE, font=F_TITLE, bg=C_HEADER_BG, fg="#ffffff").pack(anchor="w")
        tk.Label(left, text=APP_SUBTITLE, font=F_SUB, bg=C_HEADER_BG, fg=C_HEADER_SUB).pack(
            anchor="w", pady=(4, 0)
        )
        tk.Label(
            header,
            text=f" v{self.app_version} ",
            font=F_VERSION,
            bg=C_BADGE_BG,
            fg=C_BADGE_FG,
            padx=8,
            pady=3,
        ).pack(side="right", padx=24)

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=C_PAGE_BG)
        body.pack(fill="x", padx=PAD_PAGE, pady=(16, 0))
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")

        dir_card, dir_inner = self._make_card(body, "目录设置")
        dir_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        dir_inner.columnconfigure(0, minsize=LABEL_W)
        dir_inner.columnconfigure(1, weight=1)

        data_picker = PathPicker(dir_inner, self.data_var, self._choose_data)
        self._form_row(dir_inner, 0, "日志目录", data_picker)
        output_picker = PathPicker(dir_inner, self.output_var, self._choose_output)
        self._form_row(dir_inner, 1, "输出目录", output_picker)
        self.path_pickers = [data_picker, output_picker]
        data_picker.entry.bind("<FocusOut>", lambda _e: self._save_preferences(), add="+")
        output_picker.entry.bind("<FocusOut>", lambda _e: self._save_preferences(), add="+")

        param_card, param_inner = self._make_card(body, "训练参数")
        param_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        param_inner.columnconfigure(0, minsize=LABEL_W)
        param_inner.columnconfigure(1, weight=1)
        param_inner.columnconfigure(2, minsize=LABEL_W)

        tk.Label(param_inner, text="目标车型", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=0, column=0, sticky="w", pady=PAD_ROW
        )
        self.car_box = self._fixed_box(param_inner)
        self.car_box.grid(row=0, column=1, sticky="ew", padx=(10, 14), pady=PAD_ROW)
        self.car_entry = self._make_form_entry(self.car_box, self.car_var)
        self.config_widgets.append(self.car_entry)

        tk.Label(param_inner, text="训练模式", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=0, column=2, sticky="w", pady=PAD_ROW
        )
        self.mode_box = self._fixed_box(param_inner, width=MODE_BOX_W)
        self.mode_box.grid(row=0, column=3, sticky="w", padx=(10, 0), pady=PAD_ROW)
        self.training_mode_combo = ttk.Combobox(
            self.mode_box,
            textvariable=self.training_mode_var,
            values=tuple(TRAINING_MODES),
            state="readonly",
            font=F_BASE,
        )
        self.training_mode_combo.pack(fill="both", expand=True)
        self.training_mode_combo.bind("<<ComboboxSelected>>", self._on_training_mode_changed)
        self.config_widgets.append(self.training_mode_combo)

        tk.Label(param_inner, text="训练维度", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=1, column=2, sticky="w", pady=PAD_ROW
        )
        self.input_dim_box = self._fixed_box(param_inner, width=MODE_BOX_W)
        self.input_dim_box.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=PAD_ROW)
        self.input_dim_combo = ttk.Combobox(
            self.input_dim_box,
            textvariable=self.input_dim_var,
            values=tuple(INPUT_DIM_OPTIONS),
            state="readonly",
            font=F_BASE,
        )
        self.input_dim_combo.pack(fill="both", expand=True)
        self.input_dim_combo.bind("<<ComboboxSelected>>", self._on_input_dim_changed)
        self.config_widgets.append(self.input_dim_combo)

        tk.Label(param_inner, text="路线阈值", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=1, column=0, sticky="w", pady=PAD_ROW
        )
        threshold_controls = tk.Frame(param_inner, bg=C_CARD_BG)
        threshold_controls.grid(row=1, column=1, sticky="ew", padx=(10, 14), pady=PAD_ROW)
        threshold_controls.columnconfigure(0, weight=1)
        self.thr_box = self._fixed_box(threshold_controls)
        self.thr_box.grid(row=0, column=0, sticky="ew")
        self.threshold_entry = self._make_form_entry(self.thr_box, self.threshold_var)
        self.auto_threshold_check = tk.Checkbutton(
            threshold_controls,
            text="自动",
            variable=self.auto_threshold_var,
            font=F_BASE,
            bg=C_CARD_BG,
            activebackground=C_CARD_BG,
            selectcolor=C_CARD_BG,
            cursor="hand2",
            command=self._toggle_threshold,
        )
        self.auto_threshold_check.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.config_widgets.append(self.auto_threshold_check)
        self._toggle_threshold()

    def _build_log(self) -> None:
        card, inner = self._make_card(self.root, "训练进度")
        card.pack(fill="x", padx=PAD_PAGE, pady=(12, 8))

        header = tk.Frame(inner, bg=C_CARD_BG)
        header.pack(fill="x")
        tk.Label(
            header,
            textvariable=self.progress_title_var,
            font=F_BOLD,
            bg=C_CARD_BG,
            fg=C_TEXT,
            anchor="w",
        ).pack(side="left")
        tk.Label(
            header,
            textvariable=self.elapsed_var,
            font=F_BASE,
            bg=C_CARD_BG,
            fg=C_MUTED,
            anchor="e",
        ).pack(side="right")

        self.progress_bar = ttk.Progressbar(
            inner,
            orient="horizontal",
            mode="determinate",
            maximum=PIPELINE_STEP_COUNT,
            value=0,
            style="NNLC.Horizontal.TProgressbar",
        )
        self.progress_bar.pack(fill="x", pady=(10, 4))

        steps = tk.Frame(inner, bg=C_CARD_BG)
        steps.pack(fill="x")
        self.progress_step_labels = []
        for index, label in enumerate(PIPELINE_STEP_LABELS):
            steps.columnconfigure(index, weight=1)
            step_label = tk.Label(
                steps,
                text=label,
                font=F_SUB,
                bg=C_CARD_BG,
                fg=C_MUTED,
                anchor="center",
            )
            step_label.grid(row=0, column=index, sticky="ew")
            self.progress_step_labels.append(step_label)

        latest_box = tk.Frame(
            inner,
            bg="#fbfcfe",
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_BORDER,
        )
        latest_box.pack(fill="x", pady=(12, 0))
        latest_inner = tk.Frame(latest_box, bg="#fbfcfe")
        latest_inner.pack(fill="x", padx=12, pady=10)
        tk.Label(
            latest_inner,
            text="最新日志",
            font=F_SUB,
            bg="#fbfcfe",
            fg=C_MUTED,
            anchor="w",
        ).pack(fill="x")
        self.latest_log_label = tk.Label(
            latest_inner,
            textvariable=self.latest_log_var,
            font=F_BASE,
            bg="#fbfcfe",
            fg=C_TEXT,
            anchor="w",
            justify="left",
            wraplength=0,
        )
        self.latest_log_label.pack(fill="x", pady=(4, 0))
        self._latest_log_text = "等待开始训练"

        tk.Label(
            inner,
            text="完整日志写入程序运行目录的 logs/*.log",
            font=F_SUB,
            bg=C_CARD_BG,
            fg=C_MUTED,
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

        actions = tk.Frame(inner, bg=C_CARD_BG)
        actions.pack(fill="x", pady=(14, 0))
        self.open_output_button_box, self.open_output_button = self._make_button(
            actions, "打开输出目录", self._open_output
        )
        self.open_output_button_box.pack(side="left")
        self.open_log_button_box, self.open_log_button = self._make_button(
            actions, "打开日志", self._open_log
        )
        self.open_log_button_box.pack(side="left", padx=(8, 0))

        self.start_button = tk.Button(
            actions,
            text="开始训练",
            font=F_BOLD,
            bg=C_ACCENT,
            fg="#ffffff",
            activebackground=C_ACCENT_HOVER,
            activeforeground="#ffffff",
            disabledforeground="#ffffff",
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=28,
            pady=6,
            cursor="hand2",
            command=self.start,
        )
        self.start_button.pack(side="right")
        self._bind_primary_hover(self.start_button)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=C_STATUS_BG, height=28)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        tk.Label(
            bar,
            textvariable=self.status_var,
            font=F_SUB,
            bg=C_STATUS_BG,
            fg=C_TEXT,
        ).pack(side="left", padx=14, pady=4)
        tk.Label(
            bar,
            text=f"v{self.app_version}",
            font=F_SUB,
            bg=C_STATUS_BG,
            fg=C_MUTED,
        ).pack(side="right", padx=14, pady=4)

    def _make_card(self, parent, title):
        wrapper = tk.Frame(parent, bg=C_PAGE_BG)
        header = tk.Frame(wrapper, bg=C_PAGE_BG)
        header.pack(anchor="w", pady=(0, 8))
        bar = tk.Frame(header, bg=C_ACCENT, width=4, height=15)
        bar.pack(side="left")
        bar.pack_propagate(False)
        tk.Label(header, text=title, font=F_SECTION, bg=C_PAGE_BG, fg=C_TEXT).pack(
            side="left", padx=(8, 0)
        )
        card = tk.Frame(
            wrapper,
            bg=C_CARD_BG,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_BORDER,
            takefocus=0,
        )
        card.pack(fill="both", expand=True)
        inner = tk.Frame(card, bg=C_CARD_BG)
        inner.pack(fill="both", expand=True, padx=PAD_CARD, pady=PAD_CARD - 4)
        return wrapper, inner

    def _form_row(self, parent, row, label_text, widget) -> None:
        tk.Label(parent, text=label_text, font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=row, column=0, sticky="w", pady=PAD_ROW
        )
        widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=PAD_ROW)

    def _make_form_entry(self, box, variable):
        entry = tk.Entry(
            box,
            textvariable=variable,
            font=F_BASE,
            bd=0,
            highlightthickness=0,
            fg=C_TEXT,
            bg=C_CARD_BG,
            disabledbackground=C_DISABLED,
            disabledforeground=C_MUTED,
        )
        entry.pack(fill="both", expand=True, padx=(10, 0))
        entry.bind("<FocusIn>", lambda _e: box.configure(highlightbackground=C_ACCENT))
        entry.bind("<FocusOut>", lambda _e: box.configure(highlightbackground=C_BORDER))
        return entry

    def _fixed_box(self, parent, width=None):
        box = tk.Frame(
            parent,
            bg=C_CARD_BG,
            height=INPUT_H,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_ACCENT,
        )
        if width:
            box.configure(width=width)
        box.pack_propagate(False)
        box.grid_propagate(False)
        return box

    def _make_button(self, parent, text, command=None):
        box = tk.Frame(parent, bg=C_BTN_BORDER)
        btn = tk.Button(
            box,
            text=text,
            font=F_BASE,
            command=command,
            bg=C_CARD_BG,
            fg=C_BTN_TEXT,
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=13,
            pady=4,
            cursor="hand2",
            activebackground=C_CARD_BG,
            activeforeground=C_BTN_TEXT_HOVER,
            disabledforeground=C_BTN_DISABLED_FG,
        )
        btn.pack(padx=1, pady=1)
        btn.bind("<Enter>", lambda _e: self._on_secondary_button_enter(box, btn))
        btn.bind("<Leave>", lambda _e: self._on_secondary_button_leave(box, btn))
        return box, btn

    def _on_secondary_button_enter(self, box, button) -> None:
        if str(button["state"]) == "disabled":
            return
        box.configure(bg=C_BTN_BORDER_HOVER)
        button.configure(bg=C_CARD_BG, fg=C_BTN_TEXT_HOVER)

    def _on_secondary_button_leave(self, box, button) -> None:
        if str(button["state"]) == "disabled":
            return
        box.configure(bg=C_BTN_BORDER)
        button.configure(bg=C_CARD_BG, fg=C_BTN_TEXT)

    def _style_secondary_button(self, box, button, enabled: bool) -> None:
        if enabled:
            box.configure(bg=C_BTN_BORDER)
            button.configure(
                state="normal",
                bg=C_CARD_BG,
                fg=C_BTN_TEXT,
                disabledforeground=C_BTN_TEXT,
                activebackground=C_CARD_BG,
                activeforeground=C_BTN_TEXT_HOVER,
                cursor="hand2",
            )
        else:
            box.configure(bg=C_BTN_DISABLED_BORDER)
            button.configure(
                state="disabled",
                bg=C_BTN_DISABLED_BG,
                fg=C_BTN_DISABLED_FG,
                disabledforeground=C_BTN_DISABLED_FG,
                activebackground=C_BTN_DISABLED_BG,
                activeforeground=C_BTN_DISABLED_FG,
                cursor="arrow",
            )

    def _bind_primary_hover(self, button) -> None:
        button.bind(
            "<Enter>",
            lambda _event: (
                button.configure(bg=C_ACCENT_HOVER)
                if str(button["state"]) != "disabled"
                else None
            ),
        )
        button.bind(
            "<Leave>",
            lambda _event: (
                button.configure(bg=C_ACCENT)
                if str(button["state"]) != "disabled"
                else None
            ),
        )

    def _disabled_cursor(self) -> str:
        return "no" if sys.platform == "win32" else "arrow"

    def _toggle_threshold(self) -> None:
        auto = self.auto_threshold_var.get()
        busy = bool((self.worker and self.worker.is_alive()) or self.updater.in_progress)
        disabled = auto or busy
        self.threshold_entry.configure(state="disabled" if disabled else "normal")
        fill = C_DISABLED if disabled else C_CARD_BG
        if getattr(self, "thr_box", None) is not None:
            self.thr_box.configure(bg=fill)
            self.threshold_entry.configure(bg=fill)

    def _update_streaming_options(self) -> None:
        return

    def _on_training_mode_changed(self, _event=None) -> None:
        self._update_streaming_options()
        self._save_preferences()

    def _on_input_dim_changed(self, _event=None) -> None:
        self._save_preferences()

    def _choose_data(self) -> None:
        path = filedialog.askdirectory(title="选择日志目录")
        if path:
            self.data_var.set(self._normalize_path(path))
            self._save_preferences()

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(self._normalize_path(path))
            self._save_preferences()

    def _open_output(self) -> None:
        output_dir = Path(self.output_var.get().strip())
        if not output_dir.is_dir():
            messagebox.showinfo("输出目录", "输出目录尚未创建，请先完成一次训练。")
            return
        self._open_path(output_dir)

    def _open_log(self) -> None:
        if self.current_log_path is not None and self.current_log_path.is_file():
            self._open_path(self.current_log_path)
            return
        log_dir = training_log_directory()
        if log_dir.is_dir():
            self._open_path(log_dir)
            return
        messagebox.showinfo("打开日志", "还没有训练日志，请先开始一次训练。")

    def _reset_progress_card(self) -> None:
        self.progress_current = 0
        self.progress_total = PIPELINE_STEP_COUNT
        self.progress_heading = "等待开始"
        self.progress_title_var.set(format_progress_title(0, PIPELINE_STEP_COUNT, "等待开始"))
        self._set_latest_log("等待开始训练")
        self._update_progress_step_labels(0)
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.configure(maximum=PIPELINE_STEP_COUNT, value=0)

    def _set_progress(self, current: int, total: int, title: str) -> None:
        self.progress_current = current
        self.progress_total = total
        self.progress_heading = title
        self.progress_title_var.set(format_progress_title(current, total, title))
        self._update_progress_step_labels(current)
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.configure(maximum=total, value=current)

    def _update_progress_step_labels(self, current: int, *, completed: bool = False) -> None:
        for index, label in enumerate(getattr(self, "progress_step_labels", ())):
            color = progress_step_color(index, current, completed=completed)
            label.configure(
                fg=color,
                font=F_BOLD if color == C_STEP_CURRENT else F_SUB,
            )

    def _open_training_log(self, car: str, input_dim: int = DEFAULT_INPUT_DIM) -> None:
        self._close_training_log()
        path, self.log_file = create_training_log_file(car, input_dim=input_dim)
        self.current_log_path = path
        self._log_write_failed = False
        self._log_line_buffer = ""

    def _write_training_log(self, text: str) -> None:
        if self.log_file is None or self._log_write_failed:
            return
        try:
            self.log_file.write(strip_ansi(text))
            self.log_file.flush()
        except OSError as exc:
            self._log_write_failed = True
            handle = self.log_file
            self.log_file = None
            try:
                handle.close()
            except OSError:
                pass
            message = (
                "无法继续保存本次完整训练日志。训练仍会继续，但之后的输出不会写入日志文件。\n\n"
                f"日志文件：{self.current_log_path}\n原因：{exc}"
            )
            self.status_var.set("日志保存失败，训练仍继续")
            if messagebox is not None:
                messagebox.showerror("训练日志写入失败", message)

    def _close_training_log(self) -> None:
        handle = self.log_file
        self.log_file = None
        if handle is None:
            return
        try:
            handle.flush()
        except OSError:
            pass
        try:
            handle.close()
        except OSError:
            pass

    def _append_log(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        self._write_training_log(normalized)
        self._log_line_buffer += normalized
        if "\n" not in self._log_line_buffer:
            return
        *complete, self._log_line_buffer = self._log_line_buffer.split("\n")
        chunk = "\n".join(complete)
        step = parse_pipeline_step(chunk)
        if step is not None:
            current, total, title = step
            self._set_progress(current, total, title)
        latest = latest_visible_log_line(chunk)
        if latest:
            self._set_latest_log(latest)

    def _append_event(self, text: str) -> None:
        self._append_log(f"[{time.strftime('%H:%M:%S')}] {text}\n")

    def _append_training_error(self, exc: BaseException, traceback_text: str) -> None:
        summary = f"训练失败：{exc}"
        details = traceback_text.rstrip()
        if details:
            self._append_log(f"{summary}\n完整异常堆栈：\n{details}\n")
        else:
            self._append_log(f"{summary}\n")

    def _set_latest_log(self, text: str) -> None:
        self._latest_log_text = text
        self._refresh_latest_log_display()

    def _refresh_latest_log_display(self) -> None:
        text = getattr(self, "_latest_log_text", "")
        label = getattr(self, "latest_log_label", None)
        if label is None or tkfont is None:
            self.latest_log_var.set(text)
            return
        container = label.master
        container.update_idletasks()
        max_width = container.winfo_width()
        if max_width <= 1:
            self.latest_log_var.set(text)
            return
        try:
            font = tkfont.Font(font=label.cget("font"))
        except tk.TclError:
            self.latest_log_var.set(text)
            return
        self.latest_log_var.set(ellipsize_to_width(text, font, max_width))

    def _set_running(self, running: bool) -> None:
        busy = running or self.updater.in_progress
        state = "disabled" if busy else "normal"
        for widget in self.config_widgets:
            widget.configure(state=state)
        for picker in self.path_pickers:
            picker.set_enabled(not busy)
        fill = C_DISABLED if busy else C_CARD_BG
        if getattr(self, "car_box", None) is not None:
            self.car_box.configure(bg=fill)
            self.car_entry.configure(bg=fill)
        self._toggle_threshold()
        self.start_button.configure(
            state="disabled" if busy else "normal",
            bg=C_ACCENT_DISABLED if busy else C_ACCENT,
            fg="#ffffff",
            disabledforeground="#ffffff",
            activebackground=C_ACCENT_DISABLED if busy else C_ACCENT_HOVER,
            cursor=self._disabled_cursor() if busy else "hand2",
        )
        self._style_secondary_button(
            self.open_output_button_box,
            self.open_output_button,
            enabled=True,
        )
        self._style_secondary_button(
            self.open_log_button_box,
            self.open_log_button,
            enabled=True,
        )
        if not busy:
            # Comboboxes are intentionally readonly; restoring every widget
            # to ``normal`` would let an invalid training mode be typed in.
            self.training_mode_combo.configure(state="readonly")
            self._update_streaming_options()

    def _update_elapsed(self) -> None:
        if not self.worker or not self.worker.is_alive() or self.started_at is None:
            return
        seconds = int(time.monotonic() - self.started_at)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.elapsed_var.set(f"耗时 {hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.elapsed_var.set(f"耗时 {minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._update_elapsed)

    def _finish_running(self, status: str, *, completed: bool = False) -> None:
        leftover = self._log_line_buffer.strip()
        if leftover:
            latest = latest_visible_log_line(leftover)
            if latest:
                self._set_latest_log(latest)
            self._log_line_buffer = ""
        if completed:
            self._update_progress_step_labels(self.progress_current, completed=True)
        self._close_training_log()
        self._set_running(False)
        self.status_var.set(status)
        self.updater.on_training_finished()

    def _on_close(self) -> None:
        self._save_preferences()
        if not self.updater.confirm_close_if_downloading():
            return
        if self.worker and self.worker.is_alive():
            should_close = messagebox.askyesno(
                "训练仍在进行",
                "关闭窗口会中断当前训练，确定要退出吗？",
                icon="warning",
            )
            if not should_close:
                return
            self.cancel_event.set()
            process = self.process_holder.get("process")
            if process is not None and process.poll() is None:
                _terminate_process_tree(process)
        self._close_training_log()
        self._closing = True
        self.updater.cancel_and_join_on_close()
        self.root.destroy()

    def start(self) -> None:
        if self.updater.in_progress:
            messagebox.showinfo("正在下载更新", "请等待更新下载完成后再开始训练。")
            return
        if self.worker and self.worker.is_alive():
            return
        data_dir = self._normalize_path(self.data_var.get())
        output_dir = self._normalize_path(self.output_var.get())
        self.data_var.set(data_dir)
        self.output_var.set(output_dir)
        car = self.car_var.get().strip()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("参数错误", "请选择有效的日志目录。")
            return
        if not output_dir:
            messagebox.showerror("参数错误", "请选择输出目录。")
            return
        if not car:
            messagebox.showerror("参数错误", "目标车型不能为空。")
            return
        self._save_preferences()

        min_score = None
        if not self.auto_threshold_var.get():
            try:
                min_score = int(self.threshold_var.get().strip())
            except ValueError:
                messagebox.showerror("参数错误", "路线阈值必须是 0-100 的整数，或勾选“自动”。")
                return
            if not 0 <= min_score <= 100:
                messagebox.showerror("参数错误", "路线阈值必须在 0-100 之间。")
                return

        training_mode = self.training_mode_var.get().strip()
        training_mode = LEGACY_TRAINING_MODES.get(training_mode, training_mode)
        batch_size = TRAINING_MODES.get(training_mode)
        if batch_size is None:
            messagebox.showerror("参数错误", "请选择有效的训练模式。")
            return
        input_dim = INPUT_DIM_OPTIONS.get(self.input_dim_var.get())
        if input_dim not in INPUT_DIMENSIONS:
            messagebox.showerror("参数错误", "请选择有效的训练维度。")
            return
        try:
            self._open_training_log(car, input_dim=input_dim)
        except OSError as exc:
            messagebox.showerror("保存失败", f"无法创建训练日志：\n{exc}")
            return
        self._reset_progress_card()
        self._append_event("训练任务已启动")
        self._set_running(True)
        self.status_var.set("训练中，请保持窗口打开...")
        self.elapsed_var.set("耗时 00:00")
        self.started_at = time.monotonic()
        self.cancel_event = threading.Event()
        self.process_holder = {}
        skip_visualize = not self.skip_viz_var.get()
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(data_dir, output_dir, car, min_score, skip_visualize,
                  batch_size,
                  training_mode == "流式模式",
                  bool(self.keep_intermediates_var.get()),
                  self.cancel_event, self.process_holder, input_dim),
            daemon=True,
        )
        self.worker.start()
        self._update_elapsed()

    def _run_worker(
        self,
        data_dir,
        output_dir,
        car,
        min_score,
        skip_visualize,
        batch_size,
        streaming_mode,
        keep_intermediates,
        cancel_event,
        process_holder,
        input_dim=DEFAULT_INPUT_DIM,
    ) -> None:
        writer = _QueueWriter(self.messages)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                model_path = auto_train(
                    data_dir,
                    car,
                    min_score=min_score,
                    skip_visualize=skip_visualize,
                    skip_corrupt_rlogs=True,
                    output_dir=output_dir,
                    deploy_dir=output_dir,
                    batch_size=batch_size,
                    streaming_mode=streaming_mode,
                    keep_intermediates=keep_intermediates,
                    cancel_event=cancel_event,
                    process_holder=process_holder,
                    input_dim=input_dim,
                )
            self.messages.put(("done", model_path))
        except BaseException as exc:
            self.messages.put(("error", (exc, traceback.format_exc())))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._append_event("训练完成")
                    self._finish_running("训练完成", completed=True)
                    messagebox.showinfo("训练完成", f"模型已输出到：\n{payload}")
                elif kind == "error":
                    if isinstance(payload, tuple) and len(payload) == 2:
                        exc, traceback_text = payload
                    else:
                        exc, traceback_text = payload, ""
                    self._append_training_error(exc, traceback_text)
                    self._finish_running("训练失败")
                    messagebox.showerror("训练失败", str(exc))
                elif self.updater.handle_message(kind, payload):
                    pass
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))


def launch_gui() -> None:
    if tk is None:
        raise RuntimeError("当前 Python 未包含 Tk；Windows 官方 Python 安装包自带 Tk。")
    root = tk.Tk()
    NNLCApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
