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
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # Some minimal Python builds (including CI macOS) omit Tk.
    tk = None
    filedialog = messagebox = ttk = None

from nnlc_auto_train import _terminate_process_tree, auto_train
from nnlc_runtime import application_directory
from nnlc_update import UpdateError, check_for_update, download_update, format_bytes
from nnlc_version import get_version


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_COLOR_TAGS = {
    31: "error",
    32: "success",
    33: "warning",
    34: "info",
    35: "heading",
    36: "step",
    91: "error",
    92: "success",
    93: "warning",
    94: "info",
    95: "heading",
    96: "step",
}
TRAINING_MODES = {
    "CPU 标准模式": 16384,
    "CPU 流式低内存模式": 4096,
}
LEGACY_TRAINING_MODES = {
    "标准模式": "CPU 标准模式",
    "低内存模式": "CPU 流式低内存模式",
    "CPU 低内存模式": "CPU 流式低内存模式",
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


APP_TITLE = "NNLC 横向控制模型训练"
APP_SUBTITLE = "从 reallog 提取数据，完成评分、训练与模型部署"
C_HEADER_BG = "#16324f"
C_HEADER_SUB = "#8fb0d1"
C_BADGE_BG = "#274d72"
C_BADGE_FG = "#d5e4f4"
C_PAGE_BG = "#f2f4f8"
C_CARD_BG = "#ffffff"
C_BORDER = "#e2e7ef"
C_ACCENT = "#2f7de1"
C_ACCENT_HOVER = "#2060b8"
C_ACCENT_DISABLED = "#9db8d2"
C_TEXT = "#2b333d"
C_MUTED = "#7d8590"
C_BTN_BORDER = "#d6dce5"
C_BTN_HOVER = "#f0f3f7"
C_STATUS_BG = "#e8ecf3"
C_DISABLED = "#f5f6f8"
F_TITLE = ("Microsoft YaHei UI", 16, "bold")
F_VERSION = ("Microsoft YaHei UI", 9, "bold")
F_SUB = ("Microsoft YaHei UI", 9)
F_SECTION = ("Microsoft YaHei UI", 11, "bold")
F_BASE = ("Microsoft YaHei UI", 10)
F_BOLD = ("Microsoft YaHei UI", 10, "bold")
F_LOG = ("Consolas", 10)
PAD_PAGE = 20
PAD_CARD = 18
PAD_ROW = 7
INPUT_H = 34
LABEL_W = 92


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
        self.root.geometry("1200x880")
        self.root.minsize(1060, 780)
        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.started_at: float | None = None
        self.config_widgets = []
        self.path_pickers = []
        self._ansi_log_tag: str | None = None
        self.cancel_event = threading.Event()
        self.process_holder = {}

        saved_preferences = self._load_preferences()
        saved_mode = saved_preferences.get("training_mode")
        saved_mode = LEGACY_TRAINING_MODES.get(saved_mode, saved_mode)
        if not isinstance(saved_mode, str) or saved_mode not in TRAINING_MODES:
            saved_mode = "CPU 标准模式"

        saved_car = saved_preferences.get("car")
        if not isinstance(saved_car, str) or not saved_car.strip():
            saved_car = "BYD_TANG_DMI_24"

        self.data_var = tk.StringVar()
        self.output_var = tk.StringVar(value=self._default_output())
        self.car_var = tk.StringVar(value=saved_car)
        self.threshold_var = tk.StringVar()
        self.training_mode_var = tk.StringVar(value=saved_mode)
        self.auto_threshold_var = tk.BooleanVar(value=True)
        self.skip_viz_var = tk.BooleanVar(value=True)
        self.keep_intermediates_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.elapsed_var = tk.StringVar(value="未开始")
        try:
            self.app_version = get_version()
        except Exception:
            self.app_version = "unknown"
        self.root.title(f"{APP_TITLE} v{self.app_version}")
        self.pending_update = None
        self.update_prompt_shown = False
        self.update_in_progress = False
        self.update_cancel_event = threading.Event()
        self.update_thread = None
        self.download_window = None
        self.download_progress = None
        self.download_status_var = None
        self._closing = False
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_messages()
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            self.root.after(1000, self._start_update_check)

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
            return preferences
        except (OSError, ValueError, TypeError):
            return {}

    def _save_preferences(self) -> None:
        car = self.car_var.get().strip()
        training_mode = self.training_mode_var.get()
        training_mode = LEGACY_TRAINING_MODES.get(training_mode, training_mode)
        if training_mode not in TRAINING_MODES:
            training_mode = "CPU 标准模式"
        path = self._preferences_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(".tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "car": car,
                        "training_mode": training_mode,
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
        self._build_statusbar()
        self._build_log()

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
        style.configure("Dialog.TFrame", background=C_PAGE_BG)
        style.configure("DialogCard.TFrame", background=C_CARD_BG)
        style.configure(
            "DialogTitle.TLabel",
            background=C_CARD_BG,
            foreground=C_HEADER_BG,
            font=("Microsoft YaHei UI", 12, "bold"),
        )
        style.configure(
            "DialogBody.TLabel",
            background=C_CARD_BG,
            foreground="#36516d",
            font=F_SUB,
        )
        style.configure(
            "DialogNotes.TLabel",
            background="#f7faff",
            foreground="#36516d",
            padding=(10, 8),
            font=F_SUB,
        )
        style.configure(
            "DialogPrimary.TButton",
            background=C_ACCENT,
            foreground="#ffffff",
            font=("Microsoft YaHei UI", 9, "bold"),
            padding=(16, 7),
        )
        style.map("DialogPrimary.TButton", background=[("active", C_ACCENT_HOVER)])
        style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor="#e8eef5",
            background=C_ACCENT,
            bordercolor="#e8eef5",
            lightcolor=C_ACCENT,
            darkcolor=C_ACCENT,
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
        self._form_row(dir_inner, 0, "reallog 目录", data_picker)
        output_picker = PathPicker(dir_inner, self.output_var, self._choose_output)
        self._form_row(dir_inner, 1, "输出目录", output_picker)
        self.path_pickers = [data_picker, output_picker]

        param_card, param_inner = self._make_card(body, "训练参数")
        param_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        param_inner.columnconfigure(0, minsize=LABEL_W)
        param_inner.columnconfigure(1, weight=1)

        tk.Label(param_inner, text="车型", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=0, column=0, sticky="e", pady=PAD_ROW
        )
        self.car_box = self._fixed_box(param_inner)
        self.car_box.grid(row=0, column=1, sticky="ew", padx=(10, 14), pady=PAD_ROW)
        self.car_entry = self._make_form_entry(self.car_box, self.car_var)
        self.config_widgets.append(self.car_entry)

        tk.Label(param_inner, text="训练模式", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=0, column=2, sticky="e", pady=PAD_ROW
        )
        self.mode_box = self._fixed_box(param_inner, width=200)
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

        tk.Label(param_inner, text="路线阈值", font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=1, column=0, sticky="e", pady=PAD_ROW
        )
        self.thr_box = self._fixed_box(param_inner)
        self.thr_box.grid(row=1, column=1, sticky="ew", padx=(10, 14), pady=PAD_ROW)
        self.threshold_entry = self._make_form_entry(self.thr_box, self.threshold_var)

        check_frame = tk.Frame(param_inner, bg=C_CARD_BG)
        check_frame.grid(row=1, column=2, columnspan=2, sticky="w", pady=PAD_ROW)
        auto_lbl = tk.Label(
            check_frame,
            text="自动推荐",
            font=F_BASE,
            bg=C_CARD_BG,
            fg=C_TEXT,
            cursor="hand2",
        )
        auto_lbl.pack(side="left")
        self.auto_threshold_check = tk.Checkbutton(
            check_frame,
            variable=self.auto_threshold_var,
            bg=C_CARD_BG,
            activebackground=C_CARD_BG,
            selectcolor=C_CARD_BG,
            command=self._toggle_threshold,
        )
        self.auto_threshold_check.pack(side="left")
        auto_lbl.bind("<Button-1>", lambda _e: self.auto_threshold_check.invoke())
        self.config_widgets.append(self.auto_threshold_check)
        self._toggle_threshold()

    def _build_log(self) -> None:
        card, inner = self._make_card(self.root, "运行日志")
        card.pack(fill="both", expand=True, padx=PAD_PAGE, pady=(14, 0))

        actions = tk.Frame(inner, bg=C_CARD_BG)
        actions.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Frame(inner, bg=C_BORDER, height=1).pack(side="bottom", fill="x", pady=(12, 0))

        open_box, self.open_output_button = self._make_button(actions, "打开输出目录", self._open_output)
        open_box.pack(side="left")
        clear_box, self.clear_log_button = self._make_button(actions, "清空日志", self.clear_log)
        clear_box.pack(side="left", padx=(8, 0))
        save_box, self.save_log_button = self._make_button(actions, "保存日志", self.save_log)
        save_box.pack(side="left", padx=(8, 0))

        tk.Label(
            actions,
            textvariable=self.elapsed_var,
            font=F_BASE,
            bg=C_CARD_BG,
            fg=C_MUTED,
        ).pack(side="left", expand=True)

        self.start_button = tk.Button(
            actions,
            text="开始训练",
            font=F_BOLD,
            bg=C_ACCENT,
            fg="#ffffff",
            activebackground=C_ACCENT_HOVER,
            activeforeground="#ffffff",
            bd=0,
            padx=32,
            pady=7,
            cursor="hand2",
            command=self.start,
        )
        self.start_button.pack(side="right")
        self._bind_hover_primary(self.start_button)

        log_area = tk.Frame(inner, bg=C_CARD_BG)
        log_area.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_area,
            wrap="word",
            state="disabled",
            height=16,
            font=F_LOG,
            bg="#fbfcfe",
            fg=C_TEXT,
            insertbackground=C_TEXT,
            selectbackground="#cfe2f5",
            bd=0,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_BORDER,
            padx=8,
            pady=8,
        )
        scroll_y = ttk.Scrollbar(log_area, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("info", foreground="#2563a6")
        self.log.tag_configure("success", foreground="#16803a")
        self.log.tag_configure("warning", foreground="#a15c00")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("heading", foreground="#7a3e9d")
        self.log.tag_configure("step", foreground="#007c91")

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=C_STATUS_BG, height=28)
        bar.pack(fill="x", side="bottom", pady=(14, 0))
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
            row=row, column=0, sticky="e", pady=PAD_ROW
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
            fg=C_TEXT,
            bd=0,
            padx=13,
            pady=4,
            cursor="hand2",
            activebackground=C_BTN_HOVER,
        )
        btn.pack(padx=1, pady=1)
        btn.bind("<Enter>", lambda _e: btn.configure(bg=C_BTN_HOVER if str(btn["state"]) != "disabled" else C_CARD_BG))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=C_CARD_BG))
        return box, btn

    def _bind_hover_primary(self, btn) -> None:
        def on_enter(_event):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=C_ACCENT_HOVER)

        def on_leave(_event):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=C_ACCENT)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def _make_dialog_secondary(self, parent, text, command=None):
        box = tk.Frame(parent, bg=C_BTN_BORDER, width=112, height=34)
        box.pack_propagate(False)
        btn = tk.Button(
            box,
            text=text,
            font=F_BASE,
            command=command,
            bg=C_CARD_BG,
            fg=C_TEXT,
            bd=0,
            cursor="hand2",
            activebackground=C_BTN_HOVER,
        )
        btn.pack(fill="both", expand=True, padx=1, pady=1)
        btn.bind("<Enter>", lambda _e: btn.configure(bg=C_BTN_HOVER))
        btn.bind("<Leave>", lambda _e: btn.configure(bg=C_CARD_BG))
        return box

    def _make_dialog_primary(self, parent, text, command=None):
        box = tk.Frame(parent, bg=C_ACCENT, width=112, height=34)
        box.pack_propagate(False)
        btn = tk.Button(
            box,
            text=text,
            font=F_BOLD,
            command=command,
            bg=C_ACCENT,
            fg="#ffffff",
            bd=0,
            cursor="hand2",
            activebackground=C_ACCENT_HOVER,
            activeforeground="#ffffff",
        )
        btn.pack(fill="both", expand=True)
        self._bind_hover_primary(btn)
        return box

    def _toggle_threshold(self) -> None:
        auto = self.auto_threshold_var.get()
        busy = bool((self.worker and self.worker.is_alive()) or self.update_in_progress)
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

    def _choose_data(self) -> None:
        path = filedialog.askdirectory(title="选择包含 reallog/rlog 的目录")
        if path:
            self.data_var.set(self._normalize_path(path))

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(self._normalize_path(path))

    def _open_output(self) -> None:
        output_dir = Path(self.output_var.get().strip())
        if not output_dir.is_dir():
            messagebox.showinfo("输出目录", "输出目录尚未创建，请先完成一次训练。")
            return
        try:
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(output_dir)])
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc))

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self._ansi_log_tag = None

    def save_log(self) -> None:
        content = self.log.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("保存日志", "当前没有可保存的日志。")
            return
        filename = time.strftime("%Y%m%d_%H%M%S") + "_log.log"
        path = application_directory() / filename
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        messagebox.showinfo("保存日志", f"日志已保存到：\n{path}")

    def _append_log(self, text: str, tag: str | None = None) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.log.configure(state="normal")
        if tag is not None:
            self.log.insert("end", _ANSI_ESCAPE_RE.sub("", text), tag)
        else:
            current_tag = self._ansi_log_tag
            position = 0
            for match in _ANSI_ESCAPE_RE.finditer(text):
                if match.start() > position:
                    self.log.insert("end", text[position:match.start()], current_tag)
                sequence = match.group()
                if sequence.endswith("m"):
                    parameters = sequence[2:-1]
                    codes = [int(value) if value.isdecimal() else 0 for value in parameters.split(";")]
                    for code in codes:
                        if code in (0, 39):
                            current_tag = None
                        elif code in _ANSI_COLOR_TAGS:
                            current_tag = _ANSI_COLOR_TAGS[code]
                position = match.end()
            if position < len(text):
                self.log.insert("end", text[position:], current_tag)
            self._ansi_log_tag = current_tag
        self.log.see("end")
        self.log.configure(state="disabled")

    def _append_event(self, text: str, tag: str) -> None:
        self._append_log(f"[{time.strftime('%H:%M:%S')}] {text}\n", tag)

    def _set_running(self, running: bool) -> None:
        busy = running or self.update_in_progress
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
        if busy:
            self.start_button.configure(state="disabled", bg=C_ACCENT_DISABLED, cursor="arrow")
        else:
            self.start_button.configure(state="normal", bg=C_ACCENT, cursor="hand2")
        self.open_output_button.configure(state=state)
        self.clear_log_button.configure(state=state)
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

    def _finish_running(self, status: str) -> None:
        self._set_running(False)
        self.status_var.set(status)
        if self.pending_update is not None and not self.update_prompt_shown:
            self.root.after(200, self._prompt_pending_update)

    def _on_close(self) -> None:
        self._save_preferences()
        if self.update_in_progress:
            should_close = messagebox.askyesno(
                "正在下载更新",
                "关闭窗口会取消当前更新下载，确定要退出吗？",
                icon="warning",
            )
            if not should_close:
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
        self._closing = True
        if self.update_in_progress:
            self.update_cancel_event.set()
            if self.download_status_var is not None:
                self.download_status_var.set("正在取消...")
            thread = self.update_thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=2)
            manifest = self.pending_update
            if manifest is not None:
                partial_path = application_directory() / (manifest.filename + ".part")
                try:
                    partial_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._close_download_window()
        self.root.destroy()

    def start(self) -> None:
        if self.update_in_progress:
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
            messagebox.showerror("参数错误", "请选择有效的 reallog 目录。")
            return
        if not output_dir:
            messagebox.showerror("参数错误", "请选择输出目录。")
            return
        if not car:
            messagebox.showerror("参数错误", "车型不能为空。")
            return
        self._save_preferences()

        min_score = None
        if not self.auto_threshold_var.get():
            try:
                min_score = int(self.threshold_var.get().strip())
            except ValueError:
                messagebox.showerror("参数错误", "路线阈值必须是 0-100 的整数，或勾选自动推荐。")
                return
            if not 0 <= min_score <= 100:
                messagebox.showerror("参数错误", "路线阈值必须在 0-100 之间。")
                return

        self.clear_log()
        self._append_event("训练任务已启动", "info")
        self._set_running(True)
        self.status_var.set("训练中，请保持窗口打开...")
        self.elapsed_var.set("耗时 00:00")
        self.started_at = time.monotonic()
        self.cancel_event = threading.Event()
        self.process_holder = {}
        training_mode = self.training_mode_var.get().strip()
        batch_size = TRAINING_MODES.get(training_mode)
        if batch_size is None:
            messagebox.showerror("参数错误", "请选择有效的训练模式。")
            self._set_running(False)
            return
        skip_visualize = not self.skip_viz_var.get()
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(data_dir, output_dir, car, min_score, skip_visualize,
                  batch_size,
                  training_mode == "CPU 流式低内存模式",
                  bool(self.keep_intermediates_var.get()),
                  self.cancel_event, self.process_holder),
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
                )
            self.messages.put(("done", model_path))
        except BaseException as exc:
            self.messages.put(("error", exc))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    self._finish_running("训练完成")
                    self._append_event("训练完成", "success")
                    messagebox.showinfo("训练完成", f"模型已输出到：\n{payload}")
                elif kind == "error":
                    self._finish_running("训练失败")
                    self._append_event(f"训练失败：{payload}", "error")
                    messagebox.showerror("训练失败", str(payload))
                elif kind == "update_available":
                    self._handle_update_available(payload)
                elif kind == "update_download_progress":
                    self._update_download_progress(*payload)
                elif kind == "update_downloaded":
                    self._finish_download_success(payload)
                elif kind == "update_failed":
                    self._finish_download_failure(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _should_check_for_updates(self) -> bool:
        return bool(
            getattr(sys, "frozen", False)
            and sys.platform == "win32"
            and self.app_version
            and self.app_version != "unknown"
        )

    def _start_update_check(self) -> None:
        if not self._should_check_for_updates():
            return
        thread = threading.Thread(target=self._check_for_update_worker, daemon=True)
        thread.start()

    def _check_for_update_worker(self) -> None:
        try:
            manifest = check_for_update(self.app_version)
        except Exception:
            return
        if manifest is not None:
            self.messages.put(("update_available", manifest))

    def _handle_update_available(self, manifest) -> None:
        self.pending_update = manifest
        if self.worker and self.worker.is_alive():
            self._append_event(f"发现新版本 v{manifest.version}，将在训练结束后提示。", "info")
            return
        self._prompt_pending_update()

    def _prompt_pending_update(self) -> None:
        manifest = self.pending_update
        if manifest is None or self.update_prompt_shown or self.update_in_progress:
            return
        if self.worker and self.worker.is_alive():
            return
        self.update_prompt_shown = True
        notes = manifest.notes.strip() or "无更新说明。"
        dialog = tk.Toplevel(self.root)
        dialog.title("发现新版本")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(background="#eef3f8")
        choice = {"download": False}

        frame = ttk.Frame(dialog, padding=(18, 16), style="Dialog.TFrame")
        frame.grid(row=0, column=0, sticky="nsew")
        card = ttk.Frame(frame, padding=(16, 14), style="DialogCard.TFrame")
        card.grid(row=0, column=0, sticky="ew")
        ttk.Label(card, text="发现新版本", style="DialogTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        details = (
            f"当前版本：v{self.app_version}\n"
            f"最新版本：v{manifest.version}\n"
            f"文件大小：{format_bytes(manifest.size)}"
        )
        ttk.Label(card, text=details, justify="left", style="DialogBody.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Label(card, text=notes, justify="left", wraplength=450, style="DialogNotes.TLabel").grid(
            row=2, column=0, sticky="ew", pady=(12, 0)
        )
        ttk.Label(
            card,
            text="下载后不会替换正在运行的程序。关闭当前程序后，运行新的版本化 EXE 即可。",
            justify="left",
            wraplength=450,
            style="DialogBody.TLabel",
        ).grid(row=3, column=0, sticky="w", pady=(12, 0))
        button_row = tk.Frame(frame, bg=C_PAGE_BG)
        button_row.grid(row=1, column=0, sticky="e", pady=(16, 0))

        def choose(download: bool) -> None:
            choice["download"] = download
            dialog.destroy()

        self._make_dialog_secondary(button_row, "稍后", lambda: choose(False)).pack(side="right")
        self._make_dialog_primary(button_row, "立即下载", lambda: choose(True)).pack(
            side="right", padx=(0, 8)
        )
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose(False))
        dialog.update_idletasks()
        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()
        x = max(0, (dialog.winfo_screenwidth() - width) // 2)
        y = max(0, (dialog.winfo_screenheight() - height) // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        self.root.wait_window(dialog)
        if choice["download"]:
            self._start_update_download(manifest)
        else:
            self._append_event(f"已跳过 v{manifest.version} 更新下载。", "info")

    def _start_update_download(self, manifest) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("训练进行中", "请等待训练结束后再下载更新。")
            self.update_prompt_shown = False
            return
        self.update_in_progress = True
        self.update_cancel_event = threading.Event()
        self._set_running(False)
        self.status_var.set("正在下载更新...")
        self._show_download_window(manifest)
        self.update_thread = threading.Thread(
            target=self._download_update_worker,
            args=(manifest,),
            daemon=True,
        )
        self.update_thread.start()

    def _show_download_window(self, manifest) -> None:
        window = tk.Toplevel(self.root)
        window.title("下载更新")
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self._request_cancel_download)
        window.configure(background="#eef3f8")
        frame = ttk.Frame(window, padding=(18, 16), style="Dialog.TFrame")
        frame.grid(row=0, column=0, sticky="nsew")
        card = ttk.Frame(frame, padding=(16, 14), style="DialogCard.TFrame")
        card.grid(row=0, column=0, sticky="ew")
        ttk.Label(card, text="正在下载更新", style="DialogTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(card, text=manifest.filename, style="DialogBody.TLabel").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.download_status_var = tk.StringVar(value="准备下载...")
        ttk.Label(card, textvariable=self.download_status_var, style="DialogBody.TLabel").grid(
            row=2, column=0, sticky="w", pady=(12, 8)
        )
        progress = ttk.Progressbar(card, length=420, maximum=100, mode="determinate",
                                   style="Modern.Horizontal.TProgressbar")
        progress.grid(row=3, column=0, sticky="ew")
        cancel_row = tk.Frame(frame, bg=C_PAGE_BG)
        cancel_row.grid(row=1, column=0, sticky="e", pady=(12, 0))
        self._make_dialog_secondary(cancel_row, "取消", self._request_cancel_download).pack(
            side="right"
        )
        window.update_idletasks()
        width = window.winfo_reqwidth()
        height = window.winfo_reqheight()
        x = max(0, (window.winfo_screenwidth() - width) // 2)
        y = max(0, (window.winfo_screenheight() - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
        self.download_window = window
        self.download_progress = progress

    def _request_cancel_download(self) -> None:
        if not self.update_in_progress:
            return
        should_cancel = messagebox.askyesno(
            "取消下载",
            "确定要取消当前更新下载吗？",
            icon="warning",
        )
        if should_cancel:
            self.update_cancel_event.set()
            if self.download_status_var is not None:
                self.download_status_var.set("正在取消...")

    def _update_download_progress(self, downloaded: int, total: int, percent: int) -> None:
        if self.download_progress is not None:
            self.download_progress["value"] = percent
        if self.download_status_var is not None:
            self.download_status_var.set(
                f"{format_bytes(downloaded)} / {format_bytes(total)}  ({percent}%)"
            )

    def _download_update_worker(self, manifest) -> None:
        try:
            path = download_update(
                manifest,
                dest_dir=application_directory(),
                progress_callback=lambda downloaded, total, percent: self.messages.put(
                    ("update_download_progress", (downloaded, total, percent))
                ),
                cancel_event=self.update_cancel_event,
            )
            self.messages.put(("update_downloaded", path))
        except Exception as exc:
            self.messages.put(("update_failed", exc))

    def _close_download_window(self) -> None:
        window = self.download_window
        self.download_window = None
        self.download_progress = None
        self.download_status_var = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    def _finish_download_success(self, path) -> None:
        self.update_in_progress = False
        self.update_thread = None
        self._close_download_window()
        if self._closing:
            return
        self._set_running(False)
        self.status_var.set("更新已下载")
        self._append_event(f"新版本已准备好：{path}", "success")
        should_open = messagebox.askyesno(
            "更新已下载",
            "新程序已准备好：\n"
            f"{path}\n\n"
            "关闭当前程序后运行这个新 EXE 即可，原来的 Julia 环境可以继续使用。\n\n"
            "是否打开所在目录？",
        )
        if should_open:
            self._open_path(Path(path).parent)

    def _finish_download_failure(self, exc) -> None:
        self.update_in_progress = False
        self.update_thread = None
        self._close_download_window()
        if self._closing:
            return
        self._set_running(False)
        self.status_var.set("更新下载失败")
        message = str(exc)
        if isinstance(exc, UpdateError) and "取消" in message:
            self._append_event("已取消更新下载。", "warning")
            messagebox.showinfo("已取消下载", "未完成的下载文件已删除。")
            return
        self._append_event(f"更新下载失败：{message}", "error")
        messagebox.showerror("更新下载失败", f"{message}\n未完成的下载文件已删除。")

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
