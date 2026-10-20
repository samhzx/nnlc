"""Tkinter interface for the standalone C3 rlog downloader."""

from __future__ import annotations

import ipaddress
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from c3_rlog_downloader.settings import default_destination, load_settings, save_settings, settings_directory
from c3_rlog_downloader.transfer import (
    CleanupPreview,
    HostKeyMismatch,
    InvalidKeyPassphrase,
    KeyPassphraseRequired,
    TransferCancelled,
    TransferProgress,
    close_sftp,
    connect_sftp,
    run_cleanup,
    run_download,
)

APP_TITLE = "C3 rlog 下载器"
APP_VERSION = "1.1.0"
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
C_DANGER = "#ff4d4f"
C_TEXT = "#2b333d"
C_MUTED = "#7d8590"
C_BTN_TEXT = "#606266"
C_BTN_TEXT_HOVER = "#303133"
C_BTN_BORDER = "#dcdfe6"
C_BTN_BORDER_HOVER = "#c0c4cc"
C_BTN_DISABLED_BG = "#f5f7fa"
C_BTN_DISABLED_FG = "#c0c4cc"
C_BTN_DISABLED_BORDER = "#e4e7ed"
C_CONNECTING_BG = "#e4e7ed"
C_CONNECTING_FG = "#909399"
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
LABEL_W = 86


def _format_size(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(max(0, value))
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _format_rate(value: float) -> str:
    if value <= 0:
        return "--"
    return f"{_format_size(int(value))}/s"


def _format_eta(value: float | None) -> str:
    if value is None:
        return "--:--"
    seconds = max(0, int(round(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class PathPicker(tk.Frame):
    """Path entry and compact browse control styled like the NNLC GUI."""

    def __init__(self, parent, variable, browse_command) -> None:
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
        self.enabled = True
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
        self.entry.bind(
            "<FocusIn>", lambda _event: self.configure(highlightbackground=C_ACCENT)
        )
        self.entry.bind(
            "<FocusOut>", lambda _event: self.configure(highlightbackground=C_BORDER)
        )
        tk.Frame(self, bg=C_BORDER, width=1).pack(side="left", fill="y", pady=8)
        self.button = tk.Label(
            self,
            text="···",
            font=("Microsoft YaHei UI", 11, "bold"),
            bg=C_CARD_BG,
            fg=C_MUTED,
            cursor="hand2",
            padx=10,
            anchor="center",
        )
        self.button.pack(side="left", fill="both")
        self.button.bind("<Button-1>", lambda _event: self._browse(browse_command))
        self.button.bind("<Enter>", self._on_enter)
        self.button.bind("<Leave>", lambda _event: self.button.configure(fg=C_MUTED))

    def _browse(self, command) -> None:
        if self.enabled:
            command()

    def _on_enter(self, _event=None) -> None:
        self.button.configure(fg=C_ACCENT if self.enabled else C_MUTED)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        background = C_CARD_BG if enabled else C_DISABLED
        self.entry.configure(state="normal" if enabled else "disabled", bg=background)
        self.configure(bg=background)
        self.button.configure(bg=background, cursor="hand2" if enabled else "arrow")


class C3DownloaderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings_dir = settings_directory()
        settings = load_settings(self.settings_dir)
        self.host_var = tk.StringVar(value=settings.get("host", ""))
        self.key_var = tk.StringVar(value=settings.get("key_path", ""))
        self.destination_var = tk.StringVar(
            value=settings.get("destination") or default_destination()
        )
        self.connection_status_var = tk.StringVar(value="● 未连接")
        self.status_var = tk.StringVar(value="就绪")
        self.progress_var = tk.StringVar(value="0.0%")
        self.progress_detail_var = tk.StringVar(value="尚未开始下载")
        self.speed_var = tk.StringVar(value="--")
        self.eta_var = tk.StringVar(value="--:--")
        self.current_var = tk.StringVar(value="")
        self.messages: queue.Queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.busy = False
        self.closing = False
        self.passphrase: str | None = None
        self.passphrase_attempts = 0
        self.operation: str | None = None
        self._pending_cleanup: CleanupPreview | None = None
        self.connection_client = None
        self.connection_sftp = None
        self.connected_host: str | None = None
        self.connected_key_path: str | None = None
        self.form_controls = []
        self.path_pickers = []

        root.title(f"{APP_TITLE} v{APP_VERSION}")
        root.geometry("900x700")
        root.minsize(780, 620)
        root.configure(bg=C_PAGE_BG)
        self._set_window_icon()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        self._poll_messages()

    def _set_window_icon(self) -> None:
        if sys.platform != "win32":
            return
        resource_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_path = resource_root / "C3_Rlog_Downloader.ico"
        if icon_path.is_file():
            try:
                self.root.iconbitmap(default=str(icon_path))
            except tk.TclError:
                pass

    def _build(self) -> None:
        self._build_header()
        self._build_statusbar()

        body = tk.Frame(self.root, bg=C_PAGE_BG)
        body.pack(fill="x", padx=PAD_PAGE, pady=(16, 0))
        body.columnconfigure(0, weight=1, uniform="col")
        body.columnconfigure(1, weight=1, uniform="col")

        connection_card, connection_inner = self._make_card(
            body, "设备连接", self.connection_status_var
        )
        connection_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        connection_inner.columnconfigure(0, minsize=LABEL_W)
        connection_inner.columnconfigure(1, weight=1)
        host_actions = tk.Frame(connection_inner, bg=C_CARD_BG)
        host_actions.grid(row=0, column=1, sticky="ew", padx=(10, 0), pady=PAD_ROW)
        host_box = self._fixed_box(host_actions)
        self.host_box = host_box
        self.host_entry = self._make_form_entry(host_box, self.host_var)
        host_box.pack(side="left", fill="x", expand=True)
        self.connection_button_box = tk.Frame(
            host_actions, bg=C_ACCENT, height=INPUT_H
        )
        self.connection_button_box.pack_propagate(False)
        self.connection_button_box.pack(side="right", padx=(8, 0))
        self.connection_button = tk.Button(
            self.connection_button_box,
            text="连接设备",
            font=F_BASE,
            command=self.connect,
            bg=C_ACCENT,
            fg="#ffffff",
            activebackground=C_ACCENT_HOVER,
            activeforeground="#ffffff",
            disabledforeground="#ffffff",
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=13,
            cursor="hand2",
        )
        self.connection_button.pack(fill="both", expand=True, padx=1, pady=1)
        self._fit_connection_button_width()
        self.connection_button.bind("<Enter>", self._connection_button_enter)
        self.connection_button.bind("<Leave>", self._connection_button_leave)
        tk.Label(
            connection_inner,
            text="C3 IP 地址",
            font=F_BASE,
            bg=C_CARD_BG,
            fg=C_TEXT,
        ).grid(row=0, column=0, sticky="e", pady=PAD_ROW)
        self.form_controls.append(self.host_entry)
        key_picker = PathPicker(connection_inner, self.key_var, self._browse_key)
        self._form_row(connection_inner, 1, "SSH 私钥", key_picker)
        self.key_picker = key_picker
        self.path_pickers.append(key_picker)

        destination_card, destination_inner = self._make_card(body, "下载设置")
        destination_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        destination_inner.columnconfigure(0, minsize=LABEL_W)
        destination_inner.columnconfigure(1, weight=1)
        destination_picker = PathPicker(
            destination_inner, self.destination_var, self._browse_destination
        )
        self._form_row(destination_inner, 0, "保存目录", destination_picker)
        self.destination_picker = destination_picker
        self.path_pickers.append(destination_picker)
        source_box = self._fixed_box(destination_inner)
        source_box.configure(bg=C_DISABLED)
        tk.Label(
            source_box,
            text="/data/media/0/realdata",
            font=F_BASE,
            bg=C_DISABLED,
            fg=C_MUTED,
            anchor="w",
        ).pack(fill="both", expand=True, padx=10)
        self._form_row(destination_inner, 1, "设备目录", source_box)

        progress_card, progress_inner = self._make_card(body, "下载进度")
        progress_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        progress_summary = tk.Frame(progress_inner, bg=C_CARD_BG)
        progress_summary.pack(fill="x")
        progress_summary.columnconfigure(0, weight=2)
        progress_summary.columnconfigure(1, weight=1)
        progress_summary.columnconfigure(2, weight=1)
        progress_summary.columnconfigure(3, weight=1)
        percent_box = tk.Frame(progress_summary, bg=C_CARD_BG)
        percent_box.grid(row=0, column=0, sticky="w")
        tk.Label(
            percent_box,
            textvariable=self.progress_var,
            font=("Microsoft YaHei UI", 16, "bold"),
            bg=C_CARD_BG,
            fg=C_ACCENT,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            percent_box,
            textvariable=self.progress_detail_var,
            font=F_SUB,
            bg=C_CARD_BG,
            fg=C_MUTED,
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))
        self._make_progress_metric(progress_summary, 1, "实时速率", self.speed_var)
        self._make_progress_metric(progress_summary, 2, "剩余时间", self.eta_var)
        self._make_progress_metric(
            progress_summary, 3, "当前状态", self.status_var, compact=True
        )
        self.progress_track = tk.Frame(
            progress_inner,
            bg=C_STATUS_BG,
            height=12,
            highlightthickness=1,
            highlightbackground=C_BORDER,
        )
        self.progress_track.pack(fill="x", pady=(9, 0))
        self.progress_track.pack_propagate(False)
        self.progress_fill = tk.Frame(self.progress_track, bg=C_ACCENT)
        self.progress_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)
        self.current_label = tk.Label(
            progress_inner,
            textvariable=self.current_var,
            font=F_SUB,
            bg=C_CARD_BG,
            fg=C_MUTED,
            anchor="w",
        )
        self.current_label.pack(fill="x", pady=(5, 0))

        log_card, log_inner = self._make_card(self.root, "运行日志")
        log_card.pack(fill="both", expand=True, padx=PAD_PAGE, pady=(14, 0))
        actions = tk.Frame(log_inner, bg=C_CARD_BG)
        actions.pack(side="bottom", fill="x", pady=(10, 0))
        tk.Frame(log_inner, bg=C_BORDER, height=1).pack(
            side="bottom", fill="x", pady=(12, 0)
        )
        self.open_destination_button_box, self.open_destination_button = self._make_button(
            actions, "打开下载目录", self._open_destination
        )
        self.open_destination_button_box.pack(side="left")
        self.clear_log_button_box, self.clear_log_button = self._make_button(
            actions, "清空日志", self.clear_log
        )
        self.clear_log_button_box.pack(side="left", padx=(8, 0))
        self.cancel_button_box, self.cancel_button = self._make_button(
            actions, "取消", self.cancel
        )
        self.start_button = tk.Button(
            actions,
            text="开始下载",
            font=F_BOLD,
            command=self.start,
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
        )
        self.start_button.pack(side="right")
        self.cancel_button_box.pack(side="right", padx=(0, 8))
        self._bind_primary_hover(self.start_button)

        log_area = tk.Frame(log_inner, bg=C_CARD_BG)
        log_area.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_area,
            wrap="word",
            state="disabled",
            height=8,
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
        scrollbar = ttk.Scrollbar(log_area, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("info", foreground="#2563a6")
        self.log.tag_configure("success", foreground="#16803a")
        self.log.tag_configure("warning", foreground="#a15c00")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("muted", foreground=C_MUTED)
        self._refresh_controls()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=C_HEADER_BG, height=84)
        header.pack(fill="x")
        header.pack_propagate(False)
        left = tk.Frame(header, bg=C_HEADER_BG)
        left.pack(side="left", padx=24, pady=13, fill="y")
        tk.Label(left, text=APP_TITLE, font=F_TITLE, bg=C_HEADER_BG, fg="#ffffff").pack(
            anchor="w"
        )
        tk.Label(
            left,
            text="通过 SSH 连接设备，按本机导出日期归档 rlog.zst",
            font=F_SUB,
            bg=C_HEADER_BG,
            fg=C_HEADER_SUB,
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            header,
            text=f" v{APP_VERSION} ",
            font=F_VERSION,
            bg=C_BADGE_BG,
            fg=C_BADGE_FG,
            padx=8,
            pady=3,
        ).pack(side="right", padx=24)

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=C_STATUS_BG, height=28)
        bar.pack(fill="x", side="bottom", pady=(14, 0))
        bar.pack_propagate(False)
        tk.Label(
            bar, textvariable=self.status_var, font=F_SUB, bg=C_STATUS_BG, fg=C_TEXT
        ).pack(side="left", padx=14, pady=4)
        tk.Label(
            bar, text=f"v{APP_VERSION}", font=F_SUB, bg=C_STATUS_BG, fg=C_MUTED
        ).pack(side="right", padx=14, pady=4)

    def _make_card(self, parent, title, status_variable=None):
        wrapper = tk.Frame(parent, bg=C_PAGE_BG)
        header = tk.Frame(wrapper, bg=C_PAGE_BG)
        header.pack(anchor="w", pady=(0, 8))
        tk.Frame(header, bg=C_ACCENT, width=4, height=15).pack(side="left")
        tk.Label(header, text=title, font=F_SECTION, bg=C_PAGE_BG, fg=C_TEXT).pack(
            side="left", padx=(8, 0)
        )
        if status_variable is not None:
            self.connection_status_label = tk.Label(
                header,
                textvariable=status_variable,
                font=F_SUB,
                bg=C_PAGE_BG,
                fg=C_MUTED,
            )
            self.connection_status_label.pack(side="left", padx=(10, 0))
        card = tk.Frame(
            wrapper,
            bg=C_CARD_BG,
            highlightthickness=1,
            highlightbackground=C_BORDER,
            highlightcolor=C_BORDER,
        )
        card.pack(fill="both", expand=True)
        inner = tk.Frame(card, bg=C_CARD_BG)
        inner.pack(fill="both", expand=True, padx=PAD_CARD, pady=PAD_CARD - 4)
        return wrapper, inner

    def _make_progress_metric(
        self, parent, column: int, label: str, variable, compact: bool = False
    ) -> None:
        box = tk.Frame(parent, bg=C_CARD_BG)
        box.grid(row=0, column=column, sticky="ew", padx=(12, 0))
        tk.Frame(box, bg=C_BORDER, width=1).pack(side="left", fill="y", pady=2)
        inner = tk.Frame(box, bg=C_CARD_BG)
        inner.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tk.Label(
            inner,
            text=label,
            font=("Microsoft YaHei UI", 9),
            bg=C_CARD_BG,
            fg=C_MUTED,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            inner,
            textvariable=variable,
            font=("Microsoft YaHei UI", 10, "bold" if not compact else "normal"),
            bg=C_CARD_BG,
            fg=C_TEXT if not compact else C_MUTED,
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

    def _form_label(self, parent, row: int, label: str) -> None:
        tk.Label(parent, text=label, font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=row, column=0, sticky="e", pady=PAD_ROW
        )

    def _form_row(self, parent, row: int, label: str, widget) -> None:
        self._form_label(parent, row, label)
        widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=PAD_ROW)

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
        entry.bind(
            "<FocusIn>", lambda _event: box.configure(highlightbackground=C_ACCENT)
        )
        entry.bind(
            "<FocusOut>", lambda _event: box.configure(highlightbackground=C_BORDER)
        )
        return entry

    def _make_button(self, parent, text: str, command=None):
        box = tk.Frame(parent, bg=C_BTN_BORDER)
        button = tk.Button(
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
        button.pack(padx=1, pady=1)
        button.bind("<Enter>", lambda _event: self._on_secondary_button_enter(box, button))
        button.bind("<Leave>", lambda _event: self._on_secondary_button_leave(box, button))
        return box, button

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

    def _fit_connection_button_width(self) -> None:
        current = self.connection_button.cget("text")
        max_width = 0
        for label in ("连接设备", "断开连接", "连接中..."):
            self.connection_button.configure(text=label)
            self.connection_button.update_idletasks()
            max_width = max(max_width, self.connection_button.winfo_reqwidth())
        self.connection_button.configure(text=current)
        self.connection_button_box.configure(width=max_width + 2)

    def _form_row(self, parent, row: int, label: str, widget) -> None:
        tk.Label(parent, text=label, font=F_BASE, bg=C_CARD_BG, fg=C_TEXT).grid(
            row=row, column=0, sticky="e", pady=PAD_ROW
        )
        widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=PAD_ROW)

    def _browse_key(self) -> None:
        current = self.key_var.get().strip()
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择 SSH 私钥",
            initialdir=str(Path(current).parent) if current else str(Path.home()),
            filetypes=(("所有文件", "*"),),
        )
        if path:
            self.key_var.set(path)

    def _browse_destination(self) -> None:
        current = self.destination_var.get().strip()
        path = filedialog.askdirectory(
            parent=self.root,
            title="选择下载目录",
            initialdir=current if Path(current).is_dir() else str(Path.home()),
            mustexist=False,
        )
        if path:
            self.destination_var.set(path)
            try:
                self._persist_settings()
            except OSError as exc:
                messagebox.showerror("设置保存失败", str(exc), parent=self.root)

    def _open_destination(self) -> None:
        path = Path(self.destination_var.get().strip()).expanduser()
        if not path.is_dir():
            messagebox.showinfo(
                "下载目录", "下载目录尚未创建，请先选择一个有效目录。", parent=self.root
            )
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("打开失败", str(exc), parent=self.root)

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _set_progress(self, completed: int, total: int) -> None:
        ratio = 0.0 if total <= 0 else min(max(completed / total, 0.0), 1.0)
        self.progress_fill.place_configure(relwidth=ratio)

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {text}\n", tag or "")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_connection_status(self, text: str, connected: bool = False, busy: bool = False) -> None:
        self.connection_status_var.set(text)
        if connected:
            color = "#16803a"
        elif busy:
            color = C_ACCENT
        else:
            color = C_MUTED
        self.connection_status_label.configure(fg=color)

    def _set_connection_button_style(self, connected: bool, connecting: bool = False) -> None:
        if connecting:
            self.connection_button_box.configure(bg=C_CONNECTING_BG)
            self.connection_button.configure(
                text="连接中...",
                command=self.connect,
                bg=C_CONNECTING_BG,
                fg=C_CONNECTING_FG,
                disabledforeground=C_CONNECTING_FG,
                activebackground=C_CONNECTING_BG,
                activeforeground=C_CONNECTING_FG,
            )
            return
        if connected:
            self.connection_button_box.configure(bg=C_DANGER)
            self.connection_button.configure(
                text="断开连接",
                command=self.disconnect,
                bg=C_CARD_BG,
                fg=C_DANGER,
                disabledforeground=C_DANGER,
                activebackground=C_DANGER,
                activeforeground="#ffffff",
            )
            return
        self.connection_button_box.configure(bg=C_ACCENT)
        self.connection_button.configure(
            text="连接设备",
            command=self.connect,
            bg=C_ACCENT,
            fg="#ffffff",
            disabledforeground="#ffffff",
            activebackground=C_ACCENT_HOVER,
            activeforeground="#ffffff",
        )

    def _connection_button_enter(self, _event=None) -> None:
        if str(self.connection_button["state"]) == "disabled":
            return
        if self.connection_sftp is not None:
            self.connection_button_box.configure(bg=C_DANGER)
            self.connection_button.configure(bg=C_DANGER, fg="#ffffff")
            return
        self.connection_button_box.configure(bg=C_ACCENT_HOVER)
        self.connection_button.configure(bg=C_ACCENT_HOVER)

    def _connection_button_leave(self, _event=None) -> None:
        self._set_connection_button_style(
            connected=self.connection_sftp is not None,
            connecting=self.busy and self.connection_sftp is None,
        )

    def _refresh_controls(self) -> None:
        connected = self.connection_sftp is not None
        input_enabled = not self.busy and not connected
        self.host_entry.configure(state="normal" if input_enabled else "disabled")
        self.host_box.configure(bg=C_CARD_BG if input_enabled else C_DISABLED)
        self.key_picker.set_enabled(input_enabled)
        self.destination_picker.set_enabled(not self.busy)
        self._set_connection_button_style(
            connected, connecting=self.busy and not connected
        )
        self.connection_button.configure(
            state="disabled" if self.busy else "normal",
            cursor="arrow" if self.busy else "hand2",
        )
        can_start = connected and not self.busy
        self.start_button.configure(
            state="normal" if can_start else "disabled",
            bg=C_ACCENT if can_start else C_ACCENT_DISABLED,
            fg="#ffffff",
            disabledforeground="#ffffff",
            activebackground=C_ACCENT_HOVER if can_start else C_ACCENT_DISABLED,
            cursor="hand2" if can_start else self._disabled_cursor(),
        )
        self._style_secondary_button(
            self.cancel_button_box, self.cancel_button, enabled=self.busy
        )
        self._style_secondary_button(
            self.open_destination_button_box,
            self.open_destination_button,
            enabled=not self.busy,
        )
        self._style_secondary_button(
            self.clear_log_button_box, self.clear_log_button, enabled=not self.busy
        )

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        self._refresh_controls()

    def _validate_connection_inputs(self):
        host = self.host_var.get().strip()
        try:
            ipaddress.ip_address(host)
        except ValueError:
            messagebox.showerror(
                "IP 地址无效", "请输入有效的 C3 IPv4 或 IPv6 地址。", parent=self.root
            )
            return None
        key_path = Path(self.key_var.get().strip()).expanduser()
        if not key_path.is_file():
            messagebox.showerror(
                "私钥无效", "请选择存在的 SSH 私钥文件。", parent=self.root
            )
            return None
        return host, key_path

    def _persist_settings(
        self,
        host: str | None = None,
        key_path: str | None = None,
        destination: str | None = None,
    ) -> None:
        save_settings(
            self.host_var.get() if host is None else host,
            self.key_var.get() if key_path is None else key_path,
            self.destination_var.get() if destination is None else destination,
            self.settings_dir,
        )

    def connect(self) -> None:
        if self.busy or self.connection_sftp is not None:
            return
        values = self._validate_connection_inputs()
        if values is None:
            return
        host, key_path = values
        try:
            self._persist_settings(host=host, key_path=str(key_path))
        except OSError as exc:
            messagebox.showerror("设置保存失败", str(exc), parent=self.root)
            return

        self.passphrase = None
        self.passphrase_attempts = 0
        self.cancel_event = threading.Event()
        self.operation = "connect"
        self.status_var.set("正在连接设备")
        self._set_connection_status("● 连接中...", busy=True)
        self._append_log(f"开始连接 {host}:22")
        self._set_busy(True)
        self._start_connect_worker(host, str(key_path))

    def _start_connect_worker(self, host: str, key_path: str) -> None:
        self.worker = threading.Thread(
            target=self._connect_worker,
            args=(host, key_path, self.passphrase),
            daemon=True,
        )
        self.worker.start()

    def _connect_worker(
        self, host: str, key_path: str, passphrase: str | None
    ) -> None:
        try:
            client, sftp = connect_sftp(
                host, key_path, self.settings_dir, passphrase
            )
            self.messages.put(("connected", (client, sftp, host, key_path)))
        except KeyPassphraseRequired:
            self.messages.put(("passphrase_required", "connect"))
        except InvalidKeyPassphrase:
            self.messages.put(("passphrase_invalid", "connect"))
        except TransferCancelled as exc:
            self.messages.put(("cancelled", str(exc)))
        except HostKeyMismatch as exc:
            self.messages.put(("failed", str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface unexpected transport failures in the UI.
            self.messages.put(("failed", str(exc)))

    def _on_connected(self, payload) -> None:
        client, sftp, host, key_path = payload
        self.connection_client = client
        self.connection_sftp = sftp
        self.connected_host = host
        self.connected_key_path = key_path
        self._set_connection_status("● 已连接", connected=True)
        self._set_busy(False)
        self.operation = None
        self.passphrase = None
        self.status_var.set("设备已连接")
        self._append_log("SSH 身份验证成功，已打开 SFTP", "success")
        self._append_log("设备已连接，可以开始下载", "success")

    def disconnect(self) -> None:
        if self.busy or self.connection_sftp is None:
            return
        client, sftp = self.connection_client, self.connection_sftp
        self.connection_client = None
        self.connection_sftp = None
        self.connected_host = None
        self.connected_key_path = None
        try:
            close_sftp(client, sftp)
        except Exception as exc:  # noqa: BLE001 - cleanup must not leave stale UI state.
            self._append_log(f"断开连接时出现提示：{exc}", "warning")
        self._set_connection_status("● 未连接")
        self.status_var.set("设备未连接")
        self._append_log("已断开设备连接", "muted")
        self._refresh_controls()

    def start(self) -> None:
        if self.busy:
            return
        if self.connection_sftp is None or self.connected_host is None:
            messagebox.showinfo("尚未连接", "请先点击“连接设备”并等待连接成功。", parent=self.root)
            return
        destination_value = self.destination_var.get().strip()
        if not destination_value:
            messagebox.showerror("目录无效", "请选择下载目录。", parent=self.root)
            return
        destination = Path(destination_value).expanduser()
        try:
            destination.mkdir(parents=True, exist_ok=True)
            self._persist_settings(
                host=self.connected_host,
                key_path=self.connected_key_path or self.key_var.get().strip(),
                destination=str(destination),
            )
        except OSError as exc:
            messagebox.showerror("目录不可用", str(exc), parent=self.root)
            return

        self.cancel_event = threading.Event()
        self._set_progress(0, 1)
        self.progress_var.set("0.0%")
        self.progress_detail_var.set("正在扫描远端 rlog.zst")
        self.speed_var.set("--")
        self.eta_var.set("--:--")
        self.current_var.set("")
        self.status_var.set("扫描 C3 日志")
        self._append_log("开始扫描设备目录 /data/media/0/realdata")
        self._set_busy(True)
        self._start_download_worker(destination)

    def _start_download_worker(self, destination: Path) -> None:
        if self.connection_sftp is None or self.connected_host is None:
            self._finish("设备未连接")
            return
        self.operation = "download"
        connection = (self.connection_client, self.connection_sftp)
        self.worker = threading.Thread(
            target=self._download_worker,
            args=(
                self.connected_host,
                self.connected_key_path or self.key_var.get().strip(),
                destination,
                self.cancel_event,
                connection,
            ),
            daemon=True,
        )
        self.worker.start()

    def _download_worker(
        self,
        host: str,
        key_path: str,
        destination: Path,
        cancel_event: threading.Event,
        connection,
    ) -> None:
        try:
            result, cleanup_preview, inventory = run_download(
                host,
                key_path,
                destination,
                self.settings_dir,
                None,
                cancel_event,
                progress_callback=lambda value: self.messages.put(("progress", value)),
                connection=connection,
            )
            self.messages.put(
                ("download_complete", (result, cleanup_preview, inventory))
            )
        except TransferCancelled as exc:
            self.messages.put(("cancelled", str(exc)))
        except HostKeyMismatch as exc:
            self.messages.put(("failed", str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface unexpected transport failures in the UI.
            self.messages.put(("failed", str(exc)))

    def _start_cleanup_worker(self, preview: CleanupPreview) -> None:
        if self.connection_sftp is None or self.connected_host is None:
            self._finish("设备未连接")
            return
        self.cancel_event = threading.Event()
        self.operation = "cleanup"
        self.status_var.set("正在清理设备日志")
        self.progress_var.set("0.0%")
        self.progress_detail_var.set("清理设备文件")
        self.speed_var.set("--")
        self.eta_var.set("--:--")
        self.current_var.set("")
        self._set_progress(0, preview.total_bytes)
        connection = (self.connection_client, self.connection_sftp)
        self.worker = threading.Thread(
            target=self._cleanup_worker,
            args=(
                self.connected_host,
                self.connected_key_path or self.key_var.get().strip(),
                preview,
                self.cancel_event,
                connection,
            ),
            daemon=True,
        )
        self.worker.start()

    def _cleanup_worker(
        self,
        host: str,
        key_path: str,
        preview: CleanupPreview,
        cancel_event: threading.Event,
        connection,
    ) -> None:
        try:
            result = run_cleanup(
                host,
                key_path,
                self.settings_dir,
                None,
                preview,
                cancel_event,
                progress_callback=lambda value: self.messages.put(("progress", value)),
                connection=connection,
            )
            self.messages.put(("cleanup_complete", result))
        except TransferCancelled as exc:
            self.messages.put(("cancelled", str(exc)))
        except HostKeyMismatch as exc:
            self.messages.put(("failed", str(exc)))
        except Exception as exc:  # noqa: BLE001 - surface unexpected transport failures in the UI.
            self.messages.put(("failed", str(exc)))

    def cancel(self) -> None:
        if self.busy:
            self.cancel_event.set()
            self.status_var.set("正在取消，请稍候...")
            self._style_secondary_button(
                self.cancel_button_box, self.cancel_button, enabled=False
            )

    def _ask_for_passphrase(self, stage: str, invalid: bool = False) -> None:
        if invalid:
            self.passphrase_attempts += 1
            if self.passphrase_attempts >= 3:
                self._finish("密钥口令验证失败")
                messagebox.showerror(
                    "密钥口令错误",
                    "连续三次无法解锁私钥，请检查密钥文件和口令。",
                    parent=self.root,
                )
                return
        prompt = "SSH 私钥需要口令，请输入。" if not invalid else "口令不正确，请重试。"
        value = simpledialog.askstring(
            "SSH 私钥口令", prompt, parent=self.root, show="*"
        )
        if value is None:
            self._finish("已取消")
            self._append_log("用户取消了私钥口令输入", "muted")
            return
        if value == "":
            messagebox.showerror("口令为空", "密钥口令不能为空。", parent=self.root)
            self._ask_for_passphrase(stage, invalid=True)
            return
        self.passphrase = value
        self.cancel_event = threading.Event()
        if stage == "connect":
            self._start_connect_worker(
                self.host_var.get().strip(), self.key_var.get().strip()
            )
        else:
            self._pending_cleanup = getattr(self, "_pending_cleanup", None)
            if self._pending_cleanup is not None:
                self._start_cleanup_worker(self._pending_cleanup)

    def _finish(self, status: str) -> None:
        was_connecting = self.operation == "connect"
        self._set_busy(False)
        self.operation = None
        self.status_var.set(status)
        self.passphrase = None
        self._pending_cleanup = None
        if was_connecting and self.connection_sftp is None:
            self._set_connection_status("● 未连接")

    def _on_download_complete(self, payload) -> None:
        result, cleanup_preview, inventory = payload
        self._set_busy(False)
        self._set_progress(result.completed_bytes, result.total_bytes)
        ratio = (
            0.0
            if result.total_bytes <= 0
            else min(max(result.completed_bytes / result.total_bytes, 0.0), 1.0)
        )
        self.progress_var.set(f"{ratio * 100:.1f}%")
        self.progress_detail_var.set(
            f"总体 {_format_size(result.completed_bytes)} / {_format_size(result.total_bytes)}"
            f"    文件 {result.completed_files}/{result.planned_files}"
        )
        self.eta_var.set("00:00")
        self._append_log(
            f"扫描到 {len(inventory.rlog_files)} 个稳定 rlog，"
            f"跳过最近变化的 rlog {len(inventory.recent_rlogs)} 个。"
        )
        if result.failures:
            for failure in result.failures:
                self._append_log(
                    f"下载失败：{failure.remote_path}：{failure.message}", "error"
                )
            self._finish("下载有失败项，设备日志未清理")
            messagebox.showwarning(
                "下载未全部完成",
                f"{len(result.failures)} 个 rlog 下载失败。设备日志不会被清理。\n\n"
                + "\n".join(f.remote_path for f in result.failures[:8]),
                parent=self.root,
            )
            return
        if not result.planned_files:
            self._finish("没有可下载的稳定 rlog")
            self._append_log("未发现可下载的 rlog.zst；近期活动文件已保留。", "muted")
            return
        self._append_log(
            f"下载完成：{result.downloaded_files} 个新增，"
            f"{result.skipped_existing} 个已存在，传输 {_format_size(result.downloaded_bytes)}。",
            "success",
        )
        if cleanup_preview is None:
            self._finish("下载完成，无法安全预览设备清理范围")
            messagebox.showwarning(
                "无法预览清理范围",
                "下载已完成，但无法读取设备目录清理清单，因此未执行清理。",
                parent=self.root,
            )
            return
        if not cleanup_preview.entries:
            self._finish("下载完成，没有可清理的旧文件")
            self._append_log(
                f"近期文件保留 {cleanup_preview.recent_count} 个。", "muted"
            )
            return
        answer = messagebox.askyesno(
            "下载完成，是否清理 C3 日志？",
            "所有稳定的 rlog.zst 已成功下载。\n\n"
            f"将清理 realdata 下约 {len(cleanup_preview.entries)} 个文件，"
            f"共 {_format_size(cleanup_preview.total_bytes)}。\n"
            "清理范围也包括本程序未下载的其他日志和摄像头文件。\n"
            "文件删除后会一并删除已空的路线目录。\n"
            f"最近 5 分钟有变化的 {cleanup_preview.recent_count} 个文件会保留。\n\n"
            "此操作不可撤销，是否继续？",
            parent=self.root,
        )
        if answer:
            self._pending_cleanup = cleanup_preview
            self._set_busy(True)
            self._start_cleanup_worker(cleanup_preview)
        else:
            self._finish("下载完成，设备日志已保留")

    def _handle_message(self, kind: str, payload) -> None:
        if kind == "connected":
            self._on_connected(payload)
        elif kind == "progress":
            progress: TransferProgress = payload
            self._set_progress(progress.completed_bytes, progress.total_bytes)
            if progress.phase == "cleanup":
                self.status_var.set("正在清理 C3 日志")
            else:
                self.status_var.set("正在下载 rlog.zst")
            ratio = (
                0.0
                if progress.total_bytes <= 0
                else min(max(progress.completed_bytes / progress.total_bytes, 0.0), 1.0)
            )
            self.progress_var.set(f"{ratio * 100:.1f}%")
            self.progress_detail_var.set(
                f"总体 {_format_size(progress.completed_bytes)} / {_format_size(progress.total_bytes)}"
                f"    文件 {progress.completed_files}/{progress.total_files}"
            )
            self.speed_var.set(_format_rate(progress.speed_bytes_per_second))
            self.eta_var.set(_format_eta(progress.eta_seconds))
            self.current_var.set(
                f"{progress.current_path}    {_format_size(progress.current_bytes)} / {_format_size(progress.current_size)}"
            )
        elif kind == "download_complete":
            self._on_download_complete(payload)
        elif kind == "cleanup_complete":
            result = payload
            self._finish("设备日志清理完成" if not result.failures else "清理部分完成")
            dir_note = (
                f"，目录 {result.deleted_dirs} 个" if result.deleted_dirs else ""
            )
            self._append_log(
                f"已删除 {result.deleted_files} 个文件（{_format_size(result.deleted_bytes)}）"
                f"{dir_note}，因变化跳过 {result.skipped_changed} 个。",
                "success",
            )
            for failure in result.failures:
                self._append_log(
                    f"清理失败：{failure.remote_path}：{failure.message}", "error"
                )
            if result.failures:
                messagebox.showwarning(
                    "清理部分失败",
                    f"{len(result.failures)} 个文件未能清理，请查看任务记录。",
                    parent=self.root,
                )
        elif kind == "passphrase_required":
            self._ask_for_passphrase(payload)
        elif kind == "passphrase_invalid":
            self._ask_for_passphrase(payload, invalid=True)
        elif kind == "cancelled":
            self._finish("任务已取消")
            self._append_log(payload, "muted")
        elif kind == "failed":
            was_connecting = self.operation == "connect"
            self._finish("任务失败")
            if was_connecting:
                self._set_connection_status("● 未连接")
            self._append_log(payload, "error")
            messagebox.showerror("任务失败", payload, parent=self.root)

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                self._handle_message(kind, payload)
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(100, self._poll_messages)

    def _on_close(self) -> None:
        if self.busy:
            messagebox.showinfo(
                "任务正在运行",
                "请先取消当前任务，并等待它结束后再关闭窗口。",
                parent=self.root,
            )
            return
        self.closing = True
        if self.connection_sftp is not None:
            close_sftp(self.connection_client, self.connection_sftp)
            self.connection_client = None
            self.connection_sftp = None
        self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    C3DownloaderApp(root)
    root.mainloop()
