"""Tkinter UI for Windows EXE update checks and downloads."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # Some minimal Python builds (including CI macOS) omit Tk.
    tk = None
    messagebox = ttk = None

from nnlc_runtime import application_directory
from nnlc_update import UpdateError, check_for_update, download_update, format_bytes


C_HEADER_BG = "#16324f"
C_PAGE_BG = "#f2f4f8"
C_CARD_BG = "#ffffff"
C_ACCENT = "#2f7de1"
C_ACCENT_HOVER = "#2060b8"
C_TEXT = "#2b333d"
C_BTN_BORDER = "#d6dce5"
C_BTN_HOVER = "#f0f3f7"
F_SUB = ("Microsoft YaHei UI", 9)
F_BASE = ("Microsoft YaHei UI", 10)
F_BOLD = ("Microsoft YaHei UI", 10, "bold")


class GuiUpdateController:
    """Owns update dialogs and download state for the training GUI host.

    Host attributes used: root, app_version, messages, status_var, worker,
    _closing, _set_running, _append_event, _open_path.
    """

    def __init__(self, host) -> None:
        self.host = host
        self.pending_update = None
        self.update_prompt_shown = False
        self.update_in_progress = False
        self.update_cancel_event = threading.Event()
        self.update_thread = None
        self.download_window = None
        self.download_progress = None
        self.download_status_var = None
        self._styles_installed = False

    @property
    def in_progress(self) -> bool:
        return self.update_in_progress

    def start_background_check(self) -> None:
        if not self._should_check_for_updates():
            return
        thread = threading.Thread(target=self._check_for_update_worker, daemon=True)
        thread.start()

    def handle_message(self, kind, payload) -> bool:
        if kind == "update_available":
            self._handle_update_available(payload)
        elif kind == "update_download_progress":
            self._update_download_progress(*payload)
        elif kind == "update_downloaded":
            self._finish_download_success(payload)
        elif kind == "update_failed":
            self._finish_download_failure(payload)
        else:
            return False
        return True

    def on_training_finished(self) -> None:
        if self.pending_update is not None and not self.update_prompt_shown:
            self.host.root.after(200, self._prompt_pending_update)

    def confirm_close_if_downloading(self) -> bool:
        if not self.update_in_progress:
            return True
        return bool(
            messagebox.askyesno(
                "正在下载更新",
                "关闭窗口会取消当前更新下载，确定要退出吗？",
                icon="warning",
            )
        )

    def cancel_and_join_on_close(self) -> None:
        if not self.update_in_progress:
            return
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

    def _should_check_for_updates(self) -> bool:
        return bool(
            getattr(sys, "frozen", False)
            and sys.platform == "win32"
            and self.host.app_version
            and self.host.app_version != "unknown"
        )

    def _check_for_update_worker(self) -> None:
        try:
            manifest = check_for_update(self.host.app_version)
        except Exception:
            return
        if manifest is not None:
            self.host.messages.put(("update_available", manifest))

    def _handle_update_available(self, manifest) -> None:
        self.pending_update = manifest
        if self.host.worker and self.host.worker.is_alive():
            self.host._append_event(f"发现新版本 v{manifest.version}，将在训练结束后提示。", "info")
            return
        self._prompt_pending_update()

    def _ensure_styles(self) -> None:
        if self._styles_installed or tk is None or ttk is None:
            return
        style = ttk.Style(self.host.root)
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
        self._styles_installed = True

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

        def on_enter(_event):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=C_ACCENT_HOVER)

        def on_leave(_event):
            if str(btn["state"]) != "disabled":
                btn.configure(bg=C_ACCENT)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return box

    def _prompt_pending_update(self) -> None:
        manifest = self.pending_update
        if manifest is None or self.update_prompt_shown or self.update_in_progress:
            return
        if self.host.worker and self.host.worker.is_alive():
            return
        self.update_prompt_shown = True
        self._ensure_styles()
        notes = manifest.notes.strip() or "无更新说明。"
        dialog = tk.Toplevel(self.host.root)
        dialog.title("发现新版本")
        dialog.resizable(False, False)
        dialog.transient(self.host.root)
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
            f"当前版本：v{self.host.app_version}\n"
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
        self.host.root.wait_window(dialog)
        if choice["download"]:
            self._start_update_download(manifest)
        else:
            self.host._append_event(f"已跳过 v{manifest.version} 更新下载。", "info")

    def _start_update_download(self, manifest) -> None:
        if self.host.worker and self.host.worker.is_alive():
            messagebox.showinfo("训练进行中", "请等待训练结束后再下载更新。")
            self.update_prompt_shown = False
            return
        self.update_in_progress = True
        self.update_cancel_event = threading.Event()
        self.host._set_running(False)
        self.host.status_var.set("正在下载更新...")
        self._show_download_window(manifest)
        self.update_thread = threading.Thread(
            target=self._download_update_worker,
            args=(manifest,),
            daemon=True,
        )
        self.update_thread.start()

    def _show_download_window(self, manifest) -> None:
        self._ensure_styles()
        window = tk.Toplevel(self.host.root)
        window.title("下载更新")
        window.resizable(False, False)
        window.transient(self.host.root)
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
        progress = ttk.Progressbar(
            card,
            length=420,
            maximum=100,
            mode="determinate",
            style="Modern.Horizontal.TProgressbar",
        )
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
                progress_callback=lambda downloaded, total, percent: self.host.messages.put(
                    ("update_download_progress", (downloaded, total, percent))
                ),
                cancel_event=self.update_cancel_event,
            )
            self.host.messages.put(("update_downloaded", path))
        except Exception as exc:
            self.host.messages.put(("update_failed", exc))

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
        if self.host._closing:
            return
        self.host._set_running(False)
        self.host.status_var.set("更新已下载")
        self.host._append_event(f"新版本已准备好：{path}", "success")
        should_open = messagebox.askyesno(
            "更新已下载",
            "新程序已准备好：\n"
            f"{path}\n\n"
            "关闭当前程序后运行这个新 EXE 即可，原来的 Julia 环境可以继续使用。\n\n"
            "是否打开所在目录？",
        )
        if should_open:
            self.host._open_path(Path(path).parent)

    def _finish_download_failure(self, exc) -> None:
        self.update_in_progress = False
        self.update_thread = None
        self._close_download_window()
        if self.host._closing:
            return
        self.host._set_running(False)
        self.host.status_var.set("更新下载失败")
        message = str(exc)
        if isinstance(exc, UpdateError) and "取消" in message:
            self.host._append_event("已取消更新下载。", "warning")
            messagebox.showinfo("已取消下载", "未完成的下载文件已删除。")
            return
        self.host._append_event(f"更新下载失败：{message}", "error")
        messagebox.showerror("更新下载失败", f"{message}\n未完成的下载文件已删除。")
