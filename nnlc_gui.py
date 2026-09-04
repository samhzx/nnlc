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

from nnlc_auto_train import auto_train


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
CORRUPT_LOG_MODES = {
    "严格模式（遇到损坏日志停止）": False,
    "容错模式（跳过损坏日志）": True,
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


class NNLCApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NNLC 横向控制模型训练")
        self.root.geometry("900x720")
        self.root.minsize(760, 600)
        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.started_at: float | None = None
        self.config_widgets = []
        self._ansi_log_tag: str | None = None
        self.cancel_event = threading.Event()
        self.process_holder = {}

        saved_preferences = self._load_preferences()
        saved_mode = saved_preferences.get("training_mode")
        saved_mode = LEGACY_TRAINING_MODES.get(saved_mode, saved_mode)
        if not isinstance(saved_mode, str) or saved_mode not in TRAINING_MODES:
            saved_mode = "CPU 标准模式"

        saved_corrupt_mode = saved_preferences.get("corrupt_log_mode")
        if not isinstance(saved_corrupt_mode, str) or saved_corrupt_mode not in CORRUPT_LOG_MODES:
            saved_corrupt_mode = "严格模式（遇到损坏日志停止）"

        saved_car = saved_preferences.get("car")
        if not isinstance(saved_car, str) or not saved_car.strip():
            saved_car = "BYD_TANG_DMI_24"

        self.data_var = tk.StringVar()
        self.output_var = tk.StringVar(value=self._default_output())
        self.car_var = tk.StringVar(value=saved_car)
        self.threshold_var = tk.StringVar()
        self.training_mode_var = tk.StringVar(value=saved_mode)
        self.corrupt_log_mode_var = tk.StringVar(value=saved_corrupt_mode)
        self.auto_threshold_var = tk.BooleanVar(value=True)
        self.skip_viz_var = tk.BooleanVar(value=True)
        self.keep_intermediates_var = tk.BooleanVar(
            value=bool(saved_preferences.get("keep_intermediates", True))
        )
        self.status_var = tk.StringVar(value="就绪")
        self.elapsed_var = tk.StringVar(value="")
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_messages()

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
        corrupt_log_mode = self.corrupt_log_mode_var.get()
        if corrupt_log_mode not in CORRUPT_LOG_MODES:
            corrupt_log_mode = "严格模式（遇到损坏日志停止）"
        path = self._preferences_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_suffix(".tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "car": car,
                        "training_mode": training_mode,
                        "corrupt_log_mode": corrupt_log_mode,
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
        root = self.root
        style = ttk.Style(root)
        style.configure("TButton", padding=(10, 6))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(18, 8))
        style.configure("Section.TLabelframe", padding=10)
        style.configure("Section.TLabelframe.Label", font=("Microsoft YaHei UI", 10, "bold"))

        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        container = ttk.Frame(root, padding=(20, 16, 20, 14))
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(6, weight=1)

        ttk.Label(container, text="NNLC 横向控制模型训练", font=("Microsoft YaHei UI", 17, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 12)
        )

        path_frame = ttk.LabelFrame(container, text="目录设置", style="Section.TLabelframe")
        path_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        path_frame.columnconfigure(1, weight=1)
        data_widgets = self._path_row(path_frame, 0, "reallog 目录", self.data_var, self._choose_data)
        output_widgets = self._path_row(path_frame, 1, "输出目录", self.output_var, self._choose_output)
        self.config_widgets.extend((*data_widgets, *output_widgets))

        options_frame = ttk.LabelFrame(container, text="训练参数", style="Section.TLabelframe")
        options_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        options_frame.columnconfigure(1, weight=1)

        ttk.Label(options_frame, text="车型").grid(row=0, column=0, sticky="w", pady=6)
        self.car_entry = ttk.Entry(options_frame, textvariable=self.car_var)
        self.car_entry.grid(row=0, column=1, sticky="ew", padx=(12, 10), pady=6)
        ttk.Label(options_frame, text="示例：BYD_TANG_DMI_24").grid(row=0, column=2, sticky="w", pady=6)
        self.config_widgets.append(self.car_entry)

        ttk.Label(options_frame, text="路线阈值").grid(row=1, column=0, sticky="w", pady=6)
        threshold_frame = ttk.Frame(options_frame)
        threshold_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=(12, 0), pady=6)
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=12, state="disabled")
        self.threshold_entry.pack(side="left")
        self.auto_threshold_check = ttk.Checkbutton(
            threshold_frame,
            text="自动推荐（默认）",
            variable=self.auto_threshold_var,
            command=self._toggle_threshold,
        )
        self.auto_threshold_check.pack(side="left", padx=(12, 0))
        self.visualize_check = ttk.Checkbutton(
            threshold_frame,
            text="生成覆盖度图（默认）",
            variable=self.skip_viz_var,
        )
        self.visualize_check.pack(side="left", padx=(28, 0))
        self.config_widgets.extend((self.auto_threshold_check, self.visualize_check))

        ttk.Label(options_frame, text="训练模式").grid(row=2, column=0, sticky="w", pady=6)
        self.training_mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.training_mode_var,
            values=tuple(TRAINING_MODES),
            state="readonly",
            width=16,
        )
        self.training_mode_combo.grid(row=2, column=1, sticky="w", padx=(12, 10), pady=6)
        self.training_mode_combo.bind("<<ComboboxSelected>>", self._on_training_mode_changed)
        ttk.Label(options_frame, text="流式模式适合超大数据和 16GB 内存电脑").grid(
            row=2, column=2, sticky="w", pady=6
        )
        self.config_widgets.append(self.training_mode_combo)

        ttk.Label(options_frame, text="日志处理模式").grid(row=3, column=0, sticky="w", pady=6)
        self.corrupt_log_mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.corrupt_log_mode_var,
            values=tuple(CORRUPT_LOG_MODES),
            state="readonly",
            width=30,
        )
        self.corrupt_log_mode_combo.grid(row=3, column=1, sticky="w", padx=(12, 10), pady=6)
        self.corrupt_log_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._save_preferences())
        ttk.Label(options_frame, text="容错模式会跳过损坏 rlog 并记录文件").grid(row=3, column=2, sticky="w", pady=6)
        self.config_widgets.append(self.corrupt_log_mode_combo)

        self.keep_intermediates_check = ttk.Checkbutton(
            options_frame,
            text="保留完整中间 CSV（默认）",
            variable=self.keep_intermediates_var,
            command=self._save_preferences,
        )
        self.keep_intermediates_check.grid(row=4, column=1,
                                           sticky="w", padx=(12, 0), pady=6)
        ttk.Label(options_frame, text="关闭后仅在流式训练成功时清理中间文件").grid(
            row=4, column=2, sticky="e", pady=6
        )
        self.config_widgets.append(self.keep_intermediates_check)
        self._update_streaming_options()

        controls = ttk.Frame(container)
        controls.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="开始训练", style="Primary.TButton", command=self.start)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(controls, mode="indeterminate", length=220)
        self.progress.grid(row=0, column=1, sticky="ew", padx=14)
        ttk.Label(controls, textvariable=self.elapsed_var, width=13, anchor="e").grid(row=0, column=2)
        self.open_output_button = ttk.Button(controls, text="打开输出目录", command=self._open_output)
        self.open_output_button.grid(row=0, column=3, padx=(12, 0))
        self.clear_log_button = ttk.Button(controls, text="清空日志", command=self.clear_log)
        self.clear_log_button.grid(row=0, column=4, padx=(8, 0))

        log_frame = ttk.LabelFrame(container, text="运行日志", style="Section.TLabelframe")
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            height=16,
            font=("Consolas", 9),
            borderwidth=0,
            padx=8,
            pady=8,
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.tag_configure("info", foreground="#2563a6")
        self.log.tag_configure("success", foreground="#16803a")
        self.log.tag_configure("warning", foreground="#a15c00")
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("heading", foreground="#7a3e9d")
        self.log.tag_configure("step", foreground="#007c91")

        ttk.Label(container, textvariable=self.status_var, relief="sunken", anchor="w", padding=(8, 5)).grid(
            row=7, column=0, sticky="ew", pady=(10, 0)
        )

    @staticmethod
    def _path_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=(12, 10), pady=6)
        button = ttk.Button(parent, text="浏览...", command=browse_command)
        button.grid(row=row, column=2, sticky="e", pady=6)
        return entry, button

    def _toggle_threshold(self) -> None:
        self.threshold_entry.configure(state="disabled" if self.auto_threshold_var.get() else "normal")

    def _update_streaming_options(self) -> None:
        is_streaming = self.training_mode_var.get() == "CPU 流式低内存模式"
        self.keep_intermediates_check.configure(state="normal" if is_streaming else "disabled")

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
        state = "disabled" if running else "normal"
        for widget in self.config_widgets:
            widget.configure(state=state)
        self.threshold_entry.configure(
            state="disabled" if running or self.auto_threshold_var.get() else "normal"
        )
        self.start_button.configure(state=state)
        self.open_output_button.configure(state=state)
        self.clear_log_button.configure(state=state)
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
            # Comboboxes are intentionally readonly; restoring every widget
            # to ``normal`` would let an invalid training mode be typed in.
            self.training_mode_combo.configure(state="readonly")
            self.corrupt_log_mode_combo.configure(state="readonly")
            self._update_streaming_options()

    def _update_elapsed(self) -> None:
        if not self.worker or not self.worker.is_alive() or self.started_at is None:
            return
        seconds = int(time.monotonic() - self.started_at)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            self.elapsed_var.set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            self.elapsed_var.set(f"{minutes:02d}:{seconds:02d}")
        self.root.after(1000, self._update_elapsed)

    def _finish_running(self, status: str) -> None:
        self._set_running(False)
        self.status_var.set(status)

    def _on_close(self) -> None:
        self._save_preferences()
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
                try:
                    process.terminate()
                except OSError:
                    pass
        self.root.destroy()

    def start(self) -> None:
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
        self.elapsed_var.set("00:00")
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
        corrupt_log_mode = self.corrupt_log_mode_var.get().strip()
        if corrupt_log_mode not in CORRUPT_LOG_MODES:
            corrupt_log_mode = "严格模式（遇到损坏日志停止）"
            self.corrupt_log_mode_var.set(corrupt_log_mode)
        skip_corrupt_rlogs = CORRUPT_LOG_MODES[corrupt_log_mode]
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(data_dir, output_dir, car, min_score, skip_visualize,
                  skip_corrupt_rlogs, batch_size,
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
        skip_corrupt_rlogs,
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
                    skip_corrupt_rlogs=skip_corrupt_rlogs,
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
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)


def launch_gui() -> None:
    if tk is None:
        raise RuntimeError("当前 Python 未包含 Tk；Windows 官方 Python 安装包自带 Tk。")
    root = tk.Tk()
    NNLCApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
