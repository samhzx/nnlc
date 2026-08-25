"""Small Tkinter front-end for the NNLC one-click training pipeline."""

from __future__ import annotations

import contextlib
import io
import os
import queue
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # Some minimal Python builds (including CI macOS) omit Tk.
    tk = None
    filedialog = messagebox = ttk = None

from nnlc_auto_train import auto_train


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
        self.root.geometry("820x650")
        self.root.minsize(700, 520)
        self.messages: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        self.data_var = tk.StringVar()
        self.output_var = tk.StringVar(value=self._default_output())
        self.car_var = tk.StringVar(value="BYD_TANG_DMI_24")
        self.threshold_var = tk.StringVar()
        self.auto_threshold_var = tk.BooleanVar(value=True)
        # This control is phrased positively in the UI; checked means keep the
        # existing pipeline's default of generating coverage plots.
        self.skip_viz_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self._build_widgets()
        self._poll_messages()

    @staticmethod
    def _default_output() -> str:
        documents = Path.home() / "Documents"
        return str(documents / "NNLC_Output") if documents.exists() else str(Path.cwd() / "NNLC_Output")

    def _build_widgets(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="NNLC 横向控制模型训练", font=("Microsoft YaHei UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )

        self._path_row(header, 1, "reallog 目录", self.data_var, self._choose_data)
        self._path_row(header, 2, "输出目录", self.output_var, self._choose_output)

        ttk.Label(header, text="车型").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(header, textvariable=self.car_var).grid(row=3, column=1, sticky="ew", padx=8, pady=5)
        ttk.Label(header, text="如 BYD_TANG_DMI_24").grid(row=3, column=2, sticky="w", pady=5)

        ttk.Label(header, text="路线阈值").grid(row=4, column=0, sticky="w", pady=5)
        threshold_frame = ttk.Frame(header)
        threshold_frame.grid(row=4, column=1, sticky="w", padx=8, pady=5)
        self.threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=12, state="disabled")
        self.threshold_entry.pack(side="left")
        ttk.Checkbutton(
            threshold_frame,
            text="自动推荐（默认）",
            variable=self.auto_threshold_var,
            command=self._toggle_threshold,
        ).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(
            threshold_frame,
            text="生成覆盖度图（默认）",
            variable=self.skip_viz_var,
            command=self._toggle_viz,
        ).pack(side="left", padx=(24, 0))

        buttons = ttk.Frame(header)
        buttons.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        buttons.columnconfigure(0, weight=1)
        self.start_button = ttk.Button(buttons, text="开始训练", command=self.start)
        self.start_button.grid(row=0, column=1, padx=(8, 0))
        ttk.Button(buttons, text="清空日志", command=self.clear_log).grid(row=0, column=2, padx=(8, 0))

        log_frame = ttk.LabelFrame(root, text="运行日志", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew", padx=18, pady=(4, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w").grid(
            row=2, column=0, sticky="ew", padx=18, pady=(0, 10)
        )

    @staticmethod
    def _path_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(parent, text="浏览...", command=browse_command).grid(row=row, column=2, sticky="e", pady=5)

    def _toggle_threshold(self) -> None:
        self.threshold_entry.configure(state="disabled" if self.auto_threshold_var.get() else "normal")

    def _toggle_viz(self) -> None:
        # The checkbox text is intentionally positive; the internal option is
        # named skip_visualize to match the existing pipeline API.
        pass

    def _choose_data(self) -> None:
        path = filedialog.askdirectory(title="选择包含 reallog/rlog 的目录")
        if path:
            self.data_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        data_dir = self.data_var.get().strip()
        output_dir = self.output_var.get().strip()
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

        min_score = None
        if not self.auto_threshold_var.get():
            try:
                min_score = int(self.threshold_var.get().strip())
            except ValueError:
                messagebox.showerror("参数错误", "路线阈值必须是整数，或勾选自动推荐。")
                return

        self.clear_log()
        self.start_button.configure(state="disabled")
        self.status_var.set("训练中，请保持窗口打开...")
        skip_visualize = not self.skip_viz_var.get()
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(data_dir, output_dir, car, min_score, skip_visualize),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, data_dir, output_dir, car, min_score, skip_visualize) -> None:
        writer = _QueueWriter(self.messages)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                model_path = auto_train(
                    data_dir,
                    car,
                    min_score=min_score,
                    skip_visualize=skip_visualize,
                    output_dir=output_dir,
                    deploy_dir=output_dir,
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
                    self.start_button.configure(state="normal")
                    self.status_var.set("训练完成")
                    messagebox.showinfo("训练完成", f"模型已输出到：\n{payload}")
                elif kind == "error":
                    self.start_button.configure(state="normal")
                    self.status_var.set("训练失败")
                    self._append_log(f"\n[ERROR] {payload}\n")
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
