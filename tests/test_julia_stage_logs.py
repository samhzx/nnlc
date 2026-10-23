import queue
import threading
import time
from pathlib import Path

import nnlc_auto_train


def test_parse_julia_stage_line_strips_marker_and_keeps_message():
    display, stage = nnlc_auto_train._parse_julia_log_line(
        "[NNLC_STAGE:loading_deps] 正在加载 Julia 依赖（已缓存则直接加载，首次才会编译）\n"
    )
    assert stage == nnlc_auto_train.JULIA_STAGE_LOADING_DEPS
    assert display == "正在加载 Julia 依赖（已缓存则直接加载，首次才会编译）"
    assert "[NNLC_STAGE:" not in display


def test_parse_julia_stage_fallbacks():
    cases = (
        ("Loading data...", nnlc_auto_train.JULIA_STAGE_LOADING_DATA),
        ("使用两遍扫描的流式低内存预处理", nnlc_auto_train.JULIA_STAGE_LOADING_DATA),
        ("Streaming pass 1: 10 valid rows across 1 routes", nnlc_auto_train.JULIA_STAGE_LOADING_DATA),
        ("Training progress: 0/1000", nnlc_auto_train.JULIA_STAGE_COMPILING_TRAIN),
        ("Training progress: 1/1000", nnlc_auto_train.JULIA_STAGE_TRAINING),
        ("CPU mode enabled", nnlc_auto_train.JULIA_STAGE_DEPS_READY),
        ("random unrelated line", None),
    )
    for line, expected in cases:
        _, stage = nnlc_auto_train._parse_julia_log_line(line)
        assert stage == expected, line


def test_julia_idle_messages_distinguish_stages():
    starting = nnlc_auto_train._julia_idle_message(nnlc_auto_train.JULIA_STAGE_STARTING)
    loading_deps = nnlc_auto_train._julia_idle_message(nnlc_auto_train.JULIA_STAGE_LOADING_DEPS)
    loading_data = nnlc_auto_train._julia_idle_message(nnlc_auto_train.JULIA_STAGE_LOADING_DATA)
    compiling = nnlc_auto_train._julia_idle_message(nnlc_auto_train.JULIA_STAGE_COMPILING_TRAIN)
    training = nnlc_auto_train._julia_idle_message(nnlc_auto_train.JULIA_STAGE_TRAINING)
    assert "启动" in starting
    assert "缓存" in loading_deps
    assert "读取训练数据" in loading_data
    assert "编译训练代码" in compiling
    assert "训练进行中" in training
    assert starting != loading_deps != loading_data != compiling != training
    assert "正在编译依赖" not in starting
    assert "正在编译依赖" not in loading_deps


def test_forward_process_output_heartbeat_follows_julia_stage(monkeypatch):
    messages = []
    emitted = []
    monkeypatch.setattr(nnlc_auto_train, "print_info", messages.append)
    monkeypatch.setattr(nnlc_auto_train, "_emit", emitted.append)
    monkeypatch.setattr(nnlc_auto_train, "_terminate_process_tree", lambda process: None)

    output_items = queue.Queue()

    class FakeStdout:
        def __iter__(self):
            while True:
                item = output_items.get()
                if item is None:
                    return
                yield item

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

    def feeder():
        time.sleep(0.05)
        output_items.put(
            "[NNLC_STAGE:loading_data] 正在读取训练数据\n"
        )
        time.sleep(0.8)
        process._alive = False
        output_items.put(None)

    threading.Thread(target=feeder, daemon=True).start()
    assert nnlc_auto_train._forward_process_output(
        process,
        idle_message=nnlc_auto_train._julia_idle_message(
            nnlc_auto_train.JULIA_STAGE_STARTING
        ),
        idle_timeout=0.2,
    ) == 0

    assert messages[0] == nnlc_auto_train._julia_idle_message(
        nnlc_auto_train.JULIA_STAGE_STARTING
    )
    assert emitted == ["    正在读取训练数据"]
    loading_data_idle = nnlc_auto_train._julia_idle_message(
        nnlc_auto_train.JULIA_STAGE_LOADING_DATA
    )
    assert any(message.startswith(loading_data_idle) for message in messages)
    assert not any("正在编译依赖" in message for message in messages)


def test_step_train_no_longer_claims_every_run_is_first_compile():
    source = Path(nnlc_auto_train.__file__).read_text(encoding="utf-8")
    assert "Julia 首次运行需编译依赖" not in source
    assert "Julia 正在编译依赖，请继续等待" not in source
    assert "首次在该 Julia 环境训练时会编译依赖并缓存" in source
    assert "idle_message=_julia_idle_message(JULIA_STAGE_STARTING)" in source
