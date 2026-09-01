#!/usr/bin/env python3
"""NNLC 横向控制模型一键训练脚本。

从 rlog 数据到模型输出的全自动化训练流程。
支持交互模式（双击运行）和命令行模式。

使用方式:
  # 交互模式（双击运行或无参数执行）
  python nnlc_auto_train.py

  # 命令行模式
  python nnlc_auto_train.py --data ./data --car BYD_TANG_DMI_24
  python nnlc_auto_train.py --data ./data --car TOYOTA_CAMRY --min-score 40
  python nnlc_auto_train.py --data ./data --car HONDA_CIVIC --skip-deploy

流程:
  1. 提取横向控制数据（含时序特征）
  2. 评估路线质量（自动推荐 min-score）
  3. 修剪低质量路线
  4. 移除干预帧
  5. 可视化数据覆盖度
  6. Julia 训练模型
  7. 验证模型质量
  8. 部署到 openpilot 项目
"""

import argparse
import contextlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import time


# ============================================================
# 配置
# ============================================================

# 训练库根目录（脚本自动定位，支持目录迁移）
NNLC_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))

# PyInstaller extracts a one-file application to ``sys._MEIPASS``.  Keep all
# resource lookups in one place so the same entry point works from source and
# from the bundled Windows executable.
RESOURCE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(NNLC_TOOLS_DIR))


def _resource_path(*parts):
    return os.path.join(RESOURCE_DIR, *parts)

# Julia 可执行文件路径（相对于 NNLC_TOOLS_DIR 的上级目录）
def get_julia_exe():
    """Return the bundled Julia executable, or the source-tree copy."""
    if getattr(sys, "frozen", False):
        return _resource_path("julia-runtime", "bin", "julia.exe")
    return os.path.join(
        os.path.dirname(NNLC_TOOLS_DIR),
        "julia-1.10.11", "julia-1.10.11", "bin", "julia.exe"
    )


def get_julia_depot():
    """Return the bundled Julia depot when running from the packaged app."""
    if getattr(sys, "frozen", False):
        return _resource_path("julia-depot")
    return os.environ.get("JULIA_DEPOT_PATH", "")

# 训练脚本路径（相对于 NNLC_TOOLS_DIR）
TRAINING_SCRIPT = "training/latmodel_temporal.jl"

# 默认模型部署目录（nnlc 同级目录下的 models 文件夹）
DEFAULT_DEPLOY_DIR = os.path.join(os.path.dirname(NNLC_TOOLS_DIR), "models")

# Set by the GUI while it redirects the worker's stdout into its log window.
_LOG_CALLBACK = None


def set_log_callback(callback):
    """Send console lines to a GUI (or disable forwarding with ``None``)."""
    global _LOG_CALLBACK
    _LOG_CALLBACK = callback


def _emit(message):
    if sys.stdout is not None:
        print(message)
    if _LOG_CALLBACK is not None:
        try:
            _LOG_CALLBACK(str(message))
        except Exception:
            pass

# 模型验收标准
MAX_TEST_LOSS = 0.05
REQUIRED_INPUT_SIZE = 18

# 评分等级
SCORE_LEVELS = {
    "excellent": 90,
    "good": 70,
    "acceptable": 60,
    "poor": 40,
}


# ============================================================
# 工具函数
# ============================================================

class Colors:
    """终端颜色（Windows 兼容）。"""
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _enable_ansi():
    """启用 Windows 终端 ANSI 颜色支持。"""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)


_enable_ansi()


def print_step(step_num, total, message):
    """打印步骤标题。"""
    _emit(f"\n{Colors.BOLD}{Colors.CYAN}[{step_num}/{total}] {message}{Colors.RESET}")
    _emit("=" * 60)


def print_ok(message):
    """打印成功信息。"""
    _emit(f"  {Colors.GREEN}[OK]{Colors.RESET} {message}")


def print_warn(message):
    """打印警告信息。"""
    _emit(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {message}")


def print_error(message):
    """打印错误信息。"""
    _emit(f"  {Colors.RED}[ERROR]{Colors.RESET} {message}")


def print_info(message):
    """打印普通信息。"""
    _emit(f"  {Colors.BLUE}[INFO]{Colors.RESET} {message}")


def run_command(cmd, description, check=True):
    """执行命令并实时显示输出。

    Args:
        cmd: 命令列表或字符串
        description: 命令描述
        check: 是否在非零退出码时抛出异常

    Returns:
        subprocess.CompletedProcess 对象
    """
    print_info(f"{description}...")
    _emit(f"    命令: {cmd if isinstance(cmd, str) else ' '.join(str(c) for c in cmd)}")

    # A frozen exe cannot launch ``python -m ...`` because the target machine
    # has no Python interpreter.  Execute our own bundled modules in-process
    # instead.  Keeping the source-mode subprocess path preserves the normal
    # CLI behaviour and makes debugging with a virtualenv straightforward.
    if (
        getattr(sys, "frozen", False)
        and isinstance(cmd, (list, tuple))
        and len(cmd) >= 3
        and os.path.abspath(str(cmd[0])) == os.path.abspath(sys.executable)
        and cmd[1] == "-m"
    ):
        old_argv = sys.argv
        stdout = io.StringIO()
        stderr = io.StringIO()
        returncode = 0
        try:
            sys.argv = [str(cmd[1])] + [str(arg) for arg in cmd[3:]]
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                runpy.run_module(str(cmd[2]), run_name="__main__")
        except SystemExit as exc:
            returncode = int(exc.code) if isinstance(exc.code, int) else 1
        except Exception:
            returncode = 1
            import traceback
            traceback.print_exc(file=stderr)
        finally:
            sys.argv = old_argv
        result = subprocess.CompletedProcess(cmd, returncode, stdout.getvalue(), stderr.getvalue())
        for line in result.stdout.strip().splitlines():
            _emit(f"    {line}")
        for line in result.stderr.strip().splitlines():
            _emit(f"    {line}")
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, cmd, result.stdout, result.stderr)
        if returncode:
            print_warn(f"命令退出码: {returncode}")
        else:
            print_ok("完成")
        return result

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print_error(f"命令失败 (耗时 {elapsed:.1f}s, 退出码 {e.returncode})")
        if e.stdout:
            for line in e.stdout.strip().split("\n")[-5:]:
                _emit(f"    {line}")
        if e.stderr:
            for line in e.stderr.strip().split("\n")[-5:]:
                _emit(f"    {line}")
        raise
    except FileNotFoundError:
        print_error(f"命令未找到: {cmd[0] if isinstance(cmd, list) else cmd.split()[0]}")
        raise

    elapsed = time.time() - start_time
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            _emit(f"    {line}")
    print_ok(f"完成 (耗时 {elapsed:.1f}s)")
    return result


def _raise_if_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("训练已取消")


def get_python_exe():
    """获取 Python 可执行文件路径。"""
    venv_python = os.path.join(NNLC_TOOLS_DIR, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


def validate_julia():
    """验证 Julia 安装。"""
    julia_exe = get_julia_exe()
    if not os.path.isfile(julia_exe):
        print_error(f"Julia 未找到: {julia_exe}")
        if os.path.isdir(julia_exe):
            print_error("打包目录结构错误：julia.exe 被创建成了文件夹")
        else:
            print_info("请先安装 Julia 1.10.11，参考 nnlc_training_tutorial.md 第 2.1 节")
        return False
    try:
        result = subprocess.run(
            [julia_exe, "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        print_error(f"Julia 无法启动: {julia_exe}")
        print_error(f"系统错误: {exc}")
        return False
    if result.returncode != 0:
        print_error(f"Julia 执行失败: {result.stderr.strip() or result.returncode}")
        return False
    print_ok(f"Julia: {result.stdout.strip()}")
    return True


def validate_data_dir(data_dir):
    """验证数据目录。"""
    if not os.path.isdir(data_dir):
        print_error(f"数据目录不存在: {data_dir}")
        return False

    # 检查是否有 rlog 文件
    rlog_count = 0
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.startswith("rlog"):
                rlog_count += 1

    if rlog_count == 0:
        print_error(f"数据目录中未找到 rlog 文件: {data_dir}")
        print_info("请确认目录路径正确，且包含 rlog.zst / rlog.bz2 文件")
        return False

    print_ok(f"数据目录: {data_dir} (找到 {rlog_count} 个 rlog 文件)")
    return True


# ============================================================
# 训练流程
# ============================================================

def step_extract(data_dir, output_dir, python_exe):
    """步骤1: 提取横向控制数据。"""
    output_csv = os.path.join(output_dir, "lateral_data.csv")
    run_command(
        [python_exe, "-m", "nnlc_tools.extract_lateral_data", data_dir,
         "-o", output_csv, "--temporal"],
        "提取横向控制数据（含时序特征）",
    )

    # 验证输出
    if not os.path.exists(output_csv):
        raise RuntimeError(f"提取数据失败，输出文件未生成: {output_csv}")

    # 统计行数
    with open(output_csv, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1  # 减去表头
    print_ok(f"提取 {row_count:,} 行数据")

    if row_count < 10000:
        print_warn(f"数据量偏少 ({row_count:,} 行)，建议补充采集")
    else:
        print_ok(f"数据量合格 ({row_count:,} 行)")

    return output_csv


def step_score(input_csv, python_exe):
    """步骤2: 评估路线质量，返回自动推荐的 min_score。"""
    result = run_command(
        [python_exe, "-m", "nnlc_tools.score_routes", input_csv],
        "评估路线质量",
    )

    # 解析评分结果，自动推荐 min_score
    recommended_score = 60  # 默认值

    # 从输出中提取评分分布
    output = result.stdout
    good_count = 0
    acceptable_count = 0
    poor_count = 0
    total_routes = 0

    for line in output.split("\n"):
        if "routes scored" in line:
            total_routes = int(line.split()[0])
        if "routes with score >= 70" in line:
            good_count = int(line.strip().split()[0])

    if total_routes > 0:
        good_ratio = good_count / total_routes
        if good_ratio >= 0.3:
            # 30% 以上高质量，使用严格阈值
            recommended_score = 70
        elif good_ratio >= 0.1:
            # 10%-30% 高质量，使用中等阈值
            recommended_score = 60
        else:
            # 高质量路线不足 10%，放宽阈值
            recommended_score = 40
            print_warn("高质量路线不足 10%，建议补充采集更多激活行驶数据")

    print_info(f"自动推荐 min-score = {recommended_score}")
    return recommended_score


def step_prune(input_csv, min_score, output_dir, python_exe):
    """步骤3: 修剪低质量路线。"""
    output_csv = os.path.join(output_dir, "routes_pruned.csv")
    run_command(
        [python_exe, "-m", "nnlc_tools.prune_routes", input_csv,
         "--min-score", str(min_score), "-o", output_csv],
        f"修剪路线 (min-score={min_score})",
    )

    if not os.path.exists(output_csv):
        raise RuntimeError(f"修剪路线失败，输出文件未生成: {output_csv}")

    with open(output_csv, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1
    print_ok(f"修剪后 {row_count:,} 行")

    if row_count < 5000:
        print_warn(f"修剪后数据量严重不足 ({row_count:,} 行)，考虑降低 min-score 或补充数据")
    elif row_count < 20000:
        print_warn(f"修剪后数据量偏少 ({row_count:,} 行)，模型可能过拟合")

    return output_csv


def step_interventions(input_csv, output_dir, python_exe):
    """步骤4: 移除干预帧。"""
    output_csv = os.path.join(output_dir, "interventions_pruned.csv")
    run_command(
        [python_exe, "-m", "nnlc_tools.analyze_interventions", input_csv,
         "--prune-output", output_csv],
        "分类并移除干预帧",
    )

    if not os.path.exists(output_csv):
        raise RuntimeError(f"移除干预帧失败，输出文件未生成: {output_csv}")

    with open(output_csv, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1
    print_ok(f"最终训练数据 {row_count:,} 行")

    return output_csv


def step_visualize(input_csv, output_dir, python_exe):
    """步骤5: 可视化数据覆盖度。"""
    output_png = os.path.join(output_dir, "coverage.png")
    run_command(
        [python_exe, "-m", "nnlc_tools.visualize_coverage", input_csv,
         "-o", output_png],
        "生成数据覆盖度图",
    )

    if os.path.exists(output_png):
        print_ok(f"覆盖度图: {output_png}")

    return output_png


def step_train(input_csv, car_name, output_dir, cancel_event=None,
               process_holder=None, batch_size=16384):
    """步骤6: Julia 训练模型。"""
    _raise_if_cancelled(cancel_event)
    # 创建训练输入目录（只包含要训练的 CSV）
    train_input_dir = os.path.join(output_dir, "training_input")
    os.makedirs(train_input_dir, exist_ok=True)

    # 复制 CSV 为车型命名
    car_csv = os.path.join(train_input_dir, f"{car_name}.csv")
    shutil.copy2(input_csv, car_csv)

    # 清理旧的训练结果（如果存在）
    train_results_dir = os.path.join(train_input_dir, "training_results", car_name)
    if os.path.exists(train_results_dir):
        print_info(f"清理旧训练结果: {train_results_dir}")
        shutil.rmtree(train_results_dir)

    # 执行 Julia 训练并逐行转发日志；窗口版 exe 没有控制台，不能依赖
    # 子进程继承 stdout。
    script_root = RESOURCE_DIR if getattr(sys, "frozen", False) else NNLC_TOOLS_DIR
    script_path = os.path.join(script_root, TRAINING_SCRIPT)
    julia_exe = get_julia_exe()
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    cmd = [julia_exe, script_path, train_input_dir, "--cpu", f"--batch-size={batch_size}"]
    print_info(f"Julia 训练中（CPU 模式，batch size={batch_size:,}）")
    print_info(f"命令: {' '.join(cmd)}")
    print_warn("Julia 首次运行需编译依赖（约 3-10 分钟无输出属正常），请耐心等待")
    print_info("训练日志会实时显示在下方:\n" + "-" * 60)

    start_time = time.time()
    env = os.environ.copy()
    depot = get_julia_depot()
    if depot:
        env["JULIA_DEPOT_PATH"] = depot
    # Keep Julia's GR/Plots backend headless.  The trainer writes plot files,
    # so it must not try to launch gksqt or require a desktop Qt plugin.
    env["GKSwstype"] = "100"
    env["QT_QPA_PLATFORM"] = "offscreen"
    # Julia resolves relative package/artifact paths from its working tree;
    # use the extracted bundle root for the frozen app.
    process = subprocess.Popen(
        cmd,
        cwd=script_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process_holder is not None:
        process_holder["process"] = process
    if cancel_event is not None and cancel_event.is_set():
        process.terminate()
    try:
        if process.stdout is not None:
            for line in process.stdout:
                _emit("    " + line.rstrip())
        result = subprocess.CompletedProcess(cmd, process.wait())
    finally:
        if process_holder is not None:
            process_holder["process"] = None

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("训练已取消")

    elapsed = time.time() - start_time
    print("-" * 60)
    print_ok(f"Julia 训练结束 (耗时 {elapsed:.1f}s, 退出码 {result.returncode})")

    if result.returncode != 0:
        raise RuntimeError(f"Julia 训练失败，退出码 {result.returncode}")

    # 检查模型文件
    model_json = os.path.join(train_results_dir, f"{car_name}.json")
    if not os.path.exists(model_json):
        # 列出训练结果目录内容，帮助排查
        if os.path.exists(train_results_dir):
            files = os.listdir(train_results_dir)
            print_error(f"训练结果目录内容: {files}")
        raise RuntimeError(f"训练完成但模型文件未生成: {model_json}")

    print_ok(f"模型文件: {model_json} ({os.path.getsize(model_json):,} bytes)")
    return model_json, train_results_dir


def step_validate(model_json, train_data_csv, output_dir, python_exe):
    """步骤7: 验证模型质量。"""
    # 1. 检查 JSON 结构
    with open(model_json, "r", encoding="utf-8") as f:
        params = json.load(f)

    issues = []

    input_size = params.get("input_size", 0)
    if input_size != REQUIRED_INPUT_SIZE:
        issues.append(f"input_size={input_size} (应为 {REQUIRED_INPUT_SIZE})")

    test_loss = params.get("model_test_loss", float("inf"))
    if test_loss > MAX_TEST_LOSS:
        issues.append(f"model_test_loss={test_loss:.6f} (应 < {MAX_TEST_LOSS})")

    # 检查 input_vars 顺序
    input_vars = params.get("input_vars", [])
    if len(input_vars) >= 4:
        jerk_idx = next((i for i, v in enumerate(input_vars) if "jerk" in v), -1)
        roll_idx = next((i for i, v in enumerate(input_vars) if v == "roll"), -1)
        if jerk_idx >= 0 and roll_idx >= 0 and jerk_idx > roll_idx:
            issues.append(f"input_vars 顺序错误: lateral_jerk(idx={jerk_idx}) 在 roll(idx={roll_idx}) 之后")

    if issues:
        print_error("模型质量验证未通过:")
        for issue in issues:
            print_error(f"  - {issue}")
        return False

    print_ok(f"input_size: {input_size}")
    print_ok(f"model_test_loss: {test_loss:.6f} (< {MAX_TEST_LOSS})")
    print_ok(f"input_vars 顺序: 正确")

    # 2. 生成验证图表
    validation_dir = os.path.join(output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)

    # 找到训练数据（balanced.csv 或原始 CSV）
    balanced_csv = train_data_csv.replace(".csv", "_balanced.csv")
    viz_data = balanced_csv if os.path.exists(balanced_csv) else train_data_csv

    try:
        run_command(
            [python_exe, "-m", "nnlc_tools.visualize_model", model_json,
             viz_data, "-o", validation_dir],
            "生成模型验证图表",
            check=False,
        )
    except Exception as e:
        print_warn(f"验证图表生成失败: {e}")

    # 3. 输出质量总结
    layers = params.get("layers", [])
    layer_sizes = []
    for l in layers:
        for k, v in l.items():
            if k.endswith("_W"):
                # v 是二维列表 [[...], [...], ...]，列数 = 隐藏层节点数
                layer_sizes.append(len(v[0]) if v else 0)
                break
    if layer_sizes:
        print_info(f"网络结构: {input_size} -> {' -> '.join(str(s) for s in layer_sizes)} -> 1")
    print_info(f"层数: {len(layers)}")

    return True


def step_deploy(model_json, car_name, skip_deploy=False, deploy_dir=None):
    """步骤8: 部署模型到 openpilot 项目。"""
    dest_path = os.path.join(deploy_dir or DEFAULT_DEPLOY_DIR, f"{car_name}.json")

    if skip_deploy:
        print_info(f"跳过部署（模型文件: {model_json}）")
        print_info(f"如需部署，复制到: {dest_path}")
        return model_json

    # 备份现有模型
    if os.path.exists(dest_path):
        backup_path = f"{dest_path}.bak"
        shutil.copy2(dest_path, backup_path)
        print_ok(f"已备份原模型: {backup_path}")

    # 复制新模型
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(model_json, dest_path)
    print_ok(f"模型已部署: {dest_path}")

    return dest_path


# ============================================================
# 主流程
# ============================================================

def auto_train(data_dir, car_name, min_score=None, skip_deploy=False,
               skip_visualize=False, output_dir=None, deploy_dir=None,
               cancel_event=None, process_holder=None, batch_size=16384):
    """执行完整的 NNLC 模型训练流程。

    Args:
        data_dir: rlog 数据目录
        car_name: 车型名称（carFingerprint），如 BYD_TANG_DMI_24
        min_score: 路线修剪阈值，None 则自动推荐
        skip_deploy: 是否跳过部署
        skip_visualize: 是否跳过可视化
        output_dir: 中间文件和图表输出目录；不传则使用项目默认目录
        deploy_dir: 模型 JSON 部署目录；不传则使用项目同级 models 目录
        cancel_event: 可选的训练取消事件
        process_holder: 可选的当前子进程共享容器
        batch_size: Julia 训练批次大小，默认 16384；较小值可降低内存峰值

    Returns:
        模型文件路径
    """
    total_steps = 8 if not skip_visualize else 7
    python_exe = get_python_exe()
    _raise_if_cancelled(cancel_event)

    # 前置检查
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"  NNLC 横向控制模型一键训练")
    print(f"{'='*60}{Colors.RESET}")
    print_info(f"车型: {car_name}")
    print_info(f"数据: {data_dir}")
    print_info(f"Python: {python_exe}")

    if not validate_julia():
        sys.exit(1)
    if not validate_data_dir(data_dir):
        sys.exit(1)

    # 创建输出目录
    output_dir = output_dir or os.path.join(NNLC_TOOLS_DIR, "output", car_name)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print_info(f"输出目录: {output_dir}")

    start_time = time.time()

    # ---- 步骤 1: 提取数据 ----
    _raise_if_cancelled(cancel_event)
    print_step(1, total_steps, "提取横向控制数据")
    lateral_csv = step_extract(data_dir, output_dir, python_exe)

    # ---- 步骤 2: 评估路线质量 ----
    _raise_if_cancelled(cancel_event)
    print_step(2, total_steps, "评估路线质量")
    recommended_score = step_score(lateral_csv, python_exe)
    if min_score is None:
        min_score = recommended_score
        print_info(f"使用自动推荐阈值: min-score={min_score}")
    else:
        print_info(f"使用指定阈值: min-score={min_score}")

    # ---- 步骤 3: 修剪路线 ----
    _raise_if_cancelled(cancel_event)
    print_step(3, total_steps, "修剪低质量路线")
    pruned_csv = step_prune(lateral_csv, min_score, output_dir, python_exe)

    # ---- 步骤 4: 移除干预帧 ----
    _raise_if_cancelled(cancel_event)
    print_step(4, total_steps, "移除干预帧")
    final_csv = step_interventions(pruned_csv, output_dir, python_exe)

    # ---- 步骤 5: 可视化覆盖度 ----
    if not skip_visualize:
        _raise_if_cancelled(cancel_event)
        print_step(5, total_steps, "可视化数据覆盖度")
        step_visualize(final_csv, output_dir, python_exe)

    # ---- 步骤 6: 训练模型 ----
    _raise_if_cancelled(cancel_event)
    train_step = 6 if not skip_visualize else 5
    print_step(train_step, total_steps, f"训练 {car_name} 模型")
    model_json, train_results_dir = step_train(
        final_csv,
        car_name,
        output_dir,
        batch_size=batch_size,
        cancel_event=cancel_event,
        process_holder=process_holder,
    )

    # ---- 步骤 7: 验证模型 ----
    _raise_if_cancelled(cancel_event)
    validate_step = train_step + 1
    print_step(validate_step, total_steps, "验证模型质量")
    quality_ok = step_validate(model_json, final_csv, output_dir, python_exe)

    # ---- 步骤 8: 部署 ----
    _raise_if_cancelled(cancel_event)
    deploy_step = validate_step + 1
    print_step(deploy_step, total_steps, "部署模型")
    deployed_path = step_deploy(model_json, car_name, skip_deploy, deploy_dir=deploy_dir)

    # ---- 总结 ----
    elapsed = time.time() - start_time
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"  训练完成")
    print(f"{'='*60}{Colors.RESET}")

    # 读取模型质量数据
    with open(model_json, "r", encoding="utf-8") as f:
        params = json.load(f)
    test_loss = params.get("model_test_loss", float("inf"))

    print(f"  车型:        {car_name}")
    print(f"  数据目录:    {data_dir}")
    print(f"  输出目录:    {output_dir}")
    print(f"  模型文件:    {deployed_path}")
    print(f"  Test Loss:   {test_loss:.6f} {'(合格)' if test_loss < MAX_TEST_LOSS else '(不合格!)'}")
    print(f"  min-score:   {min_score}")
    print(f"  总耗时:      {elapsed:.1f}s")

    if quality_ok:
        print(f"\n  {Colors.GREEN}模型质量验证通过，可部署到设备{Colors.RESET}")
    else:
        print(f"\n  {Colors.RED}模型质量验证未通过，建议调整参数或补充数据后重新训练{Colors.RESET}")

    return deployed_path


def interactive_mode():
    """交互模式：引导用户输入参数。"""
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"  NNLC 横向控制模型一键训练（交互模式）")
    print(f"{'='*60}{Colors.RESET}\n")

    # 数据目录
    default_data = os.path.join(NNLC_TOOLS_DIR, "data")
    data_dir = input(f"  rlog 数据目录 [{default_data}]: ").strip()
    if not data_dir:
        data_dir = default_data

    # 车型名称
    car_name = input("  车型名称 (carFingerprint，如 BYD_TANG_DMI_24): ").strip()
    if not car_name:
        print_error("车型名称不能为空")
        sys.exit(1)

    # 路线修剪阈值
    min_score_input = input("  路线修剪阈值 min-score [自动推荐]: ").strip()
    min_score = None
    if min_score_input:
        try:
            min_score = int(min_score_input)
        except ValueError:
            print_warn("阈值输入无效，将使用自动推荐值")

    # 是否跳过部署
    deploy_input = input("  是否部署到 openpilot 项目? [Y/n]: ").strip().lower()
    skip_deploy = deploy_input in ("n", "no")

    # 是否跳过可视化
    viz_input = input("  是否生成可视化图表? [Y/n]: ").strip().lower()
    skip_visualize = viz_input in ("n", "no")

    # 确认
    print(f"\n  数据目录: {data_dir}")
    print(f"  车型名称: {car_name}")
    print(f"  修剪阈值: {'自动推荐' if min_score is None else min_score}")
    print(f"  部署:     {'否' if skip_deploy else '是'}")
    print(f"  可视化:   {'否' if skip_visualize else '是'}")

    confirm = input("\n  确认开始训练? [Y/n]: ").strip().lower()
    if confirm in ("n", "no"):
        print("已取消")
        sys.exit(0)

    auto_train(data_dir, car_name, min_score, skip_deploy, skip_visualize)


def main():
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="NNLC 横向控制模型一键训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互模式
  python nnlc_auto_train.py

  # 命令行模式
  python nnlc_auto_train.py --data ./data --car BYD_TANG_DMI_24
  python nnlc_auto_train.py --data ./data --car TOYOTA_CAMRY --min-score 40
  python nnlc_auto_train.py --data ./data --car HONDA_CIVIC --skip-deploy
        """,
    )
    parser.add_argument("--data", "-d", help="rlog 数据目录")
    parser.add_argument("--car", "-c", help="车型名称 (carFingerprint)")
    parser.add_argument("--min-score", type=int, default=None,
                        help="路线修剪阈值 (默认自动推荐)")
    parser.add_argument("--skip-deploy", action="store_true",
                        help="跳过部署到 openpilot 项目")
    parser.add_argument("--skip-viz", action="store_true",
                        help="跳过可视化图表生成")
    parser.add_argument("--output", "-o", help="中间文件、图表和模型输出目录")
    parser.add_argument("--deploy-dir", help="模型 JSON 部署目录（默认项目同级 models）")
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Julia 训练批次大小（默认 16384；低内存可使用 4096）")
    parser.add_argument("--gui", action="store_true", help="启动 Tkinter 操作界面")

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size 必须是正整数")

    # Double-clicking the packaged exe opens the GUI.  Source users retain the
    # original terminal prompts unless they explicitly request ``--gui``.
    if args.gui or (not args.data and not args.car):
        if args.gui or getattr(sys, "frozen", False):
            from nnlc_gui import launch_gui
            launch_gui()
        else:
            interactive_mode()
        return

    # 命令行模式需要 data 和 car 参数
    if not args.data:
        parser.error("--data 参数必填（或使用交互模式）")
    if not args.car:
        parser.error("--car 参数必填（或使用交互模式）")

    auto_train(
        args.data,
        args.car,
        args.min_score,
        args.skip_deploy,
        args.skip_viz,
        output_dir=args.output,
        deploy_dir=args.deploy_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
