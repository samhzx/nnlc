#!/usr/bin/env python3
"""Extract lateral control data from rlogs to CSV/Parquet.

Replaces the complex 3-file pipeline (lat.py -> lat_to_csv.py / lat_to_csv_torquennd.py)
with a single script that reads rlogs and emits one row per controlsState message.

Usage:
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.parquet --format parquet
  python -m nnlc_tools.extract_lateral_data /path/to/rlogs/ -o output.csv --temporal
"""

import argparse
import csv
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from nnlc_tools.bool_utils import parse_bool
from nnlc_tools.route_utils import extract_route_id


RLOG_WORKER_MODULE = "nnlc_tools.extract_rlog_worker"
MAX_RLOGS_PER_WORKER = 100
MAX_RLOG_WORKERS = 16

# Temporal offsets matching nnlc.py's past_times and future_times
PAST_TIMES = [-0.3, -0.2, -0.1]
FUTURE_TIMES = [0.3, 0.6, 1.0, 1.5]

# ---- friction_input 计算所需常量（与 latcontrol_torque_ext_base.py 一致） ----
LAT_PLAN_MIN_IDX = 5
LATERAL_LAG_MOD = 0.0
LAT_JERK_FRICTION_FACTOR = 0.4
LAT_ACCEL_FRICTION_FACTOR_DEFAULT = 0.7
FRICTION_LOOK_AHEAD_V = [1.4, 2.0]
FRICTION_LOOK_AHEAD_BP = [9.0, 30.0]
LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]
# ModelConstants.T_IDXS: 10.0 * (i/32)^2 for i in range(33)
T_IDXS = [10.0 * (i / 32.0) ** 2 for i in range(33)]
T_DIFFS = [T_IDXS[i+1] - T_IDXS[i] for i in range(len(T_IDXS) - 1)]


def _sign(x):
  """符号函数"""
  return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def _get_lookahead_value(future_vals, current_val):
  """前瞻值选取：如果未来值中有反号的则返回0，否则返回绝对值最小的"""
  if len(future_vals) == 0:
    return current_val
  same_sign_vals = [v for v in future_vals if _sign(v) == _sign(current_val)]
  if len(same_sign_vals) < len(future_vals):
    return 0.0
  return min(same_sign_vals + [current_val], key=lambda x: abs(x))


def calculate_friction_input(v_ego, desired_lat_accel, actual_lat_accel,
                              desired_curvature, actual_curvature,
                              model_accels_y, steer_actuator_delay):
  """精确计算 friction_input，与 nnlc.py 运行时 update_friction_input 逻辑一致

  Args:
    v_ego: 车速 m/s
    desired_lat_accel: 期望横向加速度
    actual_lat_accel: 实际横向加速度
    desired_curvature: 期望曲率
    actual_curvature: 实际曲率
    model_accels_y: modelV2.acceleration.y 序列（33个值）
    steer_actuator_delay: 转向执行器延迟（秒）

  Returns:
    friction_input 值；输入或计算结果非有限时返回 None
  """
  if model_accels_y is None or len(model_accels_y) < len(T_IDXS):
    return 0.0

  try:
    scalar_inputs = np.asarray([
        v_ego,
        desired_lat_accel,
        actual_lat_accel,
        desired_curvature,
        actual_curvature,
        steer_actuator_delay,
    ], dtype=np.float64)
    lat_accels = np.asarray(model_accels_y[:len(T_IDXS)], dtype=np.float64)
  except (TypeError, ValueError, OverflowError):
    return None
  if scalar_inputs.shape != (6,) or not np.all(np.isfinite(scalar_inputs)):
    return None
  if lat_accels.shape != (len(T_IDXS),) or not np.all(np.isfinite(lat_accels)):
    return None

  # low_speed_factor
  low_speed_factor = float(np.interp(v_ego, LOW_SPEED_X, LOW_SPEED_Y)) ** 2

  # setpoint 和 measurement
  setpoint = desired_lat_accel + low_speed_factor * desired_curvature
  measurement = actual_lat_accel + low_speed_factor * actual_curvature

  # desired_lat_jerk_time
  desired_lat_jerk_time = max(0.01, steer_actuator_delay) + LATERAL_LAG_MOD

  # lookahead
  lookahead = float(np.interp(v_ego, FRICTION_LOOK_AHEAD_BP, FRICTION_LOOK_AHEAD_V))
  friction_upper_idx = next((i for i, t in enumerate(T_IDXS) if t > lookahead), 16)

  # predicted_lateral_jerk
  predicted_lateral_jerk = np.diff(lat_accels) / np.array(T_DIFFS)

  # desired_lateral_jerk
  desired_lateral_jerk = (float(np.interp(desired_lat_jerk_time, T_IDXS, lat_accels)) - desired_lat_accel) / desired_lat_jerk_time

  # lookahead_lateral_jerk
  end_idx = min(friction_upper_idx, len(predicted_lateral_jerk))
  lookahead_lateral_jerk = _get_lookahead_value(predicted_lateral_jerk[LAT_PLAN_MIN_IDX:end_idx].tolist(), desired_lateral_jerk)

  # lat_accel_friction_factor
  lat_accel_friction_factor = LAT_ACCEL_FRICTION_FACTOR_DEFAULT
  if lookahead_lateral_jerk == 0.0:
    lat_accel_friction_factor = 1.0

  # friction_input
  friction_input = lat_accel_friction_factor * (setpoint - measurement) + LAT_JERK_FRICTION_FACTOR * lookahead_lateral_jerk
  return float(friction_input) if np.isfinite(friction_input) else None

COLUMNS = [
    "timestamp",
    "v_ego",
    "a_ego",
    "steering_angle_deg",
    "steering_rate_deg",
    "steering_torque",
    "steering_pressed",
    "standstill",
    "desired_curvature",
    "curvature",
    "active",
    "lateral_control_type",
    "actual_lateral_accel",
    "desired_lateral_accel",
    "torque_output",
    "saturated",
    "roll",
    "lane_change_state",
    "friction_input",
]


def find_rlogs(input_dir):
    """Find all rlog files in the input directory."""
    patterns = ["**/rlog.zst", "**/rlog.bz2", "**/rlog"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    def natural_path_key(path):
        normalized = os.fspath(path).replace("\\", "/")
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", normalized)]

    return sorted(set(files), key=natural_path_key)


def extract_segment(rlog_path, row_callback=None):
    """Extract lateral data rows from a single rlog file.

    Follows the message iteration pattern from openpilot's
    measure_steering_accuracy.py — accumulate messages by type in a state dict,
    then emit a row when controlsState arrives. One rlog is fully sorted before
    it is processed; the complete dataset is still emitted incrementally.
    """
    from nnlc_tools.logreader import LogReader

    rows = [] if row_callback is None else None
    emit_row = rows.append if row_callback is None else row_callback
    sm = {}
    steer_actuator_delay = 0.1  # 默认值

    lr = None
    try:
        # Sort the complete current rlog before extracting state transitions.
        # The caller still writes rows incrementally, so only one rlog is
        # materialized at a time.
        lr = LogReader(rlog_path, sort_by_time=True)
    except Exception as e:
        raise RuntimeError(f"Could not open {rlog_path}: {e}") from e

    try:
        for msg in lr:
            msg_type = msg.which()

            if msg_type == "carState":
                sm["carState"] = msg.carState
            elif msg_type == "controlsState":
                sm["controlsState"] = msg.controlsState
            elif msg_type == "selfdriveState":
                sm["selfdriveState"] = msg.selfdriveState
            elif msg_type == "liveParameters":
                sm["liveParameters"] = msg.liveParameters
            elif msg_type == "modelV2":
                sm["modelV2"] = msg.modelV2
            elif msg_type == "carParams":
                sm["carParams"] = msg.carParams
                # 提取 steerActuatorDelay
                cp = msg.carParams
                steer_actuator_delay = float(getattr(cp, 'steerActuatorDelay', 0.1))

            # Emit a row on each controlsState when we have carState too
            if msg_type == "controlsState" and "carState" in sm:
                cs = sm["carState"]
                ctrl = sm["controlsState"]

                timestamp = msg.logMonoTime / 1e9

                # Determine lateral control type and extract type-specific fields
                lat_state = ctrl.lateralControlState
                lat_type = lat_state.which()

                actual_lat_accel = float("nan")
                desired_lat_accel = float("nan")
                torque_output = float("nan")
                saturated = False

                if lat_type == "torqueState":
                    ts = lat_state.torqueState
                    actual_lat_accel = ts.actualLateralAccel
                    desired_lat_accel = ts.desiredLateralAccel
                    # torqueState.output 记录的是 -output_torque（见 latcontrol_torque.py:157 pid_log.output = -output_torque）
                    # nnlc 运行时 _ff 用的是未取反的内部 output_torque（与 desired_lateral_accel 同号）
                    # 因此训练数据需取反，使模型学到与运行时 _ff 一致的符号
                    torque_output = -ts.output
                    saturated = ts.saturated
                elif lat_type == "pidState":
                    ps = lat_state.pidState
                    torque_output = ps.output
                    saturated = ps.saturated

                # active moved from controlsState to selfdriveState in newer openpilot
                if "selfdriveState" in sm:
                    active = sm["selfdriveState"].active
                else:
                    active = getattr(ctrl, "active", getattr(ctrl, "activeDEPRECATED", False))

                roll = sm["liveParameters"].roll if "liveParameters" in sm else float("nan")

                lane_change_state = 0
                model_accels_y = None
                if "modelV2" in sm:
                    try:
                        lane_change_state = int(sm["modelV2"].meta.laneChangeState)
                    except (AttributeError, ValueError, TypeError):
                        pass
                    try:
                        model_accels_y = [float(v) for v in sm["modelV2"].acceleration.y]
                    except (AttributeError, TypeError):
                        pass

                # 精确计算 friction_input（与 nnlc.py 运行时逻辑一致）
                friction_input = calculate_friction_input(
                    v_ego=cs.vEgo,
                    desired_lat_accel=desired_lat_accel,
                    actual_lat_accel=actual_lat_accel,
                    desired_curvature=ctrl.desiredCurvature,
                    actual_curvature=ctrl.curvature,
                    model_accels_y=model_accels_y,
                    steer_actuator_delay=steer_actuator_delay,
                )

                row = [
                    timestamp,
                    cs.vEgo,
                    cs.aEgo,
                    cs.steeringAngleDeg,
                    cs.steeringRateDeg,
                    cs.steeringTorque,
                    cs.steeringPressed,
                    cs.standstill,
                    ctrl.desiredCurvature,
                    ctrl.curvature,
                    active,
                    lat_type,
                    actual_lat_accel,
                    desired_lat_accel,
                    torque_output,
                    saturated,
                    roll,
                    lane_change_state,
                    friction_input,
                ]
                emit_row(row)
    except Exception as e:
        raise RuntimeError(f"Error processing {rlog_path}: {e}") from e
    finally:
        if lr is not None:
            lr.close()

    return rows if rows is not None else []


def _temporal_column_specs():
    """Return temporal output columns in the same order as add_temporal_columns."""
    specs = []
    for offset in PAST_TIMES + FUTURE_TIMES:
        suffix = f"_t{offset:+.1f}".replace(".", "").replace("+", "p").replace("-", "m")
        for column in ("actual_lateral_accel", "desired_lateral_accel", "roll"):
            specs.append((int(round(offset / 0.01)), COLUMNS.index(column), f"{column}{suffix}"))
    return specs


def _replace_with_retry(source_path, target_path, attempts=5, delay=0.5):
    """Atomically replace a file, tolerating short-lived Windows file locks."""
    last_error = None
    for attempt in range(attempts):
        try:
            os.replace(source_path, target_path)
            return
        except PermissionError as exc:
            last_error = exc
            if os.name != "nt" or attempt + 1 >= attempts:
                break
            time.sleep(delay)
    if last_error is not None:
        raise PermissionError(
            f"无法替换输出文件：{target_path}。请关闭正在打开该文件的程序 "
            "（例如 Excel、同步软件或杀毒软件）后重试。"
        ) from last_error
    raise RuntimeError(f"无法替换输出文件：{target_path}")


def _remove_with_retry(path, attempts=5, delay=0.1):
    """Best-effort removal for temporary files under transient Windows locks."""
    for attempt in range(attempts):
        try:
            os.unlink(path)
            return True
        except FileNotFoundError:
            return True
        except PermissionError:
            if os.name != "nt" or attempt + 1 >= attempts:
                return False
            time.sleep(delay)
        except OSError:
            return False
    return False


class _StreamingCsvWriter:
    """Write extracted rows while retaining only the temporal look-ahead window."""

    _MAX_PAST = 30
    _MAX_FUTURE = 150

    def __init__(self, output_path, temporal=False, filter_overrides=False, route_id="unknown"):
        self.temporal = temporal
        self.filter_overrides = filter_overrides
        self.route_id = route_id or "unknown"
        self.temporal_specs = _temporal_column_specs() if temporal else []
        self.output_columns = COLUMNS + [spec[2] for spec in self.temporal_specs] + ["route_id"]
        self.handle = open(output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle, lineterminator="\n")
        self.writer.writerow(self.output_columns)
        self.pending = deque()
        self.past = deque(maxlen=self._MAX_PAST)
        self.rows_written = 0
        self.rows_seen = 0
        self.rows_filtered = 0

    @staticmethod
    def _csv_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return value.item() if isinstance(value, np.generic) else value

    def _row_at(self, offset):
        if offset < 0:
            index = len(self.past) + offset
            return self.past[index] if 0 <= index < len(self.past) else None
        return self.pending[offset] if offset < len(self.pending) else None

    def _emit_ready(self):
        center = self.pending[0]
        output_row = list(center)
        for offset, column_index, _ in self.temporal_specs:
            source_row = self._row_at(offset)
            output_row.append(None if source_row is None else source_row[column_index])

        self.past.append(center)
        self.pending.popleft()
        self.writer.writerow([self._csv_value(value) for value in output_row] + [self.route_id])
        self.rows_written += 1

    def accept(self, row):
        self.rows_seen += 1
        if self.filter_overrides and parse_bool(row[COLUMNS.index("steering_pressed")]):
            self.rows_filtered += 1
            return

        if not self.temporal:
            self.writer.writerow([self._csv_value(value) for value in row] + [self.route_id])
            self.rows_written += 1
            return

        self.pending.append(row)
        if len(self.pending) > self._MAX_FUTURE:
            self._emit_ready()

    def finish(self):
        self.finish_segment()
        self.handle.flush()

    def finish_segment(self):
        while self.pending:
            self._emit_ready()
        # Temporal context must not cross an rlog segment boundary.
        self.past.clear()

    def close(self):
        self.handle.close()


class RlogWorkerCrash(RuntimeError):
    """Raised when the isolated native rlog parser exits unexpectedly."""

    def __init__(self, returncode, stderr_text=""):
        self.returncode = returncode
        self.stderr_text = stderr_text.strip()
        if returncode is None:
            code_text = "unknown"
        elif os.name == "nt" or returncode > 0x7FFFFFFF:
            code_text = f"{returncode} (0x{returncode & 0xFFFFFFFF:08X})"
        else:
            code_text = str(returncode)
        message = f"rlog worker exited unexpectedly with code {code_text}"
        if self.stderr_text:
            message += f": {self.stderr_text}"
        super().__init__(message)


def _rlog_worker_command():
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run-module", RLOG_WORKER_MODULE]
    return [sys.executable, "-m", RLOG_WORKER_MODULE]


class _RlogWorker:
    """Keep native rlog parsing outside the parent extraction process."""

    def __init__(self):
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            self._process = subprocess.Popen(
                _rlog_worker_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
            )
        except Exception:
            self._stderr.close()
            raise
        self.files_processed = 0

    def _stderr_since(self, offset):
        self._stderr.flush()
        self._stderr.seek(offset)
        text = self._stderr.read().decode("utf-8", errors="replace")
        self._stderr.seek(0, os.SEEK_END)
        return text

    def process(self, request):
        if self._process.poll() is not None:
            raise RlogWorkerCrash(
                self._process.returncode,
                self._stderr_since(0),
            )
        self._stderr.seek(0, os.SEEK_END)
        stderr_offset = self._stderr.tell()
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
        try:
            self._process.stdin.write(payload)
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            try:
                returncode = self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                returncode = self._process.poll()
            details = self._stderr_since(stderr_offset) or str(exc)
            raise RlogWorkerCrash(returncode, details) from exc

        if not response_line:
            returncode = self._process.wait()
            raise RlogWorkerCrash(
                returncode,
                self._stderr_since(stderr_offset),
            )
        try:
            response = json.loads(response_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            details = self._stderr_since(stderr_offset)
            if not details:
                details = f"invalid worker response: {response_line[:200]!r}"
            raise RlogWorkerCrash(self._process.poll(), details) from exc
        self.files_processed += 1
        return response

    def close(self):
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            self._process = None
        stderr = getattr(self, "_stderr", None)
        if stderr is not None:
            stderr.close()
            self._stderr = None

    def interrupt(self):
        """Stop a blocked request without closing streams from another thread."""
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except (OSError, AttributeError):
            pass


def _csv_header_bytes(columns):
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator="\n").writerow(columns)
    return buffer.getvalue().encode("utf-8")


def _append_worker_csv(source_path, output_handle, expected_header):
    with open(source_path, "rb") as source:
        actual_header = source.readline()
        if actual_header.rstrip(b"\r\n") != expected_header.rstrip(b"\r\n"):
            raise RuntimeError(f"rlog worker produced an invalid CSV header: {source_path}")
        shutil.copyfileobj(source, output_handle, length=1024 * 1024)


def _discard_worker_output(path):
    if not _remove_with_retry(path):
        print(f"WARNING: Unable to remove temporary rlog output: {path}", flush=True)


def resolve_rlog_workers(value, rlog_count):
    """Resolve the requested worker count without overloading the host."""
    if rlog_count <= 0:
        return 1
    if value is None or str(value).strip().lower() == "auto":
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, rlog_count, MAX_RLOG_WORKERS))
    try:
        worker_count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("rlog worker 数必须是正整数或 auto") from exc
    if worker_count <= 0:
        raise ValueError("rlog worker 数必须是正整数或 auto")
    return min(worker_count, rlog_count)


class _RlogWorkerSlot:
    """Own one restartable worker process and serialize its requests."""

    def __init__(self, worker_factory):
        self.worker_factory = worker_factory
        self.worker = None
        self._lock = threading.Lock()

    def process(self, request):
        with self._lock:
            if self.worker is None:
                self.worker = self.worker_factory()
            elif getattr(self.worker, "files_processed", 0) >= MAX_RLOGS_PER_WORKER:
                self._close_unlocked()
                self.worker = self.worker_factory()
            return self.worker.process(request)

    def close(self):
        with self._lock:
            self._close_unlocked()

    def terminate(self):
        """Interrupt an in-flight request without waiting for its lock."""
        worker = self.worker
        if worker is not None:
            worker.interrupt()

    def _close_unlocked(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def _extract_one_rlog(slot, index, rlog_path, output_dir, segment_prefix, temporal,
                      filter_overrides, skip_corrupt, cancel_event=None,
                      stop_event=None):
    """Parse one rlog into a private segment file for ordered merging."""
    crash_retries = 0
    while True:
        if ((cancel_event is not None and cancel_event.is_set())
                or (stop_event is not None and stop_event.is_set())):
            raise RuntimeError("提取已取消")
        segment_file = tempfile.NamedTemporaryFile(
            prefix=f"{segment_prefix}{index:08d}_",
            suffix=".csv",
            dir=output_dir,
            delete=False,
        )
        segment_path = segment_file.name
        segment_file.close()
        try:
            response = slot.process({
                "rlog_path": rlog_path,
                "output_path": segment_path,
                "temporal": temporal,
                "filter_overrides": filter_overrides,
                "route_id": extract_route_id(rlog_path),
            })
        except RlogWorkerCrash as exc:
            slot.close()
            _discard_worker_output(segment_path)
            if skip_corrupt and crash_retries == 0:
                crash_retries += 1
                continue
            reason = str(exc)
            if skip_corrupt:
                return {
                    "index": index,
                    "rlog_path": rlog_path,
                    "status": "skipped",
                    "reason": reason,
                }
            raise RuntimeError(
                f"rlog worker crashed while processing {rlog_path}: {reason}"
            ) from exc
        except Exception:
            _discard_worker_output(segment_path)
            raise

        if response.get("status") != "ok":
            reason = (
                f"{response.get('error_type', 'Error')}: "
                f"{response.get('error', 'unknown worker error')}"
            )
            _discard_worker_output(segment_path)
            if skip_corrupt:
                return {
                    "index": index,
                    "rlog_path": rlog_path,
                    "status": "skipped",
                    "reason": reason,
                }
            raise RuntimeError(f"Error processing {rlog_path}: {reason}")

        return {
            "index": index,
            "rlog_path": rlog_path,
            "status": "ok",
            "segment_path": segment_path,
            "rows_written": int(response["rows_written"]),
            "rows_seen": int(response["rows_seen"]),
            "rows_filtered": int(response["rows_filtered"]),
        }


def extract_rlogs_to_csv(rlog_files, output_path, temporal=False,
                         filter_overrides=False, skip_corrupt=False,
                         show_progress=True, worker_factory=None,
                         rlog_workers="auto", cancel_event=None):
    """Extract rlogs through a restartable parallel native-parser worker pool.

    Each worker is isolated in a child process. Results are merged in the
    original input order so parallel execution remains deterministic.
    """
    worker_factory = worker_factory or _RlogWorker
    rlog_files = [os.fspath(path) for path in rlog_files]
    worker_count = resolve_rlog_workers(rlog_workers, len(rlog_files))
    output_columns = COLUMNS + (
        [spec[2] for spec in _temporal_column_specs()] if temporal else []
    ) + ["route_id"]
    expected_header = _csv_header_bytes(output_columns)
    skipped_rlogs = []
    rows_written = rows_seen = rows_filtered = 0
    output_dir = os.path.dirname(os.path.abspath(output_path))
    segment_prefix = f".nnlc_rlog_{os.getpid()}_{time.time_ns()}_"
    slots = [_RlogWorkerSlot(worker_factory) for _ in range(worker_count)]
    executor = None
    pending_segments = set()
    aborted = False
    stop_event = threading.Event()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    progress = tqdm(
        total=len(rlog_files),
        desc=f"Processing rlogs ({worker_count} workers)",
        disable=not show_progress,
    )
    try:
        with open(output_path, "wb") as output:
            output.write(expected_header)
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="nnlc-rlog-dispatch",
            )
            task_iter = iter(enumerate(rlog_files))
            future_to_slot = {}
            completed = {}
            next_index = 0

            def submit_next(slot):
                if cancel_event is not None and cancel_event.is_set():
                    return
                try:
                    index, rlog_path = next(task_iter)
                except StopIteration:
                    return
                future = executor.submit(
                    _extract_one_rlog,
                    slot,
                    index,
                    rlog_path,
                    output_dir,
                    segment_prefix,
                    temporal,
                    filter_overrides,
                    skip_corrupt,
                    cancel_event,
                    stop_event,
                )
                future_to_slot[future] = slot

            for slot in slots:
                submit_next(slot)

            while future_to_slot:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("提取已取消")
                done, _ = wait(
                    tuple(future_to_slot),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    slot = future_to_slot.pop(future)
                    result = future.result()
                    completed[result["index"]] = result
                    if result["status"] == "ok":
                        pending_segments.add(result["segment_path"])
                    submit_next(slot)

                while next_index in completed:
                    result = completed.pop(next_index)
                    rlog_path = result["rlog_path"]
                    if result["status"] == "skipped":
                        skipped_rlogs.append((rlog_path, result["reason"]))
                        print(
                            f"WARNING: Skipping unreadable rlog: {rlog_path}",
                            flush=True,
                        )
                        print(f"  Reason: {result['reason']}", flush=True)
                    else:
                        _append_worker_csv(
                            result["segment_path"], output, expected_header,
                        )
                        output.flush()
                        rows_written += result["rows_written"]
                        rows_seen += result["rows_seen"]
                        rows_filtered += result["rows_filtered"]
                        pending_segments.discard(result["segment_path"])
                        _discard_worker_output(result["segment_path"])
                    progress.update(1)
                    if show_progress:
                        print(
                            f"Completed rlog [{next_index + 1}/{len(rlog_files)}]: "
                            f"{rlog_path}",
                            flush=True,
                        )
                    next_index += 1
    except BaseException:
        aborted = True
        stop_event.set()
        raise
    finally:
        progress.close()
        if aborted or (cancel_event is not None and cancel_event.is_set()):
            # Closing the child processes first wakes tasks blocked in
            # readline(), allowing the dispatcher threads to exit cleanly.
            for slot in slots:
                slot.terminate()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for slot in slots:
            slot.close()
        for segment_path in pending_segments:
            _discard_worker_output(segment_path)
        # A worker may have finished successfully while the main thread was
        # handling another worker's failure. Remove any segment not yet
        # reported back to the merger.
        for name in os.listdir(output_dir):
            if name.startswith(segment_prefix) and name.endswith(".csv"):
                _discard_worker_output(os.path.join(output_dir, name))

    return {
        "rows_written": rows_written,
        "rows_seen": rows_seen,
        "rows_filtered": rows_filtered,
        "skipped_rlogs": skipped_rlogs,
    }


def add_temporal_columns(df):
    """Add lagged/lead columns for temporal model training.

    Adds columns at offsets matching nnlc.py's past_times [-0.3, -0.2, -0.1]
    and future_times [0.3, 0.6, 1.0, 1.5].
    """
    dt = 0.01  # controlsState runs at 100Hz
    temporal_cols = ["actual_lateral_accel", "desired_lateral_accel", "roll"]

    for offset in PAST_TIMES + FUTURE_TIMES:
        shift_frames = int(round(offset / dt))
        suffix = f"_t{offset:+.1f}".replace(".", "").replace("+", "p").replace("-", "m")

        for col in temporal_cols:
            if col in df.columns:
                df[f"{col}{suffix}"] = df[col].shift(-shift_frames)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract lateral control data from rlogs to CSV/Parquet.",
    )
    parser.add_argument("input", help="Directory containing rlog files")
    parser.add_argument("-o", "--output", default="lateral_data.csv",
                        help="Output file path (default: lateral_data.csv)")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None,
                        help="Output format (default: inferred from extension)")
    parser.add_argument("--temporal", action="store_true",
                        help="Add temporal lag/lead columns for NNLC training")
    parser.add_argument("--filter-overrides", action="store_true",
                        help="Drop rows where driver overrides (steering_pressed=True)")
    parser.add_argument("--skip-corrupt", action="store_true",
                        help="Skip unreadable/corrupt rlog files and continue; report them at the end")
    parser.add_argument(
        "--rlog-workers",
        default="auto",
        metavar="N",
        help="rlog 并行解析 worker 数（正整数或 auto，默认 auto；1 为串行模式）",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    rlog_files = find_rlogs(args.input)
    if not rlog_files:
        print(f"ERROR: No rlog files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(rlog_files)} rlog files")

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "csv"

    if fmt == "csv":
        # CSV is the pipeline's native format. Write rows as they arrive so
        # memory usage is bounded by one sorted rlog plus the temporal
        # look-ahead buffer instead of the complete dataset. Use a temporary
        # sibling file so a failure can never replace a valid existing output
        # with a partial CSV.
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(args.output)}.", suffix=".tmp",
            dir=output_dir, delete=False,
        )
        temp_path = temp_file.name
        temp_file.close()
        success = False
        try:
            stats = extract_rlogs_to_csv(
                rlog_files,
                temp_path,
                temporal=args.temporal,
                filter_overrides=args.filter_overrides,
                skip_corrupt=args.skip_corrupt,
                rlog_workers=args.rlog_workers,
            )
            rows_written = stats["rows_written"]
            rows_seen = stats["rows_seen"]
            rows_filtered = stats["rows_filtered"]
            if rows_written > 0:
                _replace_with_retry(temp_path, args.output)
                success = True
        finally:
            if not success:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

        if rows_written == 0:
            if stats["skipped_rlogs"]:
                print(f"Skipped {len(stats['skipped_rlogs'])} corrupt/unreadable rlog files")
                for path, reason in stats["skipped_rlogs"]:
                    print(f"  - {path}: {reason}")
            print("ERROR: No data extracted from any rlog files")
            sys.exit(1)
        print(f"Extracted {rows_written} rows")
        if rows_filtered:
            ratio = rows_filtered / max(rows_seen, 1)
            print(f"Filtered {rows_filtered} override rows ({ratio:.1%} of data)")
        if args.temporal:
            print("Added temporal columns with per-segment streaming")
    else:
        # Parquet engines require a DataFrame, but rows are first streamed to
        # a temporary CSV so extraction does not accumulate a second list of
        # every row in memory.
        temp_file = tempfile.NamedTemporaryFile(prefix="nnlc_extract_", suffix=".csv", delete=False)
        temp_path = temp_file.name
        temp_file.close()
        extraction_success = False
        try:
            stats = extract_rlogs_to_csv(
                rlog_files,
                temp_path,
                temporal=args.temporal,
                filter_overrides=args.filter_overrides,
                skip_corrupt=args.skip_corrupt,
                rlog_workers=args.rlog_workers,
            )
            extraction_success = stats["rows_written"] > 0
        finally:
            if not extraction_success:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

        if not extraction_success:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            if stats["skipped_rlogs"]:
                print(f"Skipped {len(stats['skipped_rlogs'])} corrupt/unreadable rlog files")
                for path, reason in stats["skipped_rlogs"]:
                    print(f"  - {path}: {reason}")
            print("ERROR: No data extracted from any rlog file")
            sys.exit(1)

        parquet_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parquet_dir, exist_ok=True)
        parquet_temp = tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(args.output)}.", suffix=".tmp",
            dir=parquet_dir, delete=False,
        )
        parquet_temp_path = parquet_temp.name
        parquet_temp.close()
        try:
            df = pd.read_csv(temp_path)
            print(f"Extracted {len(df)} rows")
            df.to_parquet(parquet_temp_path, index=False)
            _replace_with_retry(parquet_temp_path, args.output)
        finally:
            for path in (temp_path, parquet_temp_path):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    if stats["skipped_rlogs"]:
        print(f"Skipped {len(stats['skipped_rlogs'])} corrupt/unreadable rlog files")
        for path, reason in stats["skipped_rlogs"]:
            print(f"  - {path}: {reason}")
    print(f"Saved to {args.output} ({fmt})")


if __name__ == "__main__":
    main()
