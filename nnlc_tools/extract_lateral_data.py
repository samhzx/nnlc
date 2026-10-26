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
import heapq
import io
import json
import math
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
from nnlc_tools.model_inputs import (
    DEFAULT_INPUT_DIM,
    INPUT_DIMENSIONS,
    temporal_offsets,
    temporal_suffix,
)
from nnlc_tools.route_utils import extract_route_id


RLOG_WORKER_MODULE = "nnlc_tools.extract_rlog_worker"
MAX_RLOGS_PER_WORKER = 100
MAX_RLOG_WORKERS = 16
REORDER_WINDOW_SECONDS = 2.0
NANOSECONDS_PER_SECOND = 1_000_000_000

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
T_IDXS_ARRAY = np.asarray(T_IDXS, dtype=np.float64)
T_DIFFS_ARRAY = np.asarray(T_DIFFS, dtype=np.float64)
LOW_SPEED_X_ARRAY = np.asarray(LOW_SPEED_X, dtype=np.float64)
LOW_SPEED_Y_ARRAY = np.asarray(LOW_SPEED_Y, dtype=np.float64)
FRICTION_LOOK_AHEAD_BP_ARRAY = np.asarray(FRICTION_LOOK_AHEAD_BP, dtype=np.float64)
FRICTION_LOOK_AHEAD_V_ARRAY = np.asarray(FRICTION_LOOK_AHEAD_V, dtype=np.float64)


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
                              model_accels_y, steer_actuator_delay,
                              *, model_accels_valid=None):
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
    scalar_inputs = tuple(float(value) for value in (
        v_ego,
        desired_lat_accel,
        actual_lat_accel,
        desired_curvature,
        actual_curvature,
        steer_actuator_delay,
    ))
    if (
        isinstance(model_accels_y, np.ndarray)
        and model_accels_y.dtype == np.float64
        and model_accels_y.shape == (len(T_IDXS),)
    ):
      lat_accels = model_accels_y
    else:
      lat_accels = np.asarray(model_accels_y[:len(T_IDXS)], dtype=np.float64)
  except (TypeError, ValueError, OverflowError):
    return None
  if not all(math.isfinite(value) for value in scalar_inputs):
    return None
  if lat_accels.shape != (len(T_IDXS),):
    return None
  if model_accels_valid is False or (
      model_accels_valid is None and not np.all(np.isfinite(lat_accels))
  ):
    return None

  v_ego, desired_lat_accel, actual_lat_accel, desired_curvature, actual_curvature, steer_actuator_delay = scalar_inputs

  # low_speed_factor
  low_speed_factor = float(np.interp(v_ego, LOW_SPEED_X_ARRAY, LOW_SPEED_Y_ARRAY)) ** 2

  # setpoint 和 measurement
  setpoint = desired_lat_accel + low_speed_factor * desired_curvature
  measurement = actual_lat_accel + low_speed_factor * actual_curvature

  # desired_lat_jerk_time
  desired_lat_jerk_time = max(0.01, steer_actuator_delay) + LATERAL_LAG_MOD

  # lookahead
  lookahead = float(np.interp(
      v_ego, FRICTION_LOOK_AHEAD_BP_ARRAY, FRICTION_LOOK_AHEAD_V_ARRAY,
  ))
  friction_upper_idx = next((i for i, t in enumerate(T_IDXS) if t > lookahead), 16)

  # desired_lateral_jerk
  desired_lateral_jerk = (
      float(np.interp(desired_lat_jerk_time, T_IDXS_ARRAY, lat_accels))
      - desired_lat_accel
  ) / desired_lat_jerk_time

  # Scan only the active lookahead range without allocating a jerk array/list.
  lookahead_lateral_jerk = desired_lateral_jerk
  desired_jerk_sign = _sign(desired_lateral_jerk)
  end_idx = min(friction_upper_idx, len(T_DIFFS))
  for index in range(LAT_PLAN_MIN_IDX, end_idx):
    predicted_jerk = (lat_accels[index + 1] - lat_accels[index]) / T_DIFFS_ARRAY[index]
    if _sign(predicted_jerk) != desired_jerk_sign:
      lookahead_lateral_jerk = 0.0
      break
    if abs(predicted_jerk) < abs(lookahead_lateral_jerk):
      lookahead_lateral_jerk = predicted_jerk

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


DATE_DIR_NAME_RE = re.compile(r"^\d{8}$")
def day_csv_name(input_dim=DEFAULT_INPUT_DIM):
    temporal_offsets(input_dim)
    return f"lateral_data_{input_dim}d.csv"


def _natural_path_key(path):
    normalized = os.fspath(path).replace("\\", "/")
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", normalized)]


def find_rlogs(input_dir):
    """Find all rlog files in the input directory."""
    patterns = ["**/rlog.zst", "**/rlog.bz2", "**/rlog"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pattern), recursive=True))
    return sorted(set(files), key=_natural_path_key)


def is_date_dir_name(name):
    """Return True when a path component is an 8-digit YYYYMMDD directory name."""
    return bool(DATE_DIR_NAME_RE.fullmatch(os.fspath(name)))


def uses_date_cached_extraction(input_dir):
    """Enable per-day CSV caches when the rlog root is organized by YYYYMMDD.

    Date mode is used when the input directory itself is named YYYYMMDD, or
    when at least one of its direct subdirectories is named YYYYMMDD.
    """
    input_dir = os.path.abspath(input_dir)
    if is_date_dir_name(os.path.basename(input_dir)):
        return True
    try:
        names = os.listdir(input_dir)
    except OSError:
        return False
    return any(
        is_date_dir_name(name) and os.path.isdir(os.path.join(input_dir, name))
        for name in names
    )


def date_directory_for_rlog(input_dir, rlog_path):
    """Return the YYYYMMDD directory that owns an rlog, or None if it has none.

    When the input root itself is a date directory, every rlog under it belongs
    to that root. Otherwise the first YYYYMMDD path component after the input
    root is the cache directory.
    """
    input_dir = os.path.abspath(input_dir)
    rlog_path = os.path.abspath(rlog_path)
    if is_date_dir_name(os.path.basename(input_dir)):
        try:
            common = os.path.commonpath([input_dir, rlog_path])
        except ValueError:
            return None
        return input_dir if common == input_dir else None
    try:
        relative = os.path.relpath(rlog_path, input_dir)
    except ValueError:
        return None
    if relative.startswith(".."):
        return None
    current = input_dir
    for part in relative.split(os.sep)[:-1]:
        current = os.path.join(current, part)
        if is_date_dir_name(part):
            return current
    return None


def group_rlogs_by_date_directory(input_dir, rlog_files):
    """Group rlogs by their date directory, preserving the given file order."""
    grouped = {}
    skipped = []
    for rlog_path in rlog_files:
        date_dir = date_directory_for_rlog(input_dir, rlog_path)
        if date_dir is None:
            skipped.append(rlog_path)
            continue
        grouped.setdefault(date_dir, []).append(rlog_path)
    return grouped, skipped


class _LateEventBeyondWindow(RuntimeError):
    """Signals that a source event arrived behind the committed time window."""


def _snapshot_event(event, sequence):
    """Copy only fields used by extraction before a borrowed Cap'n Proto reader expires."""
    timestamp_ns = int(event.logMonoTime)
    msg_type = event.which()

    if msg_type == "carState":
        state = event.carState
        payload = (
            state.vEgo,
            state.aEgo,
            state.steeringAngleDeg,
            state.steeringRateDeg,
            state.steeringTorque,
            state.steeringPressed,
            state.standstill,
        )
    elif msg_type == "controlsState":
        ctrl = event.controlsState
        lat_state = ctrl.lateralControlState
        lat_type = lat_state.which()
        actual_lat_accel = float("nan")
        desired_lat_accel = float("nan")
        torque_output = float("nan")
        saturated = False
        if lat_type == "torqueState":
            torque_state = lat_state.torqueState
            actual_lat_accel = torque_state.actualLateralAccel
            desired_lat_accel = torque_state.desiredLateralAccel
            torque_output = -torque_state.output
            saturated = torque_state.saturated
        elif lat_type == "pidState":
            pid_state = lat_state.pidState
            torque_output = pid_state.output
            saturated = pid_state.saturated
        payload = (
            lat_type,
            actual_lat_accel,
            desired_lat_accel,
            torque_output,
            saturated,
            ctrl.desiredCurvature,
            ctrl.curvature,
            getattr(ctrl, "active", getattr(ctrl, "activeDEPRECATED", False)),
        )
    elif msg_type == "selfdriveState":
        payload = event.selfdriveState.active
    elif msg_type == "liveParameters":
        payload = event.liveParameters.roll
    elif msg_type == "modelV2":
        model = event.modelV2
        lane_change_state = 0
        try:
            lane_change_state = int(model.meta.laneChangeState)
        except (AttributeError, ValueError, TypeError):
            pass
        model_accels_y = None
        model_accels_valid = None
        try:
            model_accels_y = np.asarray(
                [float(value) for value in model.acceleration.y], dtype=np.float64,
            )
            model_accels_valid = bool(np.all(np.isfinite(model_accels_y[:len(T_IDXS)])))
        except (AttributeError, TypeError):
            pass
        payload = (lane_change_state, model_accels_y, model_accels_valid)
    elif msg_type == "carParams":
        payload = float(getattr(event.carParams, "steerActuatorDelay", 0.1))
    else:
        return None

    return timestamp_ns, sequence, msg_type, payload


def _iter_reordered_snapshots(log_reader, window_ns, stats):
    pending = []
    max_seen_ns = None
    sequence = 0
    events = iter(log_reader.iter_events(skip_copy=True))
    while True:
        try:
            event = next(events)
        except StopIteration:
            break
        snapshot = _snapshot_event(event, sequence)
        sequence += 1
        del event
        if snapshot is None:
            continue

        timestamp_ns = snapshot[0]
        if max_seen_ns is not None:
            lateness_ns = max_seen_ns - timestamp_ns
            stats["max_lateness_seconds"] = max(
                stats["max_lateness_seconds"],
                max(0, lateness_ns) / NANOSECONDS_PER_SECOND,
            )
            if lateness_ns > window_ns:
                raise _LateEventBeyondWindow(
                    f"event at {timestamp_ns} arrived {lateness_ns / NANOSECONDS_PER_SECOND:.6f}s behind the reorder watermark"
                )

        max_seen_ns = timestamp_ns if max_seen_ns is None else max(max_seen_ns, timestamp_ns)
        heapq.heappush(pending, (timestamp_ns, snapshot[1], snapshot))
        watermark_ns = max_seen_ns - window_ns
        while pending and pending[0][0] <= watermark_ns:
            yield heapq.heappop(pending)[2]

    while pending:
        yield heapq.heappop(pending)[2]


def _iter_exact_snapshots(log_reader):
    """Use the legacy full, stable sort as the correctness fallback/oracle."""
    events = list(enumerate(log_reader.iter_events(skip_copy=False)))
    events.sort(key=lambda item: item[1].logMonoTime)
    for sequence, event in events:
        snapshot = _snapshot_event(event, sequence)
        if snapshot is not None:
            yield snapshot


def _rows_from_snapshots(snapshots):
    state = {}
    steer_actuator_delay = 0.1
    rows = []

    for timestamp_ns, _sequence, msg_type, payload in snapshots:
        if msg_type == "carState":
            state["carState"] = payload
        elif msg_type == "controlsState":
            state["controlsState"] = payload
        elif msg_type == "selfdriveState":
            state["selfdriveState"] = payload
        elif msg_type == "liveParameters":
            state["liveParameters"] = payload
        elif msg_type == "modelV2":
            state["modelV2"] = payload
        elif msg_type == "carParams":
            steer_actuator_delay = payload

        if msg_type != "controlsState" or "carState" not in state:
            continue

        car_state = state["carState"]
        (
            lat_type,
            actual_lat_accel,
            desired_lat_accel,
            torque_output,
            saturated,
            desired_curvature,
            curvature,
            active_fallback,
        ) = payload
        active = state.get("selfdriveState", active_fallback)
        roll = state.get("liveParameters", float("nan"))
        model_state = state.get("modelV2")
        lane_change_state = model_state[0] if model_state is not None else 0
        model_accels_y = model_state[1] if model_state is not None else None
        model_accels_valid = model_state[2] if model_state is not None else None

        friction_input = calculate_friction_input(
            v_ego=car_state[0],
            desired_lat_accel=desired_lat_accel,
            actual_lat_accel=actual_lat_accel,
            desired_curvature=desired_curvature,
            actual_curvature=curvature,
            model_accels_y=model_accels_y,
            steer_actuator_delay=steer_actuator_delay,
            model_accels_valid=model_accels_valid,
        )
        rows.append([
            timestamp_ns / NANOSECONDS_PER_SECOND,
            *car_state,
            desired_curvature,
            curvature,
            active,
            lat_type,
            actual_lat_accel,
            desired_lat_accel,
            torque_output,
            saturated,
            roll,
            lane_change_state,
            friction_input,
        ])
    return rows


def extract_segment(rlog_path, row_callback=None, *, reorder_window_seconds=REORDER_WINDOW_SECONDS,
                    stats_out=None):
    """Extract one rlog with bounded reordering and exact-sort fallback.

    Rows remain local to this segment until its ordering is accepted, so a
    late-event fallback can never leak partial or duplicated rows to a writer.
    """
    from nnlc_tools.logreader import LogReader

    try:
        reorder_window_seconds = float(reorder_window_seconds)
        if not math.isfinite(reorder_window_seconds) or reorder_window_seconds < 0:
            raise ValueError("reorder_window_seconds must be finite and non-negative")
        window_ns = int(round(reorder_window_seconds * NANOSECONDS_PER_SECOND))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reorder_window_seconds must be finite and non-negative") from exc

    stats = stats_out if stats_out is not None else {}
    stats.setdefault("used_exact_fallback", False)
    stats.setdefault("max_lateness_seconds", 0.0)
    try:
        lr = LogReader(rlog_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open {rlog_path}: {exc}") from exc

    try:
        try:
            rows = _rows_from_snapshots(
                _iter_reordered_snapshots(lr, window_ns, stats),
            )
        except _LateEventBeyondWindow:
            stats["used_exact_fallback"] = True
            rows = _rows_from_snapshots(_iter_exact_snapshots(lr))
        if row_callback is not None:
            for row in rows:
                row_callback(row)
            return []
        return rows
    except Exception as exc:
        raise RuntimeError(f"Error processing {rlog_path}: {exc}") from exc
    finally:
        lr.close()


def _temporal_column_specs(input_dim=DEFAULT_INPUT_DIM):
    """Return temporal output columns in the same order as add_temporal_columns."""
    specs = []
    for offset in temporal_offsets(input_dim):
        suffix = temporal_suffix(offset)
        for column in ("actual_lateral_accel", "desired_lateral_accel", "roll"):
            specs.append((int(round(offset / 0.01)), COLUMNS.index(column), f"{column}{suffix}"))
    return specs


def csv_output_columns(temporal=False, input_dim=DEFAULT_INPUT_DIM):
    """Return the CSV column names produced by extraction."""
    temporal_offsets(input_dim)
    return COLUMNS + (
        [spec[2] for spec in _temporal_column_specs(input_dim)] if temporal else []
    ) + ["route_id"]


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

    _MAX_FUTURE = 150

    def __init__(self, output_path, temporal=False, filter_overrides=False,
                 route_id="unknown", input_dim=DEFAULT_INPUT_DIM):
        temporal_offsets(input_dim)
        self.temporal = temporal
        self.filter_overrides = filter_overrides
        self.route_id = route_id or "unknown"
        self.temporal_specs = _temporal_column_specs(input_dim) if temporal else []
        self.output_columns = COLUMNS + [spec[2] for spec in self.temporal_specs] + ["route_id"]
        self.handle = open(output_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.handle, lineterminator="\n")
        self.writer.writerow(self.output_columns)
        self.pending = deque()
        max_past = max(
            (abs(offset) for offset, _column, _name in self.temporal_specs if offset < 0),
            default=0,
        )
        self.past = deque(maxlen=max_past or 1)
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
                      input_dim, filter_overrides, skip_corrupt, cancel_event=None,
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
                "input_dim": input_dim,
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
            "used_exact_fallback": bool(response.get("used_exact_fallback", False)),
            "max_lateness_seconds": float(response.get("max_lateness_seconds", 0.0)),
        }


def extract_rlogs_to_csv(rlog_files, output_path, temporal=False,
                         filter_overrides=False, skip_corrupt=False,
                         show_progress=True, worker_factory=None,
                         rlog_workers="auto", cancel_event=None,
                         input_dim=DEFAULT_INPUT_DIM):
    """Extract rlogs through a restartable parallel native-parser worker pool.

    Each worker is isolated in a child process. Results are merged in the
    original input order so parallel execution remains deterministic.
    """
    worker_factory = worker_factory or _RlogWorker
    temporal_offsets(input_dim)
    rlog_files = [os.fspath(path) for path in rlog_files]
    worker_count = resolve_rlog_workers(rlog_workers, len(rlog_files))
    output_columns = csv_output_columns(temporal, input_dim)
    expected_header = _csv_header_bytes(output_columns)
    skipped_rlogs = []
    rows_written = rows_seen = rows_filtered = 0
    exact_fallbacks = 0
    max_lateness_seconds = 0.0
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
                    input_dim,
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
                        exact_fallbacks += int(result.get("used_exact_fallback", False))
                        max_lateness_seconds = max(
                            max_lateness_seconds,
                            result.get("max_lateness_seconds", 0.0),
                        )
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
        "exact_sort_fallbacks": exact_fallbacks,
        "max_lateness_seconds": max_lateness_seconds,
    }


def _temporary_output_path(output_path):
    """Create a sibling temp file in the destination directory."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(output_path)}.", suffix=".tmp",
        dir=output_dir, delete=False,
    )
    temp_path = temp_file.name
    temp_file.close()
    return temp_path


def _discard_temp_output(temp_path):
    if not _remove_with_retry(temp_path):
        print(f"WARNING: Unable to remove temporary extract output: {temp_path}", flush=True)


def daily_csv_is_reusable(path, expected_header):
    """Return True when a per-day cache exists, is readable, and has the expected header."""
    if not os.path.isfile(path):
        return False
    try:
        expected_columns = next(csv.reader(io.StringIO(
            expected_header.decode("utf-8"),
        )))
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            actual_columns = next(reader, None)
            if actual_columns != expected_columns:
                return False
            return all(len(row) == len(expected_columns) for row in reader)
    except (OSError, UnicodeError, csv.Error):
        return False


def _extract_rlogs_to_csv_atomic(rlog_files, output_path, **kwargs):
    """Extract rlogs to a temp sibling, then replace the destination on success."""
    temp_path = _temporary_output_path(output_path)
    success = False
    try:
        stats = extract_rlogs_to_csv(rlog_files, temp_path, **kwargs)
        cancel_event = kwargs.get("cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("提取已取消")
        if stats["rows_written"] > 0:
            _replace_with_retry(temp_path, output_path)
            success = True
        return stats
    finally:
        if not success:
            _discard_temp_output(temp_path)


def _merge_csv_files(csv_paths, output_path, expected_header):
    """Concatenate daily CSVs in the given order, keeping a single header."""
    temp_path = _temporary_output_path(output_path)
    success = False
    rows_written = 0
    try:
        with open(temp_path, "wb") as output:
            output.write(expected_header)
            for csv_path in csv_paths:
                with open(csv_path, "rb") as source:
                    actual_header = source.readline()
                    if actual_header.rstrip(b"\r\n") != expected_header.rstrip(b"\r\n"):
                        raise RuntimeError(f"Cached CSV has an invalid header: {csv_path}")
                    last_byte = b""
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        rows_written += chunk.count(b"\n")
                        last_byte = chunk[-1:]
                        output.write(chunk)
                    if last_byte and last_byte not in (b"\n", b"\r"):
                        output.write(b"\n")
                        rows_written += 1
        if rows_written > 0:
            _replace_with_retry(temp_path, output_path)
            success = True
        return rows_written
    finally:
        if not success:
            _discard_temp_output(temp_path)


def _empty_extract_stats():
    return {
        "rows_written": 0,
        "rows_seen": 0,
        "rows_filtered": 0,
        "skipped_rlogs": [],
        "exact_sort_fallbacks": 0,
        "max_lateness_seconds": 0.0,
        "reused_days": 0,
        "extracted_days": 0,
        "skipped_outside_date_dirs": [],
    }


def extract_directory_to_csv(input_dir, output_path, temporal=False,
                             filter_overrides=False, skip_corrupt=False,
                             show_progress=True, worker_factory=None,
                             rlog_workers="auto", cancel_event=None,
                             rlog_files=None, input_dim=DEFAULT_INPUT_DIM):
    """Extract an rlog directory to CSV, using per-day caches when applicable.

    When the input tree is organized by YYYYMMDD, each date directory stores
    a dimension-specific CSV. Only the current dimension's cache is reused.
    Newly extracted days are written atomically so a cancelled run cannot
    replace a day file with a partial CSV. The combined output is then
    concatenated in date-name order.

    Without YYYYMMDD directories, this writes only ``output_path`` and does
    not create files inside the data directory.
    """
    if rlog_files is None:
        rlog_files = find_rlogs(input_dir)
    temporal_offsets(input_dim)
    rlog_files = [os.fspath(path) for path in rlog_files]
    extract_kwargs = {
        "temporal": temporal,
        "input_dim": input_dim,
        "filter_overrides": filter_overrides,
        "skip_corrupt": skip_corrupt,
        "show_progress": show_progress,
        "worker_factory": worker_factory,
        "rlog_workers": rlog_workers,
        "cancel_event": cancel_event,
    }

    if not uses_date_cached_extraction(input_dir):
        stats = _extract_rlogs_to_csv_atomic(rlog_files, output_path, **extract_kwargs)
        stats.update({
            "reused_days": 0,
            "extracted_days": 0,
            "skipped_outside_date_dirs": [],
        })
        return stats

    grouped, skipped_outside = group_rlogs_by_date_directory(input_dir, rlog_files)
    stats = _empty_extract_stats()
    stats["skipped_outside_date_dirs"] = skipped_outside
    for path in skipped_outside:
        print(f"WARNING: Skipping rlog outside YYYYMMDD directory: {path}", flush=True)

    expected_header = _csv_header_bytes(csv_output_columns(temporal, input_dim))
    day_csvs = []
    extract_failed = False
    date_dirs = sorted(
        grouped,
        key=lambda path: (_natural_path_key(os.path.basename(path)), _natural_path_key(path)),
    )
    for date_dir in date_dirs:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("提取已取消")
        day_csv = os.path.join(date_dir, day_csv_name(input_dim))
        day_rlogs = grouped[date_dir]
        if daily_csv_is_reusable(day_csv, expected_header):
            print(f"Reusing cached CSV: {day_csv}", flush=True)
            stats["reused_days"] += 1
            day_csvs.append(day_csv)
            continue
        print(f"Extracting {len(day_rlogs)} rlog(s) to {day_csv}", flush=True)
        day_stats = _extract_rlogs_to_csv_atomic(day_rlogs, day_csv, **extract_kwargs)
        stats["rows_seen"] += day_stats["rows_seen"]
        stats["rows_filtered"] += day_stats["rows_filtered"]
        stats["skipped_rlogs"].extend(day_stats["skipped_rlogs"])
        stats["exact_sort_fallbacks"] += day_stats["exact_sort_fallbacks"]
        stats["max_lateness_seconds"] = max(
            stats["max_lateness_seconds"],
            day_stats["max_lateness_seconds"],
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("提取已取消")
        if day_stats["rows_written"] <= 0:
            extract_failed = True
            continue
        stats["extracted_days"] += 1
        day_csvs.append(day_csv)

    print(
        f"Date-cached extraction: reused {stats['reused_days']} day(s), "
        f"extracted {stats['extracted_days']} day(s)",
        flush=True,
    )
    if extract_failed:
        print(
            "ERROR: No data extracted from one or more YYYYMMDD directories",
            flush=True,
        )
        return stats
    if not day_csvs:
        return stats
    stats["rows_written"] = _merge_csv_files(day_csvs, output_path, expected_header)
    return stats


def add_temporal_columns(df, input_dim=DEFAULT_INPUT_DIM):
    """Add lagged/lead columns for temporal model training.

    Adds columns at offsets matching the selected NNLC input profile.
    """
    dt = 0.01  # controlsState runs at 100Hz
    temporal_cols = ["actual_lateral_accel", "desired_lateral_accel", "roll"]

    for offset in temporal_offsets(input_dim):
        shift_frames = int(round(offset / dt))
        suffix = temporal_suffix(offset)

        for col in temporal_cols:
            if col in df.columns:
                df[f"{col}{suffix}"] = df[col].shift(-shift_frames)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract lateral control data from rlogs to CSV/Parquet.",
    )
    parser.add_argument("input", help="Directory containing rlog files")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (default: lateral_data_<input-dim>d.csv)")
    parser.add_argument("--format", choices=["csv", "parquet"], default=None,
                        help="Output format (default: inferred from extension)")
    parser.add_argument("--temporal", action="store_true",
                        help="Add temporal lag/lead columns for NNLC training")
    parser.add_argument("--input-dim", type=int, choices=INPUT_DIMENSIONS,
                        default=DEFAULT_INPUT_DIM,
                        help="NNLC model input dimension (default: 18)")
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
    output_format = args.format or (
        "parquet" if args.output and args.output.endswith(".parquet") else "csv"
    )
    default_extension = ".parquet" if output_format == "parquet" else ".csv"
    output_path = args.output or f"lateral_data_{args.input_dim}d{default_extension}"

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    rlog_files = find_rlogs(args.input)
    if not rlog_files:
        print(f"ERROR: No rlog files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(rlog_files)} rlog files")

    # Determine output format
    fmt = output_format

    extract_kwargs = {
        "temporal": args.temporal,
        "input_dim": args.input_dim,
        "filter_overrides": args.filter_overrides,
        "skip_corrupt": args.skip_corrupt,
        "rlog_workers": args.rlog_workers,
        "rlog_files": rlog_files,
    }

    if fmt == "csv":
        # CSV is the pipeline's native format. Write rows as they arrive so
        # memory usage is bounded by one sorted rlog plus the temporal
        # look-ahead buffer instead of the complete dataset. Date-organized
        # trees also keep a per-day CSV beside the rlogs, then merge into -o.
        stats = extract_directory_to_csv(args.input, output_path, **extract_kwargs)
        rows_written = stats["rows_written"]
        rows_seen = stats["rows_seen"]
        rows_filtered = stats["rows_filtered"]

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
            stats = extract_directory_to_csv(
                args.input,
                temp_path,
                **extract_kwargs,
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

        parquet_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parquet_dir, exist_ok=True)
        parquet_temp = tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(output_path)}.", suffix=".tmp",
            dir=parquet_dir, delete=False,
        )
        parquet_temp_path = parquet_temp.name
        parquet_temp.close()
        try:
            df = pd.read_csv(temp_path)
            print(f"Extracted {len(df)} rows")
            df.to_parquet(parquet_temp_path, index=False)
            _replace_with_retry(parquet_temp_path, output_path)
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
    print(
        "Reorder summary: "
        f"{stats['exact_sort_fallbacks']} exact-sort fallbacks; "
        f"max observed lateness {stats['max_lateness_seconds']:.6f}s"
    )
    print(f"Saved to {output_path} ({fmt})")


if __name__ == "__main__":
    main()
