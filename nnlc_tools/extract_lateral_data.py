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
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    friction_input 值
  """
  if model_accels_y is None or len(model_accels_y) < 2:
    return 0.0

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
  lat_accels = np.array(model_accels_y[:len(T_IDXS)])
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
  return float(friction_input)

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
    return sorted(set(files))


def extract_segment(rlog_path):
    """Extract lateral data rows from a single rlog file.

    Follows the message iteration pattern from openpilot's
    measure_steering_accuracy.py — accumulate messages by type in a state dict,
    then emit a row when controlsState arrives.
    """
    from nnlc_tools.logreader import LogReader

    rows = []
    sm = {}
    steer_actuator_delay = 0.1  # 默认值

    try:
        lr = LogReader(rlog_path, sort_by_time=True)
    except Exception as e:
        print(f"  WARNING: Could not open {rlog_path}: {e}")
        return rows

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
                rows.append(row)
    except Exception as e:
        print(f"  WARNING: Error processing {rlog_path}: {e}")

    return rows


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
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        sys.exit(1)

    rlog_files = find_rlogs(args.input)
    if not rlog_files:
        print(f"ERROR: No rlog files found in {args.input}")
        sys.exit(1)

    print(f"Found {len(rlog_files)} rlog files")

    all_rows = []
    for rlog_path in tqdm(rlog_files, desc="Processing rlogs"):
        rows = extract_segment(rlog_path)
        all_rows.extend(rows)

    if not all_rows:
        print("ERROR: No data extracted from any rlog files")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    print(f"Extracted {len(df)} rows")

    if args.filter_overrides and "steering_pressed" in df.columns:
        before = len(df)
        df = df[~df["steering_pressed"].astype(bool)]
        dropped = before - len(df)
        print(f"Filtered {dropped} override rows ({dropped / before:.1%} of data)")

    if args.temporal:
        print("Adding temporal columns...")
        df = add_temporal_columns(df)

    # Determine output format
    fmt = args.format
    if fmt is None:
        if args.output.endswith(".parquet"):
            fmt = "parquet"
        else:
            fmt = "csv"

    if fmt == "parquet":
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    print(f"Saved to {args.output} ({fmt})")


if __name__ == "__main__":
    main()
