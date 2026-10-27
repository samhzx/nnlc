"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import deque
import json
import math
import time
import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.sunnypilot.car.interfaces import LatControlInputs
from opendbc.sunnypilot.car.lateral_ext import get_friction as get_friction_in_torque_space
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase, sign
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]
FUTURE_TIMES = [0.3, 0.6, 1.0, 1.5]
PAST_TIMES_BY_DIM = {
  18: [-0.3, -0.2, -0.1],
  23: [-2.0, -1.0, -0.3, -0.2, -0.1],
}

# 唯一总开关：关闭时不向现有 NN/PID 输出叠加任何实验性辅助扭矩。
NNLC_AUX_ASSISTS_ENABLED = False
CURVE_ASSIST_GAIN = 0.20
CURVE_ASSIST_MAX = 0.06


class PersistentCurveUndertrackAssist:
  """仅对高速、大横向加速度且持续欠跟踪的弯道给出受限扭矩补偿。"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.last_time = None
    self.undertrack_since = None
    self.output = 0.0

  def update(self, now, speed, desired, actual, nn_error, planned_jerk, permitted, limited):
    if not permitted or not all(math.isfinite(value) for value in
                                (now, speed, desired, actual, nn_error, planned_jerk)):
      self.reset()
      return 0.0

    elapsed = None if self.last_time is None else now - self.last_time
    if elapsed is None or not 0.0 < elapsed <= 0.1:
      self.reset()
      dt = 0.01
    else:
      dt = min(elapsed, 0.02)
    self.last_time = now

    direction = sign(desired)
    deficit = direction * (desired - actual)
    tracking_persists = (
      speed > 10.0 and desired * actual > 0.0 and deficit > 0.05 and
      direction * nn_error > 0.0 and abs(desired) > 0.8 and
      direction * planned_jerk >= -0.05
    )

    target = 0.0
    if tracking_persists:
      if self.undertrack_since is None:
        self.undertrack_since = now
      if now - self.undertrack_since >= 0.30:
        speed_gate = float(np.interp(speed, [10.0, 12.0], [0.0, 1.0]))
        curve_gate = float(np.interp(abs(desired), [0.8, 1.2], [0.0, 1.0]))
        # 只有 NN 误差方向与目标横向加速度一致时才叠加，避免反向加扭矩。
        target = sign(nn_error) * min(CURVE_ASSIST_GAIN * (deficit - 0.05), CURVE_ASSIST_MAX) * speed_gate * curve_gate
    else:
      self.undertrack_since = None

    # 转弯方向刚反转时先卸掉旧方向补偿，再允许新方向补偿建立。
    if self.output * target < 0.0:
      target = 0.0
    # 安全限幅期间不继续增加辅助量，但允许已建立的补偿平顺退出。
    if limited and abs(target) > abs(self.output):
      target = self.output

    rate = 0.10 if abs(target) > abs(self.output) else 0.30
    self.output += float(np.clip(target - self.output, -rate * dt, rate * dt))
    return self.output


def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


def _temporal_suffix(offset):
  sign_char = "p" if offset >= 0 else "m"
  return f"t{sign_char}{abs(offset):.1f}".replace(".", "")


def _input_vars(input_dim, past_times):
  """按训练器顺序生成输入名：基础项、横向加速度时序、roll 时序。"""
  features = ["v_ego", "desired_lateral_accel", "friction_input", "roll"]
  if input_dim == 23:
    features.append("a_ego")
  offsets = past_times + FUTURE_TIMES
  return features + ["desired_lateral_accel_" + _temporal_suffix(t) for t in offsets] + \
         ["roll_" + _temporal_suffix(t) for t in offsets]


def _validate_model_input_vars(input_dim, model_input_vars):
  """返回契约错误；契约要求维度、字段数量和字段顺序全部一致。"""
  if input_dim not in PAST_TIMES_BY_DIM:
    return f"unsupported model input_size {input_dim}, expected 18 or 23"
  if not isinstance(model_input_vars, list) or not all(isinstance(name, str) for name in model_input_vars):
    return "model input_vars is missing or invalid"

  # 即使维度相同，只要某个字段错位也必须拒绝加载，否则模型会静默用错特征。
  expected = _input_vars(input_dim, PAST_TIMES_BY_DIM[input_dim])
  if model_input_vars != expected:
    mismatch = next(
      (index for index, (expected_name, actual_name) in enumerate(zip(expected, model_input_vars))
       if expected_name != actual_name),
      min(len(expected), len(model_input_vars)),
    )
    expected_name = expected[mismatch] if mismatch < len(expected) else "<missing>"
    actual_name = model_input_vars[mismatch] if mismatch < len(model_input_vars) else "<missing>"
    return (
      f"model input_vars mismatch for input_size {input_dim} at index {mismatch}: "
      f"expected {expected_name}, got {actual_name}"
    )
  return None


class NeuralNetworkLateralControl(LatControlTorqueExtBase):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    super().__init__(lac_torque, CP, CP_SP, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    model_path = CP_SP.neuralNetworkLateralControl.model.path
    self.has_nn_model = bool(model_path) and model_path != MOCK_MODEL_PATH
    self.model = NNTorqueModel(model_path) if self.has_nn_model else None

    # input_size 决定选择哪套输入契约；真正的字段顺序还要与模型 JSON 中
    # 的 input_vars 核对，不能只因长度相同就认为模型兼容。
    try:
      self.input_size = int(self.model.input_size) if self.model is not None else 0
    except (AttributeError, TypeError, ValueError):
      self.input_size = 0

    contract_error = None
    if self.model is not None:
      if self.input_size not in PAST_TIMES_BY_DIM:
        contract_error = _validate_model_input_vars(self.input_size, None)
      else:
        try:
          # NNTorqueModel 提供维度和权重，但不暴露 input_vars，因此从同一模型文件
          # 读取字段元数据；这里只在控制器初始化时读取，不进入每帧控制循环。
          with open(model_path, encoding="utf-8") as model_file:
            model_metadata = json.load(model_file)
          model_input_vars = model_metadata.get("input_vars") if isinstance(model_metadata, dict) else None
          contract_error = _validate_model_input_vars(self.input_size, model_input_vars)
        except (OSError, ValueError) as exc:
          contract_error = f"failed to read model input_vars: {exc}"

    self.nn_contract_ok = self.model is not None and contract_error is None
    if self.nn_contract_ok:
      self.past_times = list(PAST_TIMES_BY_DIM[self.input_size])
      self.input_vars = _input_vars(self.input_size, self.past_times)
    else:
      self.past_times = list(PAST_TIMES_BY_DIM[18])
      self.input_vars = []
      if self.model is not None:
        # 不支持的维度或特征契约错误时关闭 NN 前馈，并记录原因；外层仍走
        # 已有 PID/扭矩空间控制路径，避免把错位特征送进模型。
        cloudlog.error({"nnlc event": f"{contract_error}; NNLC disabled, falling back to PID"})

    self.curve_undertrack_assist = PersistentCurveUndertrackAssist()
    self._curve_assist_permitted = False

    self.use_a_ego = self.input_size == 23 and self.nn_contract_ok
    self.expected_input_dim = len(self.input_vars)
    self._dim_mismatch_logged = False

    self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
    self.pitch_last = 0.0
    self.future_times = list(FUTURE_TIMES)
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]

    # 追加当前帧后，N 帧队列最老只能到 t-(N-1)；多留一帧才能准确取到
    # 训练定义的 t-0.3s / t-2.0s 等边界历史点。
    history_check_frames = [int(abs(t) * 100) for t in self.past_times]
    history_max_frames = max(history_check_frames)
    self.history_frame_offsets = [history_max_frames - frames for frames in history_check_frames]
    history_size = history_max_frames + 1
    self.lateral_accel_desired_deque = deque(maxlen=history_size)
    self.roll_deque = deque(maxlen=history_size)
    self.past_future_len = len(self.past_times) + len(self.nn_future_times)

  @property
  def _nnlc_enabled(self):
    return self.enabled and self.has_nn_model and self.nn_contract_ok and self.model_valid

  def update_limits(self):
    if not self._nnlc_enabled:
      return
    self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]

  def update_feedforward_torque_space(self, CS):
    torque_from_setpoint = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(self._setpoint, self._roll_compensation, CS.vEgo, CS.aEgo),
      self.torque_params, gravity_adjusted=False,
    )
    torque_from_measurement = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(self._measurement, self._roll_compensation, CS.vEgo, CS.aEgo),
      self.torque_params, gravity_adjusted=False,
    )
    self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)
    self._ff = self.torque_from_lateral_accel_in_torque_space(
      LatControlInputs(self._gravity_adjusted_lateral_accel, self._roll_compensation, CS.vEgo, CS.aEgo),
      self.torque_params, gravity_adjusted=True,
    )
    self._ff += get_friction_in_torque_space(
      self._desired_lateral_accel - self._actual_lateral_accel,
      self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params,
    )

  def update_output_torque(self, CS):
    freeze_integrator = self._steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
    kf = getattr(self._pid, "k_f", 1.0)
    valid_kf = math.isfinite(kf) and kf > 1e-3
    # k_f 会再次缩放 feedforward，因此先把辅助扭矩换回 feedforward 单位。
    extra_torque = self.curve_undertrack_assist.update(
      time.monotonic(), CS.vEgo, self._desired_lateral_accel, self._actual_lateral_accel,
      self._pid_log.error, self.lookahead_lateral_jerk,
      NNLC_AUX_ASSISTS_ENABLED and self._curve_assist_permitted and not CS.steeringPressed and valid_kf,
      limited=self._steer_limited_by_safety,
    )
    feedforward = self._ff + extra_torque / kf if extra_torque else self._ff
    self._output_torque = self._pid.update(
      self._pid_log.error,
      feedforward=feedforward,
      speed=CS.vEgo,
      freeze_integrator=freeze_integrator,
    )

  def _base_features(self, CS, desired, third, roll):
    features = [CS.vEgo, desired, third, roll]
    if self.use_a_ego:
      features.append(CS.aEgo)
    return features

  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    self._curve_assist_permitted = False
    if not self._nnlc_enabled:
      self.curve_undertrack_assist.reset()
      return

    self.update_feedforward_torque_space(CS)
    pid_error, pid_feedforward = self._pid_log.error, self._ff

    low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
    self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature

    roll = params.roll
    if calibrated_pose is not None:
      pitch = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, pitch)
      self.pitch_last = pitch
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

    adjusted_future_times = [
      t + 0.5 * CS.aEgo * (t / max(CS.vEgo, 1.0)) for t in self.nn_future_times
    ]
    past_rolls = [self.roll_deque[min(len(self.roll_deque) - 1, i)] for i in self.history_frame_offsets]
    future_rolls = [
      roll_pitch_adjust(
        np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + roll,
        np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + self.pitch_last,
      ) for t in adjusted_future_times
    ]
    past_lateral_accels_desired = [
      self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque) - 1, i)]
      for i in self.history_frame_offsets
    ]
    future_planned_lateral_accels = [
      np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y)
      for t in adjusted_future_times
    ]

    friction_input = self.update_friction_input(self._setpoint, self._measurement)
    nn_input = self._base_features(CS, self._desired_lateral_accel, friction_input, roll) + \
               past_lateral_accels_desired + future_planned_lateral_accels + past_rolls + future_rolls
    if len(nn_input) != self.expected_input_dim:
      if not self._dim_mismatch_logged:
        cloudlog.error({"nnlc event": f"input dim mismatch built={len(nn_input)} "
                                      f"expected={self.expected_input_dim}; falling back to PID"})
        self._dim_mismatch_logged = True
      self._pid_log.error, self._ff = pid_error, pid_feedforward
      self.update_output_torque(CS)
      return

    nnff_setpoint_input = self._base_features(CS, self._setpoint, self.lateral_jerk_setpoint, roll) + \
                          [self._setpoint] * self.past_future_len + past_rolls + future_rolls
    nnff_measurement_input = self._base_features(CS, self._measurement, self.lateral_jerk_measurement, roll) + \
                             [self._measurement] * self.past_future_len + past_rolls + future_rolls
    torque_from_setpoint = self.model.evaluate(nnff_setpoint_input)
    torque_from_measurement = self.model.evaluate(nnff_measurement_input)
    self._pid_log.error = torque_from_setpoint - torque_from_measurement

    error_blend_factor = float(np.interp(abs(self._desired_lateral_accel), [1.0, 2.0], [0.0, 1.0]))
    if error_blend_factor > 0.0:
      # 误差混合固定 4 个数；23d 的 a_ego 等额外输入不走这条路径，由模型侧补 0
      nnff_error_input = [
        CS.vEgo, self._setpoint - self._measurement,
        self.lateral_jerk_setpoint - self.lateral_jerk_measurement, 0.0,
      ]
      torque_from_error = self.model.evaluate(nnff_error_input)
      if sign(self._pid_log.error) == sign(torque_from_error) and abs(self._pid_log.error) < abs(torque_from_error):
        self._pid_log.error = self._pid_log.error * (1.0 - error_blend_factor) + \
                              torque_from_error * error_blend_factor

    self._ff = self.model.evaluate(nn_input)
    if self.model.friction_override:
      self._pid_log.error += get_friction(
        friction_input, self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params,
      )
    self._curve_assist_permitted = True
    self.update_output_torque(CS)
