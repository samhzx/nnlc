from types import SimpleNamespace

import numpy as np
import pytest

from nnlc_tools import extract_lateral_data as extractor


class FakeEvent:
    def __init__(self, timestamp_ns, msg_type, **fields):
        self.logMonoTime = timestamp_ns
        self._msg_type = msg_type
        self.__dict__.update(fields)

    def which(self):
        return self._msg_type


class FakeLogReader:
    events = []
    skip_copy_calls = []

    def __init__(self, _path):
        self.closed = False

    def iter_events(self, *, skip_copy=False):
        type(self).skip_copy_calls.append(skip_copy)
        return iter(type(self).events)

    def close(self):
        self.closed = True


class LateralControlState:
    def __init__(self, torque_state):
        self.torqueState = torque_state

    def which(self):
        return "torqueState"


def _car_state(speed):
    return SimpleNamespace(
        vEgo=speed,
        aEgo=0.1,
        steeringAngleDeg=1.2,
        steeringRateDeg=0.3,
        steeringTorque=0.4,
        steeringPressed=False,
        standstill=False,
    )


def _controls_state():
    torque_state = SimpleNamespace(
        actualLateralAccel=0.2,
        desiredLateralAccel=0.3,
        output=-0.4,
        saturated=False,
    )
    return SimpleNamespace(
        lateralControlState=LateralControlState(torque_state),
        desiredCurvature=0.01,
        curvature=0.02,
        active=True,
    )


def _legacy_friction_input(v_ego, desired_lat_accel, actual_lat_accel,
                           desired_curvature, actual_curvature,
                           model_accels_y, steer_actuator_delay):
    if model_accels_y is None or len(model_accels_y) < len(extractor.T_IDXS):
        return 0.0
    try:
        scalar_inputs = np.asarray([
            v_ego, desired_lat_accel, actual_lat_accel,
            desired_curvature, actual_curvature, steer_actuator_delay,
        ], dtype=np.float64)
        lat_accels = np.asarray(model_accels_y[:len(extractor.T_IDXS)], dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if scalar_inputs.shape != (6,) or not np.all(np.isfinite(scalar_inputs)):
        return None
    if lat_accels.shape != (len(extractor.T_IDXS),) or not np.all(np.isfinite(lat_accels)):
        return None

    v_ego, desired, actual, desired_curve, actual_curve, delay = scalar_inputs
    low_speed_factor = float(np.interp(v_ego, extractor.LOW_SPEED_X, extractor.LOW_SPEED_Y)) ** 2
    setpoint = desired + low_speed_factor * desired_curve
    measurement = actual + low_speed_factor * actual_curve
    jerk_time = max(0.01, delay) + extractor.LATERAL_LAG_MOD
    lookahead = float(np.interp(
        v_ego, extractor.FRICTION_LOOK_AHEAD_BP, extractor.FRICTION_LOOK_AHEAD_V,
    ))
    upper_idx = next((i for i, t in enumerate(extractor.T_IDXS) if t > lookahead), 16)
    predicted = np.diff(lat_accels) / np.asarray(extractor.T_DIFFS)
    desired_jerk = (
        float(np.interp(jerk_time, extractor.T_IDXS, lat_accels)) - desired
    ) / jerk_time
    end_idx = min(upper_idx, len(predicted))
    jerk = extractor._get_lookahead_value(
        predicted[extractor.LAT_PLAN_MIN_IDX:end_idx].tolist(), desired_jerk,
    )
    accel_factor = 1.0 if jerk == 0.0 else extractor.LAT_ACCEL_FRICTION_FACTOR_DEFAULT
    result = accel_factor * (setpoint - measurement) + extractor.LAT_JERK_FRICTION_FACTOR * jerk
    return float(result) if np.isfinite(result) else None


@pytest.mark.parametrize("seed", [0, 7, 2026])
def test_friction_input_matches_legacy_formula(seed):
    rng = np.random.default_rng(seed)
    for _ in range(250):
        lat_accels = np.cumsum(rng.normal(0, 0.02, len(extractor.T_IDXS)))
        values = (
            rng.uniform(0, 45),
            rng.uniform(-4, 4),
            rng.uniform(-4, 4),
            rng.uniform(-0.2, 0.2),
            rng.uniform(-0.2, 0.2),
            lat_accels,
            rng.uniform(0.01, 1.5),
        )
        expected = _legacy_friction_input(*values)
        actual = extractor.calculate_friction_input(*values)
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_friction_input_matches_legacy_boundaries_and_invalid_inputs():
    lat_accels = np.linspace(-0.5, 0.5, len(extractor.T_IDXS))
    cases = [
        (speed, 0.2, -0.1, 0.01, -0.02, lat_accels, delay)
        for speed in (0.0, 9.0, 10.0, 20.0, 30.0, 45.0)
        for delay in (0.0, 0.01, 0.1, 1.5)
    ]
    cases.extend([
        (10.0, np.nan, 0.0, 0.0, 0.0, lat_accels, 0.1),
        (10.0, 0.1, 0.0, 0.0, 0.0, np.full(33, np.nan), 0.1),
        (10.0, 0.1, 0.0, 0.0, 0.0, [1.0, 2.0], 0.1),
    ])
    for values in cases:
        expected = _legacy_friction_input(*values)
        actual = extractor.calculate_friction_input(*values)
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_reorder_preserves_timestamp_and_source_order_for_ties():
    FakeLogReader.events = [
        FakeEvent(10, "liveParameters", liveParameters=SimpleNamespace(roll=1.0)),
        FakeEvent(5, "liveParameters", liveParameters=SimpleNamespace(roll=2.0)),
        FakeEvent(5, "liveParameters", liveParameters=SimpleNamespace(roll=3.0)),
    ]
    FakeLogReader.skip_copy_calls = []
    reader = FakeLogReader("unused")
    stats = {"max_lateness_seconds": 0.0}

    snapshots = list(extractor._iter_reordered_snapshots(reader, 100, stats))

    assert [(row[0], row[1], row[3]) for row in snapshots] == [
        (5, 1, 2.0), (5, 2, 3.0), (10, 0, 1.0),
    ]
    assert FakeLogReader.skip_copy_calls == [True]


def test_extract_segment_falls_back_without_emitting_partial_rows(monkeypatch):
    FakeLogReader.events = [
        FakeEvent(0, "carState", carState=_car_state(10.0)),
        FakeEvent(100_000_000, "controlsState", controlsState=_controls_state()),
        FakeEvent(3_000_000_000, "controlsState", controlsState=_controls_state()),
        FakeEvent(900_000_000, "carState", carState=_car_state(20.0)),
        FakeEvent(3_100_000_000, "controlsState", controlsState=_controls_state()),
    ]
    FakeLogReader.skip_copy_calls = []
    monkeypatch.setattr("nnlc_tools.logreader.LogReader", FakeLogReader)
    callback_rows = []
    stats = {}

    returned_rows = extractor.extract_segment(
        "synthetic/rlog.zst", stats_out=stats,
    )
    callback_result = extractor.extract_segment(
        "synthetic/rlog.zst", row_callback=callback_rows.append,
    )

    assert stats["used_exact_fallback"] is True
    assert [row[0] for row in returned_rows] == [0.1, 3.0, 3.1]
    assert [row[1] for row in returned_rows] == [10.0, 20.0, 20.0]
    assert callback_result == []
    for callback_row, returned_row in zip(callback_rows, returned_rows):
        for callback_value, returned_value in zip(callback_row, returned_row):
            if isinstance(callback_value, float) and np.isnan(callback_value):
                assert isinstance(returned_value, float) and np.isnan(returned_value)
            else:
                assert callback_value == returned_value
    assert FakeLogReader.skip_copy_calls == [True, False, True, False]


def test_two_second_lateness_does_not_trigger_fallback(monkeypatch):
    FakeLogReader.events = [
        FakeEvent(3_000_000_000, "liveParameters", liveParameters=SimpleNamespace(roll=1.0)),
        FakeEvent(1_000_000_000, "liveParameters", liveParameters=SimpleNamespace(roll=2.0)),
    ]
    FakeLogReader.skip_copy_calls = []
    monkeypatch.setattr("nnlc_tools.logreader.LogReader", FakeLogReader)
    stats = {}

    rows = extractor.extract_segment("synthetic/rlog.zst", stats_out=stats)

    assert rows == []
    assert stats["used_exact_fallback"] is False
    assert stats["max_lateness_seconds"] == 2.0
