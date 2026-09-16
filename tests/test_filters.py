from pathlib import Path

import numpy as np
import pytest

from nnlc_tools.steering_classifier.filters import (
    _make_bandpass_sos,
    _sosfilt,
    bandpass_rms,
    compute_freq_energy_ratio,
)


FILTERS_PATH = Path(__file__).resolve().parents[1] / "nnlc_tools" / "steering_classifier" / "filters.py"

# macOS/Windows libm can differ by ~1 ULP; keep this tighter than classifier thresholds.
GOLDEN_RTOL = 1e-12
GOLDEN_ATOL = 1e-12

# Captured from scipy 1.17.1: butter(2, Wn, btype="bandpass", output="sos")
# with fs=100 Hz and Nyquist-normalized Wn.
GOLDEN_LOW_SOS = np.array(
    [
        [
            0.005542717210280681,
            0.011085434420561362,
            0.005542717210280681,
            1.0,
            -1.806497093172675,
            0.8323065008979615,
        ],
        [
            1.0,
            -2.0,
            1.0,
            1.0,
            -1.960929076593623,
            0.9621487346328964,
        ],
    ]
)
GOLDEN_HIGH_SOS = np.array(
    [
        [
            0.5050010290458776,
            1.0100020580917553,
            0.5050010290458776,
            1.0,
            1.1256384845378178,
            0.4216723077213376,
        ],
        [
            1.0,
            -2.0,
            1.0,
            1.0,
            -1.5571209277367393,
            0.645560386443258,
        ],
    ]
)

GOLDEN_LOW_IMPULSE = np.array(
    [
        0.005542717210280681,
        0.020881777869625487,
        0.03800437237012927,
        0.05041551386996075,
        0.058592542856376474,
        0.0630427844893644,
        0.06428538798736155,
        0.06283590685007442,
        0.05919355829712098,
        0.053831041484069156,
        0.04718674821203566,
        0.03965916598129874,
        0.03160325021850555,
        0.023328529106566624,
        0.015098699398808788,
        0.007132473633940144,
        -0.00039455295434526194,
        -0.007347235002751279,
        -0.01362761213745595,
        -0.01917123109162309,
        -0.023943218996476678,
        -0.027934250913230832,
        -0.031156533412458953,
        -0.03363990516826449,
        -0.03542813558404952,
        -0.03657548377065728,
        -0.037143563028344796,
        -0.03719854052853422,
        -0.03680868826109947,
        -0.03604228955376164,
        -0.03496589557173464,
        -0.033642918111518594,
    ]
)


def test_filters_module_does_not_import_scipy():
    import ast

    source = FILTERS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("scipy") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("scipy")


def test_default_bandpass_sos_matches_scipy_golden():
    np.testing.assert_allclose(
        _make_bandpass_sos(0.5, 3.0, 100.0), GOLDEN_LOW_SOS, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL
    )
    np.testing.assert_allclose(
        _make_bandpass_sos(5.0, 40.0, 100.0), GOLDEN_HIGH_SOS, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL
    )


def test_zero_order_bandpass_matches_scipy_identity():
    expected = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(_make_bandpass_sos(0.5, 3.0, 100.0, order=0), expected)


def test_sosfilt_impulse_matches_scipy_golden():
    impulse = np.zeros(32)
    impulse[0] = 1.0
    filtered = _sosfilt(GOLDEN_LOW_SOS, impulse)
    np.testing.assert_allclose(filtered, GOLDEN_LOW_IMPULSE, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)


def test_compute_freq_energy_ratio_matches_golden_random_window():
    rng = np.random.default_rng(123)
    signal = rng.normal(size=80)
    ratio = compute_freq_energy_ratio(signal)
    np.testing.assert_allclose(ratio, 0.2267366923452386, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)


def test_bandpass_rms_short_signal_is_zero():
    assert bandpass_rms(np.ones(6), 0.5, 3.0) == 0.0


def test_freq_energy_ratio_caps_when_high_band_is_silent():
    assert compute_freq_energy_ratio(np.zeros(64), cap=10.0) == 10.0
    assert compute_freq_energy_ratio(np.full(64, 1e-12), cap=7.5) == 7.5


def test_freq_energy_ratio_live_scipy_random_windows():
    scipy = pytest.importorskip("scipy")
    butter = scipy.signal.butter
    sosfilt = scipy.signal.sosfilt
    rng = np.random.default_rng(7)

    def scipy_ratio(signal, fs=100.0, low_band=(0.5, 3.0), high_band=(5.0, 40.0), warmup=5, cap=10.0):
        def rms(band):
            nyq = fs / 2.0
            wn = [max(band[0] / nyq, 1e-4), min(band[1] / nyq, 1.0 - 1e-4)]
            sos = butter(2, wn, btype="bandpass", output="sos")
            filtered = sosfilt(sos, signal)
            return float(np.sqrt(np.mean(filtered[warmup:] ** 2)))

        high = rms(high_band)
        if high < 1e-6:
            return cap
        return float(rms(low_band) / high)

    for fs, low, high, n in [
        (100.0, (0.5, 3.0), (5.0, 40.0), 40),
        (100.0, (0.5, 3.0), (5.0, 40.0), 200),
        (50.0, (0.4, 2.0), (4.0, 20.0), 80),
        (200.0, (0.5, 3.0), (8.0, 60.0), 120),
    ]:
        for _ in range(8):
            signal = rng.normal(size=n)
            ours = compute_freq_energy_ratio(
                signal, fs=fs, low_band=low, high_band=high
            )
            ref = scipy_ratio(signal, fs=fs, low_band=low, high_band=high)
            np.testing.assert_allclose(ours, ref, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)


def test_sos_and_sosfilt_match_scipy_across_orders():
    scipy = pytest.importorskip("scipy")
    butter = scipy.signal.butter
    sosfilt = scipy.signal.sosfilt
    rng = np.random.default_rng(11)
    x = rng.normal(size=256)
    from nnlc_tools.steering_classifier.filters import _butter_bandpass_sos

    for order in (1, 2, 3, 4):
        wn = [0.5 / 50.0, 3.0 / 50.0]
        sos = _butter_bandpass_sos(order, wn)
        sos_ref = butter(order, wn, btype="bandpass", output="sos")
        np.testing.assert_allclose(sos, sos_ref, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)
        np.testing.assert_allclose(_sosfilt(sos, x), sosfilt(sos_ref, x), rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)


def test_real_rlog_windows_match_captured_scipy_ratios():
    fixture = Path(__file__).parent / "fixtures" / "steering_torque_windows.npz"
    data = np.load(fixture)
    ratios = data["ratios"]
    for i, expected in enumerate(ratios):
        torque = data[f"torque_{i}"]
        got = compute_freq_energy_ratio(torque)
        np.testing.assert_allclose(got, float(expected), rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL)
