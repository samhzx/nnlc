"""IIR bandpass filter utilities for frequency energy ratio computation.

Butterworth SOS coefficients and cascaded sosfilt are implemented in numpy so
the steering classifier does not depend on scipy at runtime. The design path
and Direct-Form II transposed recurrence follow scipy.signal (digital
Butterworth bandpass, pairing='nearest', compiled sosfilt with FMA).
"""

from __future__ import annotations

import math
import sys
from typing import Callable

import numpy as np


def _resolve_fma() -> Callable[[float, float, float], float]:
    """Return fused multiply-add so sosfilt can match scipy's compiled kernel."""
    if hasattr(math, "fma"):
        return math.fma

    import ctypes

    candidates: list[str | None]
    if sys.platform == "win32":
        candidates = ["ucrtbase", "msvcrt"]
    else:
        candidates = [None, "m", "libm.so.6", "libm.dylib"]

    for name in candidates:
        try:
            lib = ctypes.CDLL(name) if name is not None else ctypes.CDLL(None)
            func = lib.fma
            func.restype = ctypes.c_double
            func.argtypes = (
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
            )
            if func(1.0, 2.0, 3.0) == 5.0:
                return lambda x, y, z, _func=func: float(_func(x, y, z))
        except (AttributeError, OSError, TypeError, ValueError):
            continue

    return lambda x, y, z: (x * y) + z


_FMA = _resolve_fma()


def _relative_degree(z: np.ndarray, p: np.ndarray) -> int:
    degree = int(p.shape[0] - z.shape[0])
    if degree < 0:
        raise ValueError("Improper transfer function. Must have at least as many poles as zeros.")
    return degree


def _buttap(order: int) -> tuple[np.ndarray, np.ndarray, float]:
    if abs(int(order)) != order:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.asarray([], dtype=np.float64)
    m = np.arange(-order + 1, order, 2, dtype=np.float64)
    p = -np.exp(1j * np.pi * m / (2 * order))
    return z, p, 1.0


def _lp2bp_zpk(
    z: np.ndarray,
    p: np.ndarray,
    k: float,
    wo: float,
    bw: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.atleast_1d(np.asarray(z))
    p = np.atleast_1d(np.asarray(p))
    wo = float(wo)
    bw = float(bw)
    degree = _relative_degree(z, p)

    z_lp = np.asarray(z * bw / 2.0, dtype=np.complex128)
    p_lp = np.asarray(p * bw / 2.0, dtype=np.complex128)
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp**2 - wo**2), z_lp - np.sqrt(z_lp**2 - wo**2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp**2 - wo**2), p_lp - np.sqrt(p_lp**2 - wo**2)))
    z_bp = np.concatenate((z_bp, np.zeros(degree, dtype=np.complex128)))
    k_bp = k * bw**degree
    return z_bp, p_bp, k_bp


def _bilinear_zpk(
    z: np.ndarray,
    p: np.ndarray,
    k: float,
    fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.atleast_1d(np.asarray(z))
    p = np.atleast_1d(np.asarray(p))
    degree = _relative_degree(z, p)
    fs2 = 2.0 * float(fs)

    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)
    z_z = np.concatenate((z_z, -np.ones(degree, dtype=z_z.dtype)))
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))
    return z_z, p_z, float(k_z)


def _cplxreal(z: np.ndarray, tol: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    z = np.atleast_1d(z)
    if z.size == 0:
        return z, z
    if z.ndim != 1:
        raise ValueError("_cplxreal only accepts 1-D input")

    if tol is None:
        tol = 100 * np.finfo((1.0 * z).dtype).eps

    z = z[np.lexsort((abs(z.imag), z.real))]
    real_indices = abs(z.imag) <= tol * abs(z)
    zr = z[real_indices].real
    if len(zr) == len(z):
        return np.array([]), zr

    z = z[~real_indices]
    zp = z[z.imag > 0]
    zn = z[z.imag < 0]
    if len(zp) != len(zn):
        raise ValueError("Array contains complex value with no matching conjugate.")

    same_real = np.diff(zp.real) <= tol * abs(zp[:-1])
    diffs = np.diff(np.concatenate(([0], same_real, [0])))
    run_starts = np.nonzero(diffs > 0)[0]
    run_stops = np.nonzero(diffs < 0)[0]
    for i in range(len(run_starts)):
        start = run_starts[i]
        stop = run_stops[i] + 1
        for chunk in (zp[start:stop], zn[start:stop]):
            chunk[...] = chunk[np.lexsort([abs(chunk.imag)])]

    if np.any(abs(zp - zn.conj()) > tol * abs(zn)):
        raise ValueError("Array contains complex value with no matching conjugate.")

    zc = (zp + zn.conj()) / 2
    return zc, zr


def _poly_from_roots(roots) -> np.ndarray:
    """Match scipy.signal._polyutils.poly (1-D np.poly equivalent)."""
    roots = np.atleast_1d(np.asarray(roots))
    if roots.shape[0] == 0:
        return np.asarray([1.0], dtype=np.float64)

    dt = roots.dtype
    a = np.ones((1,), dtype=dt)
    one = np.ones_like(roots[0])
    for zero in roots:
        a = np.convolve(a, np.stack((one, -zero)), mode="full")

    if np.iscomplexobj(a):
        complex_roots = np.asarray(roots, dtype=np.complex128)
        if np.all(np.sort(np.imag(complex_roots)) == np.sort(np.imag(np.conj(complex_roots)))):
            a = np.array(np.real(a), copy=True)
    return a


def _zpk2tf(z, p, k: float) -> tuple[np.ndarray, np.ndarray]:
    b = np.atleast_1d(np.asarray(k) * _poly_from_roots(z))
    a = np.atleast_1d(_poly_from_roots(p))
    return b, a


def _nearest_real_complex_idx(fro: np.ndarray, to, which: str) -> int:
    order = np.argsort(np.abs(fro - to))
    if which == "any":
        return int(order[0])
    mask = np.isreal(fro[order])
    if which == "complex":
        mask = ~mask
    return int(order[np.nonzero(mask)[0][0]])


def _as_real_coeffs(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.atleast_1d(np.asarray(coeffs)).squeeze()
    coeffs = np.real_if_close(coeffs, tol=1000)
    if np.iscomplexobj(coeffs):
        coeffs = np.real(coeffs)
    return np.asarray(coeffs, dtype=np.float64)


def _single_zpksos(z, p, k: float) -> np.ndarray:
    sos = np.zeros(6, dtype=np.float64)
    b, a = _zpk2tf(z, p, k)
    b = _as_real_coeffs(b)
    a = _as_real_coeffs(a)
    sos[3 - len(b):3] = b
    sos[6 - len(a):6] = a
    return sos


def _zpk2sos(z, p, k: float) -> np.ndarray:
    z = np.asarray(z)
    p = np.asarray(p)
    k = np.asarray(k)

    if len(z) == len(p) == 0:
        return np.asarray([[float(np.real(k)), 0.0, 0.0, 1.0, 0.0, 0.0]])

    p = np.concatenate((p, np.zeros(max(len(z) - len(p), 0))))
    z = np.concatenate((z, np.zeros(max(len(p) - len(z), 0))))
    n_sections = (max(len(p), len(z)) + 1) // 2
    if len(p) % 2 == 1:
        p = np.concatenate((p, [0.0]))
        z = np.concatenate((z, [0.0]))

    z = np.concatenate(_cplxreal(z))
    p = np.concatenate(_cplxreal(p))
    if not np.isreal(k):
        raise ValueError("k must be real")
    k = float(np.real(k))

    def idx_worst(poles: np.ndarray) -> int:
        return int(np.argmin(np.abs(1 - np.abs(poles))))

    sos = np.zeros((n_sections, 6), dtype=np.float64)
    for si in range(n_sections - 1, -1, -1):
        p1_idx = idx_worst(p)
        p1 = p[p1_idx]
        p = np.delete(p, p1_idx)

        if np.isreal(p1) and np.isreal(p).sum() == 0:
            z1_idx = _nearest_real_complex_idx(z, p1, "real")
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)
            sos[si] = _single_zpksos([z1, 0], [p1, 0], 1)
        elif (
            len(p) + 1 == len(z)
            and not np.isreal(p1)
            and np.isreal(p).sum() == 1
            and np.isreal(z).sum() == 1
        ):
            z1_idx = _nearest_real_complex_idx(z, p1, "complex")
            z1 = z[z1_idx]
            z = np.delete(z, z1_idx)
            sos[si] = _single_zpksos([z1, z1.conj()], [p1, p1.conj()], 1)
        else:
            if np.isreal(p1):
                prealidx = np.flatnonzero(np.isreal(p))
                p2_idx = prealidx[idx_worst(p[prealidx])]
                p2 = p[p2_idx]
                p = np.delete(p, p2_idx)
            else:
                p2 = p1.conj()

            if len(z) > 0:
                z1_idx = _nearest_real_complex_idx(z, p1, "any")
                z1 = z[z1_idx]
                z = np.delete(z, z1_idx)
                if not np.isreal(z1):
                    sos[si] = _single_zpksos([z1, z1.conj()], [p1, p2], 1)
                elif len(z) > 0:
                    z2_idx = _nearest_real_complex_idx(z, p1, "real")
                    z2 = z[z2_idx]
                    z = np.delete(z, z2_idx)
                    sos[si] = _single_zpksos([z1, z2], [p1, p2], 1)
                else:
                    sos[si] = _single_zpksos([z1], [p1, p2], 1)
            else:
                sos[si] = _single_zpksos([], [p1, p2], 1)

    if len(p) != 0 or len(z) != 0:
        raise RuntimeError("Unpaired poles or zeros while building SOS")
    sos[0, :3] *= k
    return sos


def _butter_bandpass_sos(order: int, wn) -> np.ndarray:
    """Digital Butterworth bandpass SOS with Nyquist-normalized Wn (fs=None)."""
    wn = np.asarray(wn, dtype=np.float64)
    if wn.size != 2:
        raise ValueError("Wn must specify start and stop frequencies for bandpass filter")
    if not wn[0] < wn[1]:
        raise ValueError("Wn[0] must be less than Wn[1]")
    if np.any(wn <= 0) or np.any(wn >= 1):
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")

    z, p, k = _buttap(order)
    fs = 2.0
    warped = 2.0 * fs * np.tan(np.pi * wn / fs)
    bw = warped[1] - warped[0]
    wo = np.sqrt(warped[0] * warped[1])
    z, p, k = _lp2bp_zpk(z, p, k, wo=wo, bw=bw)
    z, p, k = _bilinear_zpk(z, p, k, fs=fs)
    return _zpk2sos(z, p, k)


def _sosfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """1-D Direct-Form II transposed sosfilt, matching scipy's compiled kernel."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    sos = np.atleast_2d(np.asarray(sos, dtype=np.float64))
    if sos.ndim != 2 or sos.shape[1] != 6:
        raise ValueError("sos must be shape (n_sections, 6)")

    n_sections = sos.shape[0]
    sections = np.array(sos, dtype=np.float64, copy=True)
    for s in range(n_sections):
        a0 = sections[s, 3]
        if a0 != 1.0:
            sections[s, :3] /= a0
            sections[s, 4] /= a0
            sections[s, 5] /= a0
            sections[s, 3] = 1.0

    y = np.array(x, dtype=np.float64, copy=True)
    zi0 = [0.0] * n_sections
    zi1 = [0.0] * n_sections
    coeffs = [
        (float(row[0]), float(row[1]), float(row[2]), float(row[4]), float(row[5]))
        for row in sections
    ]
    n_samples = y.shape[0]
    for i in range(n_samples):
        x_n = float(y[i])
        for s, (b0, b1, b2, a1, a2) in enumerate(coeffs):
            y_n = _FMA(b0, x_n, zi0[s])
            zi0[s] = _FMA(b1, x_n, -a1 * y_n) + zi1[s]
            zi1[s] = _FMA(b2, x_n, -a2 * y_n)
            x_n = y_n
        y[i] = x_n
    return y


def _make_bandpass_sos(low_hz: float, high_hz: float, fs: float, order: int = 2):
    """Build a Butterworth bandpass SOS filter."""
    nyq = fs / 2.0
    low = max(low_hz / nyq, 1e-4)
    high = min(high_hz / nyq, 1.0 - 1e-4)
    return _butter_bandpass_sos(order, [low, high])


def bandpass_rms(signal: np.ndarray, low_hz: float, high_hz: float, fs: float = 100.0,
                 warmup_samples: int = 5) -> float:
    """Apply bandpass filter and return RMS energy of the filtered signal.

    Discards the first `warmup_samples` to reduce IIR edge-effect bias.
    Returns 0.0 if the signal is too short to filter.
    """
    if len(signal) < warmup_samples + 2:
        return 0.0
    sos = _make_bandpass_sos(low_hz, high_hz, fs)
    filtered = _sosfilt(sos, signal)
    trimmed = filtered[warmup_samples:]
    if len(trimmed) == 0:
        return 0.0
    return float(np.sqrt(np.mean(trimmed ** 2)))


def compute_freq_energy_ratio(
    signal: np.ndarray,
    fs: float = 100.0,
    low_band: tuple = (0.5, 3.0),
    high_band: tuple = (5.0, 40.0),
    warmup_samples: int = 5,
    cap: float = 10.0,
) -> float:
    """Return low_band_rms / high_band_rms.

    - Driver inputs have most energy < 3 Hz → ratio >> 1
    - Road impacts have energy concentrated 5–40 Hz → ratio ≈ 1 or < 1
    - Returns `cap` when high-band energy is negligible (essentially all driver content)
    """
    low_rms = bandpass_rms(signal, low_band[0], low_band[1], fs, warmup_samples)
    high_rms = bandpass_rms(signal, high_band[0], high_band[1], fs, warmup_samples)
    if high_rms < 1e-6:
        return cap
    return float(low_rms / high_rms)
