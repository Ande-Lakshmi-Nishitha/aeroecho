#!/usr/bin/env python3
"""GCC-PHAT bearing estimation for AeroEcho (Member 2).

Conventions (CONTRACT.md sections 1 and 3):
  * channel 0 = LEFT mic, channel 1 = RIGHT mic, seen from behind the laptop
  * tau = t_left - t_right  (positive when the RIGHT mic hears the sound first)
  * bearing = degrees(arcsin(343 * (tau - OFFSET) / SPACING)); positive = right; 0 deg = +y
    (SPACING = 0.034 m, OFFSET = +20 us, measured 2026-11-01 with `phase1_check.py endfire`)
  * bearing_deg sent to the Core is clamped to [-60, +60] (team-confirmed 2026-11-01).
    Precision gets worse with angle (about 1/cos(theta)): ~6 deg per 10 us of timing jitter
    at 0 deg, ~12 deg at 60 deg.

Band limiting (300-4000 Hz) is applied in the frequency domain on the PHAT-weighted
spectrum. A time-domain band-pass before PHAT is not enough, because PHAT forces every
bin to unit magnitude and would re-amplify the out-of-band residue.

Status: implemented; tested ONLY on synthetic signals (run `python gcc_phat.py`).
NOT yet validated on real microphones.
"""
from __future__ import annotations

import numpy as np
import json
import os

from scipy.signal import butter, sosfiltfilt

C = 343.0                      # speed of sound, m/s (contract value)
MIC_SPACING_M = 0.034          # effective spacing, built-in AMD array (endfire test 2026-11-01)
OFFSET_US = 20.0               # fixed channel offset from the same test; NOT independently validated
BAND_HZ = (300.0, 4000.0)      # NOTE: the UPDATE doc says 300-3000; undecided, see team message
CLAMP_DEG = 60.0
CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "endfire.json")


def load_calibration(path=CAL_PATH):
    """Return (spacing_m, offset_us, source). Uses endfire.json written by
    `phase1_check.py endfire`; falls back to the measured defaults above (3.4 cm / +20 us)
    if it is missing. Re-run the endfire test if the mic setup or driver settings change."""
    try:
        with open(path) as f:
            c = json.load(f)
        return float(c["spacing_m"]), float(c["offset_us"]), "endfire.json"
    except (OSError, KeyError, ValueError):
        return MIC_SPACING_M, OFFSET_US, "built-in-defaults"


def bandpass(x, fs, band=BAND_HZ, order=4):
    """Zero-phase band-pass, for SNR estimation (not needed inside gcc_phat)."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def gcc_phat(left, right, fs, band=BAND_HZ, max_tau=MIC_SPACING_M / C, interp=True):
    """Return (tau_seconds, peak_ratio). tau = t_left - t_right.

    peak_ratio = correlation peak / median |correlation| inside the search window.
    A high value means a sharp, unambiguous peak. The threshold for "unreliable"
    (bearing_deg = null) is TBD and must be tuned on real recordings.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n = len(left)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))          # zero-pad: no circular wrap
    X0 = np.fft.rfft(left, nfft)
    X1 = np.fft.rfft(right, nfft)
    R = X0 * np.conj(X1)
    R /= np.abs(R) + 1e-12                            # PHAT weighting
    if band is not None:
        f = np.fft.rfftfreq(nfft, 1.0 / fs)
        R[(f < band[0]) | (f > band[1])] = 0.0
    cc = np.fft.irfft(R, nfft)

    max_shift = int(np.ceil(max_tau * fs)) + 1        # physical limit + 1 sample margin
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))   # lags -max..+max
    k = int(np.argmax(cc))

    delta = 0.0
    if interp and 0 < k < len(cc) - 1:                # parabolic sub-sample peak
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = 0.5 * (y0 - y2) / denom
    lag = (k - max_shift) + delta
    peak_ratio = float(cc[k] / (np.median(np.abs(cc)) + 1e-12))
    return lag / fs, peak_ratio


def tau_to_bearing(tau, d=MIC_SPACING_M):
    return float(np.degrees(np.arcsin(np.clip(C * tau / d, -1.0, 1.0))))


def estimate_bearing(stereo, fs, d=None, offset_us=None, min_peak_ratio=None, silence_rms=1e-6):
    """stereo: float array (n, 2). Returns a dict.

    d / offset_us default to the measured values in endfire.json (else 12 cm / 0 us).
    bearing is computed from (tau - offset). bearing_deg is clamped to +/-60 (contract) and
    is None if the input is silent or, when min_peak_ratio is given, the peak is weaker.
    tdoa_us is the RAW measured delay (what the calibration commands need);
    tdoa_corrected_us = tdoa_us - offset_us is what the bearing is based on.
    """
    cal_d, cal_off, cal_src = load_calibration()
    d = cal_d if d is None else d
    offset_us = cal_off if offset_us is None else offset_us
    stereo = np.asarray(stereo, dtype=np.float64)
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("stereo must have shape (n, 2)")
    left, right = stereo[:, 0], stereo[:, 1]
    if max(np.sqrt(np.mean(left ** 2)), np.sqrt(np.mean(right ** 2))) < silence_rms:
        return dict(bearing_deg=None, bearing_raw_deg=None, tdoa_us=None,
                    peak_ratio=0.0, reason="silent")
    tau, pr = gcc_phat(left, right, fs, max_tau=d / C + abs(offset_us) * 1e-6 + 20e-6)
    raw = tau_to_bearing(tau - offset_us * 1e-6, d)
    clamped = float(np.clip(raw, -CLAMP_DEG, CLAMP_DEG))
    reliable = (min_peak_ratio is None) or (pr >= min_peak_ratio)
    return dict(bearing_deg=clamped if reliable else None, bearing_raw_deg=raw,
                tdoa_us=tau * 1e6, tdoa_corrected_us=tau * 1e6 - offset_us, peak_ratio=pr,
                spacing_m=d, offset_us=offset_us, calibration=cal_src,
                reason="ok" if reliable else "weak_peak")


# ----------------------------- synthetic self-test -----------------------------
def _delay(x, tau, fs):
    """Fractional delay by tau seconds (positive = later) via FFT phase shift."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * tau), len(x))


def simulate_pair(bearing_deg, fs, snr_db, rng, dur=1.0, d=MIC_SPACING_M):
    """Source at bearing_deg (positive = right). Returns (n, 2) [left, right]."""
    n = int(dur * fs)
    s = bandpass(rng.standard_normal(n), fs)
    tau = d * np.sin(np.radians(bearing_deg)) / C      # t_left - t_right
    left, right = _delay(s, tau, fs), s.copy()         # right hears first when tau > 0
    if snr_db is not None:
        sig_pow = np.mean(s ** 2)
        for ch in (left, right):
            nz = bandpass(rng.standard_normal(n), fs)
            nz *= np.sqrt(sig_pow / 10 ** (snr_db / 10) / np.mean(nz ** 2))
            ch += nz
    return np.stack([left, right], axis=1)


def _selftest():
    rng = np.random.default_rng(0)
    print("SYNTHETIC self-test (simulated delay, ideal mics) - not a hardware result")
    print(f"d = {MIC_SPACING_M*100:.1f} cm, band = {BAND_HZ[0]:.0f}-{BAND_HZ[1]:.0f} Hz, "
          "in-band SNR, 30 trials per cell\n")
    print(f"{'fs':>6} {'SNR dB':>7} {'true deg':>9} {'mean est':>9} {'RMS err':>8}")
    worst = 0.0
    for fs in (16000, 48000):
        for snr in (None, 10, 0, -5):
            for true in (-60, -30, 0, 27, 45, 60):
                est = [estimate_bearing(simulate_pair(true, fs, snr, rng), fs, d=MIC_SPACING_M,
                                        offset_us=0.0)["bearing_deg"]
                       for _ in range(30)]
                est = np.array(est, dtype=float)
                rms = float(np.sqrt(np.mean((est - true) ** 2)))
                worst = max(worst, rms) if snr in (None, 10) else worst
                print(f"{fs:>6} {'inf' if snr is None else snr:>7} {true:>9} "
                      f"{est.mean():>9.1f} {rms:>8.2f}")
    # sign convention check
    r = estimate_bearing(simulate_pair(+27, 48000, None, rng), 48000, d=MIC_SPACING_M, offset_us=0.0)["bearing_deg"]
    l = estimate_bearing(simulate_pair(-27, 48000, None, rng), 48000, d=MIC_SPACING_M, offset_us=0.0)["bearing_deg"]
    print(f"\nSign check: source at +27 deg (right) -> {r:+.1f}; at -27 deg (left) -> {l:+.1f}")
    assert r > 0 > l, "sign convention broken"
    print("Sign convention OK (positive = right).")


if __name__ == "__main__":
    _selftest()
