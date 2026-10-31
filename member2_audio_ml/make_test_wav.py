"""Synthesise a 2-channel 48 kHz test recording of a 'victim' at a known bearing, as the calibrated
laptop array would hear it. For replay tests of aeroecho_sender.py; the output is SIMULATED audio.

    python make_test_wav.py                          # victim at +25 deg -> test_victim.wav
    python make_test_wav.py --bearing -40 --seconds 30 --out left.wav
    python make_test_wav.py --silent                 # room noise only (should give bearing = null)

Model: voice-like harmonic bursts (f0 ~ 120-180 Hz, ~4 s calling / 3 s quiet) arriving with
tau = d*sin(theta)/c (fractional delay), plus the measured channel offset (+20 us on the raw delay),
two weak wall echoes and uncorrelated mic noise.
"""
from __future__ import annotations

import argparse
import math
import wave

import numpy as np


def frac_delay(x: np.ndarray, delay_s: float, fs: int) -> np.ndarray:
    n = len(x)
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_s), nfft)[:n]


def voice_like(seconds: float, fs: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    f0 = 150 + 25 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    sig = np.zeros_like(t)
    for h in range(1, 25):
        fh = h * 150
        formant = sum(np.exp(-0.5 * ((fh - F) / bw) ** 2) for F, bw in ((700, 150), (1200, 200), (2600, 300)))
        sig += (0.2 + formant) / h ** 0.6 * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
    syll = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t - np.pi / 2)) ** 1.5  # ~3 syllables/s
    calling = ((t % 7.0) < 4.0).astype(float)  # calls 4 s out of every 7
    env = syll * calling
    env = np.convolve(env, np.ones(int(0.02 * fs)) / int(0.02 * fs), mode="same")
    return sig / np.max(np.abs(sig)) * env


def make(bearing_deg: float, seconds: float, fs: int = 48000, spacing_m: float = 0.034, offset_us: float = 20.0,
         snr_db: float = 20.0, silent: bool = False, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    src = np.zeros(n) if silent else voice_like(seconds, fs, rng)
    tau = spacing_m * math.sin(math.radians(bearing_deg)) / 343.0  # t_left - t_right
    left = frac_delay(src, max(tau, 0.0) + offset_us * 1e-6, fs)  # offset: left channel lags (raw tau +20 us)
    right = frac_delay(src, max(-tau, 0.0), fs)
    for gain, extra_s, b in ((0.35, 0.0061, bearing_deg + 70), (0.25, 0.0113, bearing_deg - 110)):  # wall echoes
        te = spacing_m * math.sin(math.radians(b)) / 343.0
        echo = frac_delay(src, extra_s, fs) * gain
        left += frac_delay(echo, max(te, 0.0) + offset_us * 1e-6, fs)
        right += frac_delay(echo, max(-te, 0.0), fs)
    sig_rms = np.sqrt(np.mean(src[np.abs(src) > 1e-3] ** 2)) if not silent else 0.05
    noise = sig_rms / (10 ** (snr_db / 20))
    x = np.stack([left, right], axis=1) + rng.normal(0, noise, (n, 2))
    return (x / max(1.0, np.max(np.abs(x)) * 1.05)).astype(np.float32)


def write_wav(path: str, x: np.ndarray, fs: int = 48000) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bearing", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=21.0)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--silent", action="store_true")
    ap.add_argument("--out", default="test_victim.wav")
    a = ap.parse_args()
    write_wav(a.out, make(a.bearing, a.seconds, snr_db=a.snr, silent=a.silent))
    print(f"wrote {a.out}: {a.seconds:.0f} s, bearing {a.bearing:+.0f} deg{' (silent)' if a.silent else ''} [SIMULATED]")
