#!/usr/bin/env python3
"""AeroEcho audio sender: laptop mic array -> GCC-PHAT bearing + classifier -> PerceptionRecord -> Core.

Runs on the PROBE laptop (the one whose built-in mic array was calibrated). Contract:
contracts/CONTRACT.md section 3 and contracts/perception_schema.json.

    pip install numpy sounddevice            (+ tensorflow tensorflow-hub  for the YAMNet classifier)

    python aeroecho_sender.py --core http://<core-ip>:8000                  # live, YAMNet classifier
    python aeroecho_sender.py --core http://<core-ip>:8000 --classifier energy   # live, no TensorFlow (heuristic)
    python aeroecho_sender.py --replay test_victim.wav --classifier energy  # REPLAY a 2-channel WAV (flagged)
    python aeroecho_sender.py --dry-run                                     # print records, send nothing
    python aeroecho_sender.py --list-devices

Per 1.0 s window (0.5 s hop -> 2 records/s):
  * snr_db   : 300-4000 Hz band power over a rolling noise floor (10th percentile of the last 10 s)
  * quality  : listening_quality = clip((snr_db + 10) / 30, 0, 1)
  * bearing  : GCC-PHAT (frame-averaged cross-spectrum, PHAT weighting, 300-4000 Hz), search limited to
               the physical lag range, 8x interpolation + parabolic peak.
               tau_us = (t_left - t_right) * 1e6 - OFFSET_US ; theta = asin(343 * tau / d), clamp +/-60.
               Sent as null unless a target sound is detected, SNR >= 6 dB and the peak is clear.
  * classes  : Voice / Cough / Breathing / Machinery probabilities; detect_score = P(V)+P(C)+P(B).
Calibration (spacing, offset) is read from endfire.json or calibration.json if present.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

SENSOR_ID = "laptop_earphone_mics"
CLASSES = ("Voice", "Cough", "Breathing", "Machinery")
HUMAN = ("Voice", "Cough", "Breathing")
C_SOUND = 343.0


# ============================================================== configuration
@dataclass
class SenderConfig:
    core: str = "http://127.0.0.1:8000"
    fs: int = 48000
    window_s: float = 1.0
    hop_s: float = 0.5
    spacing_m: float = 0.034  # measured: built-in AMD array (endfire test)
    offset_us: float = 20.0  # measured channel offset, subtracted from the raw delay
    clamp_deg: float = 60.0  # contract: bearings clamped to +/-60
    band_hz: tuple[float, float] = (300.0, 4000.0)
    frame: int = 4096  # GCC-PHAT sub-frame (85 ms at 48 kHz)
    interp: int = 8
    min_peak_ratio: float = 2.5
    min_bearing_snr_db: float = 6.0
    min_detect_for_bearing: float = 0.5
    noise_floor_window_s: float = 10.0
    noise_floor_pct: float = 10.0
    swap_channels: bool = False
    mic_channels: int = 2
    replay: bool = False
    calib_source: str = "defaults"

    def load_calibration(self, path: Optional[str]) -> None:
        candidates = [path] if path else ["calibration.json", "endfire.json"]
        for p in candidates:
            if p and os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    c = json.load(f)
                d = float(c.get("spacing_m", self.spacing_m))
                if not 0.02 <= abs(d) <= 0.20:
                    raise SystemExit(f"{p}: spacing_m={d} is implausible (2-20 cm). Re-run the endfire test.")
                if d < 0:  # calibrate reports negative spacing when channels are swapped
                    self.swap_channels, d = True, -d
                self.spacing_m = d
                self.offset_us = float(c.get("offset_us", self.offset_us))
                self.calib_source = p
                return


# ============================================================== DSP
def band_power(x: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    """Mean power of a mono signal inside `band` (Hz)."""
    X = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= band[0]) & (f <= band[1])
    return float(np.mean(np.abs(X[m]) ** 2)) + 1e-20


def gcc_phat(left: np.ndarray, right: np.ndarray, fs: int, max_tau_s: float, band: tuple[float, float],
             frame: int = 4096, interp: int = 8) -> tuple[float, float]:
    """Delay tau = t_left - t_right in seconds (positive: RIGHT mic hears it first) and a peak ratio.

    Cross-spectra of Hann-windowed sub-frames are summed (louder frames weigh more), PHAT-normalised,
    band-limited, then inverse-transformed with `interp`x zero padding. The peak is searched only
    inside +/-max_tau_s and refined with a parabola. Peak ratio = peak / RMS of the full correlation.
    """
    n = len(left)
    frame = min(frame, n)
    hop = frame // 2
    win = np.hanning(frame)
    nfft = 1 << (2 * frame - 1).bit_length()
    G = np.zeros(nfft // 2 + 1, dtype=complex)
    for s in range(0, n - frame + 1, hop):
        XL = np.fft.rfft(left[s:s + frame] * win, nfft)
        XR = np.fft.rfft(right[s:s + frame] * win, nfft)
        G += XL * np.conj(XR)
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    G = G / (np.abs(G) + 1e-12)
    G[(f < band[0]) | (f > band[1])] = 0.0
    cc = np.fft.irfft(G, nfft * interp)
    cc = np.concatenate((cc[-(nfft * interp) // 2:], cc[:(nfft * interp) // 2]))  # lag 0 at centre
    centre = len(cc) // 2
    max_shift = max(1, int(math.ceil(max_tau_s * fs * interp)))
    seg = cc[centre - max_shift: centre + max_shift + 1]
    k = int(np.argmax(seg))
    delta = 0.0
    if 0 < k < len(seg) - 1:
        a, b, c = seg[k - 1], seg[k], seg[k + 1]
        den = a - 2 * b + c
        if den != 0:
            delta = 0.5 * (a - c) / den
    tau = (k - max_shift + delta) / (fs * interp)
    rms = float(np.sqrt(np.mean(cc ** 2))) + 1e-12
    return float(tau), float(seg[k] / rms)


def bearing_from_tau(tau_s: float, spacing_m: float, offset_us: float, clamp_deg: float) -> tuple[float, float]:
    """(bearing_deg, corrected tau in us). Contract sign: positive = right."""
    tau_us = tau_s * 1e6 - offset_us
    s = max(-1.0, min(1.0, C_SOUND * tau_us * 1e-6 / spacing_m))
    theta = math.degrees(math.asin(s))
    return max(-clamp_deg, min(clamp_deg, theta)), tau_us


def resample_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Integer decimation by 3 with a windowed-sinc low-pass (numpy only)."""
    taps = 95
    fc = 7000.0 / 48000.0  # cut-off below the new Nyquist (8 kHz)
    n = np.arange(taps) - (taps - 1) / 2
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
    h /= h.sum()
    return np.convolve(x, h, mode="same")[::3].astype(np.float32)


class NoiseFloor:
    """Rolling low percentile of window band powers: the 'quiet' level of the room."""

    def __init__(self, n_windows: int, pct: float) -> None:
        self.hist: deque[float] = deque(maxlen=max(4, n_windows))
        self.pct = pct

    def update(self, p: float) -> float:
        self.hist.append(p)
        return float(np.percentile(self.hist, self.pct))


# ============================================================== classifiers
class Classifier:
    name = "base"

    def predict(self, wave16k: np.ndarray) -> dict[str, float]:  # probabilities over CLASSES
        raise NotImplementedError


class EnergyClassifier(Classifier):
    """[HEURISTIC] No ML. 'Voice' rises with SNR in the speech band. Only for pipeline tests."""
    name = "energy-heuristic"

    def __init__(self) -> None:
        self.snr_db = 0.0

    def predict(self, wave16k: np.ndarray) -> dict[str, float]:
        p_voice = 1.0 / (1.0 + math.exp(-(self.snr_db - 8.0) / 2.5))
        return {"Voice": round(0.9 * p_voice, 4), "Cough": round(0.05 * p_voice, 4),
                "Breathing": round(0.05 * p_voice, 4), "Machinery": round(1.0 - p_voice, 4)}


def make_classifier(kind: str, **kw) -> Classifier:
    if kind == "energy":
        return EnergyClassifier()
    if kind == "yamnet":
        from yamnet_baseline import YamnetClassifier
        return YamnetClassifier(**kw)
    raise SystemExit(f"unknown classifier {kind!r}")


# ============================================================== window processing
@dataclass
class WindowResult:
    record: dict[str, Any]
    peak_ratio: float
    raw_bearing: Optional[float]


class Processor:
    """Turns one stereo window (n, 2) at cfg.fs into a contract record."""

    def __init__(self, cfg: SenderConfig, classifier: Classifier,
                 bearing_fn: Optional[Callable[[np.ndarray, int], dict]] = None) -> None:
        self.cfg = cfg
        self.clf = classifier
        self.bearing_fn = bearing_fn  # optional Member 2 estimate_bearing(seg, fs) -> dict
        n_win = int(cfg.noise_floor_window_s / cfg.hop_s)
        self.floor = NoiseFloor(n_win, cfg.noise_floor_pct)

    def process(self, x: np.ndarray, timestamp: float) -> WindowResult:
        cfg = self.cfg
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1 or x.shape[1] == 1 or cfg.mic_channels == 1:
            left = right = x if x.ndim == 1 else x[:, 0]
            stereo = False
        else:
            left, right = (x[:, 1], x[:, 0]) if cfg.swap_channels else (x[:, 0], x[:, 1])
            stereo = True
        mono = 0.5 * (left + right)

        p = band_power(mono, cfg.fs, cfg.band_hz)
        floor = self.floor.update(p)
        snr_db = 10.0 * math.log10(p / floor) if floor > 0 else 0.0
        quality = max(0.0, min(1.0, (snr_db + 10.0) / 30.0))

        if isinstance(self.clf, EnergyClassifier):
            self.clf.snr_db = snr_db
        probs = self.clf.predict(resample_48k_to_16k(mono))
        probs = {c: float(probs.get(c, 0.0)) for c in CLASSES}
        detect = max(0.0, min(1.0, sum(probs[c] for c in HUMAN)))
        primary = max(CLASSES, key=lambda c: probs[c])

        bearing = tdoa_us = None
        peak_ratio, raw = 0.0, None
        if stereo:
            if self.bearing_fn is not None:  # Member 2's implementation (must already apply the offset)
                r = self.bearing_fn(np.stack([left, right], axis=1), cfg.fs)
                raw, tdoa_us, peak_ratio = r.get("bearing_raw_deg"), r.get("tdoa_us"), float(r.get("peak_ratio") or 0.0)
                if raw is not None:
                    raw = max(-cfg.clamp_deg, min(cfg.clamp_deg, float(raw)))
            else:
                max_tau = cfg.spacing_m / C_SOUND + abs(cfg.offset_us) * 1e-6 + 20e-6
                tau, peak_ratio = gcc_phat(left, right, cfg.fs, max_tau, cfg.band_hz, cfg.frame, cfg.interp)
                raw, tdoa_us = bearing_from_tau(tau, cfg.spacing_m, cfg.offset_us, cfg.clamp_deg)
            reliable = (raw is not None and detect >= cfg.min_detect_for_bearing
                        and snr_db >= cfg.min_bearing_snr_db and peak_ratio >= cfg.min_peak_ratio)
            bearing = round(raw, 1) if reliable else None

        rec: dict[str, Any] = {
            "schema_version": 1,
            "sensor_id": SENSOR_ID,
            "timestamp": float(timestamp),
            "detect_score": round(detect, 4),
            "bearing_deg": bearing,
            "listening_quality": round(quality, 4),
            "primary_class": primary,
            "class_probs": {c: round(v, 4) for c, v in probs.items()},
            "snr_db": round(snr_db, 2),
            "mic_channels": 2 if stereo else 1,
        }
        if tdoa_us is not None:
            rec["tdoa_us"] = round(float(tdoa_us), 1)  # offset-corrected
        rec["peak_ratio"] = round(peak_ratio, 2)  # extra keys are allowed by the contract
        rec["classifier"] = self.clf.name
        if cfg.replay:
            rec["replay"] = True
        return WindowResult(rec, peak_ratio, raw)


# ============================================================== transport
class Poster:
    """Background HTTP sender: never blocks capture; bounded queue drops the oldest record."""

    def __init__(self, core: str, maxsize: int = 20, send: Optional[Callable[[dict], dict]] = None) -> None:
        self.url = core.rstrip("/") + "/ingest"
        self.q: queue.Queue = queue.Queue(maxsize=maxsize)
        self.sent = self.failed = self.dropped = self.rejected = 0
        self.last_error: Optional[str] = None
        self._send = send or self._http
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def submit(self, rec: dict) -> None:
        try:
            self.q.put_nowait(rec)
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.q.put_nowait(rec)

    def _http(self, rec: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(rec).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=2.0) as r:
            return json.loads(r.read() or b"{}")

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                rec = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._send(rec)
                self.sent += 1
                backoff = 0.5
            except urllib.error.HTTPError as e:
                self.rejected += 1  # 422: contract violation - show it loudly, keep going
                self.last_error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
                print("REJECTED by Core:", self.last_error, file=sys.stderr)
            except Exception as e:  # noqa: BLE001 - Core down / network: retry later
                self.failed += 1
                self.last_error = str(e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)

    def close(self, timeout: float = 2.0) -> None:
        end = time.time() + timeout
        while not self.q.empty() and time.time() < end:
            time.sleep(0.05)
        self._stop.set()


# ============================================================== audio sources
def find_mic_array(sd, name: str = "microphone array", min_channels: int = 2) -> int:
    apis = sd.query_hostapis()
    for pref in ("WASAPI", "MME", ""):
        for i, d in enumerate(sd.query_devices()):
            if (d["max_input_channels"] >= min_channels and name.lower() in d["name"].lower()
                    and pref in apis[d["hostapi"]]["name"]):
                return i
    raise SystemExit(f"No {min_channels}-channel input device matching {name!r}. Run with --list-devices.")


def live_windows(cfg: SenderConfig, device: Optional[str], exclusive: bool):
    """Yield (window (n, 2), end_timestamp) every hop from the sound card."""
    import sounddevice as sd

    mono = cfg.mic_channels == 1
    if mono and not device:
        dev = sd.default.device[0] if sd.default.device[0] is not None and sd.default.device[0] >= 0 else find_mic_array(sd, "", 1)
    else:
        dev = find_mic_array(sd, device or "microphone array", 1 if mono else 2)
    info = sd.query_devices(dev)
    host = sd.query_hostapis(info["hostapi"])["name"]
    extra = sd.WasapiSettings(exclusive=True) if (exclusive and "WASAPI" in host) else None
    n_ch = 1 if (mono and info["max_input_channels"] < 2) else 2
    n_win, n_hop = int(cfg.window_s * cfg.fs), int(cfg.hop_s * cfg.fs)
    buf = np.zeros((n_win, 2), dtype=np.float32)
    lock = threading.Lock()
    state = {"new": 0, "overflows": 0}

    def cb(indata, frames, t, status):
        if status:
            state["overflows"] += 1
        with lock:
            buf[:-frames] = buf[frames:]
            buf[-frames:] = indata[:, :2] if n_ch == 2 else np.repeat(indata[:, :1], 2, axis=1)
            state["new"] += frames

    print(f"[capture] device {dev}: {info['name']} | {host} | {cfg.fs} Hz | {n_ch} ch | exclusive={extra is not None}"
          f"{' | MONO: keep the same mic, gain and position at every stop' if mono else ''}")
    with sd.InputStream(device=dev, samplerate=cfg.fs, channels=n_ch, dtype="float32", blocksize=int(0.05 * cfg.fs),
                        callback=cb, extra_settings=extra):
        filled = 0
        while True:
            time.sleep(0.01)
            with lock:
                if state["new"] < n_hop:
                    continue
                filled += state["new"]
                state["new"] = 0
                win = buf.copy()
            if filled >= n_win:
                yield win, time.time()


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        fs, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit("replay supports 16-bit PCM WAV only")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x.reshape(-1, ch), fs


def replay_windows(cfg: SenderConfig, path: str, realtime: bool = True, loop: bool = True):
    x, fs = read_wav(path)
    if fs != cfg.fs:
        raise SystemExit(f"{path} is {fs} Hz; expected {cfg.fs} Hz")
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    n_win, n_hop = int(cfg.window_s * fs), int(cfg.hop_s * fs)
    print(f"[REPLAY] {path}: {len(x) / fs:.1f} s, {x.shape[1]} ch. Records are flagged replay=true.")
    while True:
        for s in range(0, len(x) - n_win + 1, n_hop):
            t0 = time.time()
            yield x[s:s + n_win], time.time()
            if realtime:
                time.sleep(max(0.0, cfg.hop_s - (time.time() - t0)))
        if not loop:
            return


# ============================================================== main
def _member2_bearing():
    from gcc_phat import estimate_bearing  # Member 2's module (same folder)
    return estimate_bearing


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default=os.getenv("AEROECHO_CORE", "http://127.0.0.1:8000"))
    ap.add_argument("--classifier", choices=["yamnet", "energy"], default="yamnet")
    ap.add_argument("--yamnet-model", default=None, help="path to yamnet.tflite or a TF-Hub cache dir (see yamnet_baseline.py)")
    ap.add_argument("--bearing", choices=["reference", "member2"], default="reference",
                    help="reference = built-in GCC-PHAT using calibration json; member2 = gcc_phat.estimate_bearing")
    ap.add_argument("--calib", default=None, help="calibration json (default: calibration.json, then endfire.json)")
    ap.add_argument("--spacing", type=float, default=None, help="override mic spacing (m)")
    ap.add_argument("--offset-us", type=float, default=None, help="override channel offset (us)")
    ap.add_argument("--device", default=None, help='input device name substring (default "microphone array"; with --mono: the Windows default input)')
    ap.add_argument("--shared", action="store_true", help="WASAPI shared mode (NOT recommended: Windows processing breaks timing)")
    ap.add_argument("--mono", action="store_true", help="mono fallback: mic_channels=1, no bearing, snr_db carries loudness")
    ap.add_argument("--replay", default=None, help="16-bit 48 kHz WAV to replay instead of the microphones (flagged)")
    ap.add_argument("--dry-run", action="store_true", help="print records, do not send")
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args(argv)

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    cfg = SenderConfig(core=a.core, replay=bool(a.replay), mic_channels=1 if a.mono else 2)
    cfg.load_calibration(a.calib)
    if a.spacing is not None:
        cfg.spacing_m, cfg.calib_source = a.spacing, "command line"
    if a.offset_us is not None:
        cfg.offset_us = a.offset_us
    clf = make_classifier(a.classifier, model=a.yamnet_model) if a.classifier == "yamnet" else make_classifier("energy")
    bearing_fn = _member2_bearing() if a.bearing == "member2" else None
    proc = Processor(cfg, clf, bearing_fn)
    poster = None if a.dry_run else Poster(cfg.core)
    print(f"[sender] core={cfg.core}{' (dry run)' if a.dry_run else ''} classifier={clf.name} bearing={a.bearing} "
          f"spacing={cfg.spacing_m * 100:.1f} cm offset={cfg.offset_us:+.0f} us (from {cfg.calib_source}) "
          f"clamp=+/-{cfg.clamp_deg:.0f} {'MONO' if a.mono else 'stereo'}")
    src = replay_windows(cfg, a.replay) if a.replay else live_windows(cfg, a.device, exclusive=not a.shared)
    n = 0
    try:
        for win, ts in src:
            r = proc.process(win, ts).record
            n += 1
            if poster:
                poster.submit(r)
            b = "  null" if r["bearing_deg"] is None else f"{r['bearing_deg']:+6.1f}"
            stats = f" sent={poster.sent} fail={poster.failed} rej={poster.rejected}" if poster else ""
            print(f"{'REPLAY ' if cfg.replay else ''}#{n:05d} {r['primary_class']:<9} S={r['detect_score']:.2f} "
                  f"Q={r['listening_quality']:.2f} snr={r['snr_db']:+5.1f}dB bearing={b} pk={r['peak_ratio']:.1f}{stats}")
    except KeyboardInterrupt:
        pass
    finally:
        if poster:
            poster.close()


if __name__ == "__main__":
    main()
