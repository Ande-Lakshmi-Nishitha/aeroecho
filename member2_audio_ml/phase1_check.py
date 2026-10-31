#!/usr/bin/env python3
"""Phase 1 hardware checks for AeroEcho Member 2.  Run from member2_audio_ml/.

  pip install sounddevice numpy scipy
  python phase1_check.py probe                    # REALLY opens every input device; keep talking while it runs
  python phase1_check.py channels --device 19     # are the 2 channels independent? (use a device number!)
  python phase1_check.py stream   --device 19     # buffer/overflow/clock behaviour
  python phase1_check.py clap     --device 19     # bearing sign + sanity (contract section 1)
  python phase1_check.py selftest                 # logic check on synthetic data, no hardware
  (optional: --rate 44100 if 48000 is refused)

Pass/fail thresholds below are HEURISTICS chosen by me, not validated standards.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.signal import find_peaks

from gcc_phat import MIC_SPACING_M, estimate_bearing, _delay

FS = 48000  # contract: capture at 48 kHz for GCC-PHAT
SILENT = 1e-6


def _sd():
    import sounddevice as sd   # imported lazily so `selftest` works without PortAudio
    return sd


def _pick_device(sd, a):
    """Use --device if given, otherwise find the built-in 'Microphone Array' BY NAME
    (device numbers change whenever a USB device is plugged in or removed)."""
    return a.device if a.device is not None else find_mic_array(sd)


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def db(ratio):
    return 20.0 * np.log10(ratio + 1e-12)


# ------------------------------------------------------------------ classify one capture
def classify_capture(x):
    """x: (n, ch) float array. Returns dict(kind, rms, corr).

    kind: SILENT (no signal at all), MONO (1 channel), DUPLICATED (2 identical channels),
          CANDIDATE (2 non-identical, non-summed channels - still needs the tap test).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    r = [rms(x[:, i]) for i in range(x.shape[1])]
    if max(r) < SILENT:
        return dict(kind="SILENT", rms=r, corr=None)
    if x.shape[1] == 1:
        return dict(kind="MONO", rms=r, corr=None)
    if np.max(np.abs(x[:, 0] - x[:, 1])) < 1e-9:
        return dict(kind="DUPLICATED", rms=r, corr=1.0)
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if min(r) >= SILENT else None
    kind = "CANDIDATE" if (corr is None or corr < 0.98) else "SUMMED?"
    return dict(kind=kind, rms=r, corr=corr)


# ------------------------------------------------------------------ probe
_announced = set()


def _try_open(sd, dev, fs, ch, seconds=1.0, exclusive=False):
    info = sd.query_devices(dev)
    host = sd.query_hostapis(info["hostapi"])["name"]
    if dev not in _announced:
        _announced.add(dev)
        print(f"[using device {dev}: {info['name']} | {host}]")
    extra = None
    if exclusive:
        if "WASAPI" in host:
            extra = sd.WasapiSettings(exclusive=True)
        else:   # passing WASAPI settings to another host API raises PaErrorCode -9984
            print(f"[note: device {dev} is {host}, not WASAPI - ignoring --exclusive]")
    with sd.InputStream(device=dev, samplerate=fs, channels=ch, dtype="float32",
                        extra_settings=extra) as s:
        data, _ = s.read(int(fs * seconds))
    return data


def cmd_probe(_):
    sd = _sd()
    print("Keep talking / making noise until this finishes (~1 s per device).\n")
    results = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        host = sd.query_hostapis(d["hostapi"])["name"]
        sr_def = int(d["default_samplerate"])
        n_max = d["max_input_channels"]
        attempts = []
        for cfg in ((FS, 2), (sr_def, 2), (FS, 1), (sr_def, n_max), (sr_def, 1)):
            if cfg not in attempts and cfg[1] <= max(n_max, 2):
                attempts.append(cfg)
        line, last_err = None, ""
        for fs, ch in attempts:
            try:
                data = _try_open(sd, i, fs, ch)
            except Exception as e:                       # noqa: BLE001
                last_err = str(e).splitlines()[0][:60]
                continue
            c = classify_capture(data)
            rr = ", ".join(f"{v:.4f}" for v in c["rms"])
            corr = "" if c["corr"] is None else f" corr={c['corr']:.2f}"
            line = f"{c['kind']:<10} {fs}Hz/{ch}ch rms=({rr}){corr}"
            results.append((i, d["name"], host, c["kind"], fs, ch))
            break
        if line is None:
            line = f"OPEN-FAILED  last error: {last_err}"
        print(f"[{i:>2}] {d['name'][:34]:<34} {host[:14]:<14} {line}")
        if "WASAPI" in host:                       # extra line: exclusive mode bypasses Windows effects
            try:
                c = classify_capture(_try_open(sd, i, FS, 2, exclusive=True))
                rr = ", ".join(f"{v:.4f}" for v in c["rms"])
                corr = "" if c["corr"] is None else f" corr={c['corr']:.2f}"
                print(f"{'':>5}{'  exclusive mode':<48}{c['kind']:<10} {FS}Hz/2ch rms=({rr}){corr}")
                if c["kind"] == "CANDIDATE":
                    results.append((i, d["name"], host + " exclusive", "CANDIDATE", FS, 2))
            except Exception as e:                 # noqa: BLE001
                print(f"{'':>5}{'  exclusive mode':<48}OPEN-FAILED {str(e).splitlines()[0][:50]}")
    print("\nSummary:")
    cands = [r for r in results if r[3] == "CANDIDATE"]
    if cands:
        for i, name, host, _k, fs, ch in cands:
            print(f"  CANDIDATE [{i}] {name} ({host}) at {fs} Hz / {ch} ch -> run: "
                  f"python phase1_check.py channels --device {i} --rate {fs}")
    else:
        print("  No device delivered two different, non-silent channels.")
    sil = [r for r in results if r[3] == "SILENT"]
    if sil:
        print(f"  {len(sil)} device(s) returned digital silence - they are NOT verdicts on channel "
              "independence. Check Windows Settings > Privacy & security > Microphone (allow desktop "
              "apps), mute switches, and that you were making noise during the probe.")


# ------------------------------------------------------------------ channels
def judge_channels(ambient, left_tap, right_tap):
    """Each arg: float array (n, 2). Returns list of (LEVEL, message)."""
    for a in (ambient, left_tap, right_tap):
        if a.ndim != 2 or a.shape[1] != 2:
            return [("FAIL", "recording does not have 2 columns")]
    allrec = np.vstack([ambient, left_tap, right_tap])
    if max(rms(allrec[:, 0]), rms(allrec[:, 1])) < SILENT:
        return [("FAIL", "NO SIGNAL: both channels are digital silence. This is not a verdict on "
                         "channel independence. Check Windows microphone privacy/mute settings, or "
                         "try another device or host API (run `probe`).")]
    out = []
    if np.max(np.abs(allrec[:, 0] - allrec[:, 1])) < 1e-9:
        out.append(("FAIL", "channels are bit-identical: driver is duplicating a mono mic"))
    if rms(ambient[:, 0]) > SILENT and rms(ambient[:, 1]) > SILENT:
        c = float(np.corrcoef(ambient[:, 0], ambient[:, 1])[0, 1])
        if c > 0.98:
            out.append(("WARN", f"ambient inter-channel correlation {c:.3f} > 0.98: channels may be "
                                "summed/beamformed by the driver"))
        else:
            out.append(("INFO", f"ambient inter-channel correlation {c:.3f}"))
    sep_l = db(rms(left_tap[:, 0]) / (rms(left_tap[:, 1]) + 1e-12))     # tapping LEFT mic
    sep_r = db(rms(right_tap[:, 1]) / (rms(right_tap[:, 0]) + 1e-12))   # tapping RIGHT mic
    out.append(("INFO", f"tap LEFT:  ch0 - ch1 = {sep_l:+.1f} dB;  tap RIGHT: ch1 - ch0 = {sep_r:+.1f} dB"))
    if sep_l < -6 and sep_r < -6:
        out.append(("WARN", "channels are SWAPPED: ch0 is the right mic. Swap in code (contract says so)"))
    elif sep_l > 6 and sep_r > 6:
        out.append(("PASS", "channels are independent and correctly ordered (ch0 = left, ch1 = right)"))
    else:
        out.append(("FAIL", "tapping one mic did not isolate its channel by > 6 dB: channels are not "
                            "independent (or the mics are not where you think)"))
    return out


def _rec(sd, dev, fs, seconds, prompt, exclusive=False):
    input(f"\n{prompt}\nPress Enter, then do it for {seconds:.0f} s ... ")
    return _try_open(sd, dev, fs, 2, seconds, exclusive)


def cmd_channels(a):
    sd = _sd()
    ex = a.exclusive
    dev = _pick_device(sd, a)
    amb = _rec(sd, dev, a.rate, 3, "1/3  Make steady noise (talk or hum) a little away from the mics.", ex)
    lt = _rec(sd, dev, a.rate, 3, "2/3  Rub or tap ONLY the LEFT mic, as seen from behind the laptop.", ex)
    rt = _rec(sd, dev, a.rate, 3, "3/3  Rub or tap ONLY the RIGHT mic.", ex)
    print(f"\nRMS ambient per channel: ch0={rms(amb[:,0]):.5f} ch1={rms(amb[:,1]):.5f}")
    for lvl, msg in judge_channels(amb, lt, rt):
        print(f"[{lvl}] {msg}")


# ------------------------------------------------------------------ stream
def cmd_stream(a):
    sd = _sd()
    fs = a.rate
    block = int(0.1 * fs)
    stamps, flags, frames = [], [], []

    def cb(indata, n, t, status):
        stamps.append(time.perf_counter())
        frames.append(n)
        if status:
            flags.append(str(status))

    dev = _pick_device(sd, a)
    host = sd.query_hostapis(sd.query_devices(dev)["hostapi"])["name"]
    extra = sd.WasapiSettings(exclusive=True) if (a.exclusive and "WASAPI" in host) else None
    if a.exclusive and extra is None:
        print(f"[note: device {dev} is {host}, not WASAPI - ignoring --exclusive]")
    print(f"[stream test on device {dev}: {sd.query_devices(dev)['name']} ({host}), exclusive={extra is not None}]")
    with sd.InputStream(samplerate=fs, channels=2, device=dev, blocksize=block,
                        dtype="float32", callback=cb, extra_settings=extra):
        t0 = time.perf_counter()
        time.sleep(a.seconds)
        elapsed = time.perf_counter() - t0
    total = sum(frames)
    dt = np.diff(stamps) * 1000
    print(f"blocks={len(frames)} frames={total} wall={elapsed:.2f}s")
    print(f"effective rate = {total/elapsed:.1f} Hz (nominal {fs}); "
          f"offset {(total/elapsed/fs - 1)*1e6:+.0f} ppm vs host clock")
    print(f"callback interval ms: mean={dt.mean():.1f} std={dt.std():.1f} max={dt.max():.1f} "
          f"(block = {block/fs*1000:.0f} ms)")
    print(f"overflow/status flags: {len(flags)}", set(flags) if flags else "")
    print("[PASS]" if not flags else "[WARN]",
          "no input overflows" if not flags else "overflows occurred - increase blocksize/latency")
    print("Note: a ppm offset vs the HOST clock is normal. Inter-channel sync is guaranteed only "
          "because both channels come from one device.")


# ------------------------------------------------------------------ clap
def bearing_from_claps(x, fs, win_s=0.1, max_claps=3, d=None, offset_us=None):
    """x: (n, 2). Find up to `max_claps` clap transients, return list of result dicts.

    d / offset_us default to the stored calibration. CALIBRATION commands must pass
    d=0.12, offset_us=0.0 so the search window is wide (+/-370 us) and tdoa_us is the raw
    delay; a window sized for the stored 3.4 cm would clip a larger true delay."""
    env = np.abs(x).max(axis=1)
    k = max(1, int(0.01 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    peaks, _ = find_peaks(env, height=0.3 * env.max(), distance=int(0.5 * fs))
    half = int(win_s * fs / 2)
    res = []
    for p in peaks[:max_claps]:
        seg = x[max(0, p - half): p + half]
        if len(seg) == 2 * half:
            res.append(estimate_bearing(seg, fs, d=d, offset_us=offset_us))
    return res


def cmd_clap(a):
    sd = _sd()
    input("Stand ~2 m in front of the laptop, ~1 m to the RIGHT of the centre line.\n"
          "Press Enter, then clap 3 times, ~1 s apart ... ")
    _countdown(a.delay)
    x = _try_open(sd, _pick_device(sd, a), a.rate, 2, 6.0, a.exclusive)
    res = bearing_from_claps(x, a.rate)
    if not res:
        print("[FAIL] no clap detected (too quiet? wrong device? digital silence?)")
        return
    for i, r in enumerate(res, 1):
        if r["bearing_raw_deg"] is None:
            print(f"clap {i}: silent window")
            continue
        print(f"clap {i}: bearing={r['bearing_raw_deg']:+.1f} deg  raw tdoa={r['tdoa_us']:+.0f} us  "
              f"corrected={r['tdoa_corrected_us']:+.0f} us  peak_ratio={r['peak_ratio']:.1f}")
    vals = [r["bearing_raw_deg"] for r in res if r["bearing_raw_deg"] is not None]
    if vals:
        print(f"median bearing = {np.median(vals):+.1f} deg   (about +27 if you stand 2 m ahead, 1 m right; "
              "one clap can easily be off by 6-12 deg at this mic spacing)")
        r0 = [r for r in res if r["bearing_raw_deg"] is not None][0]
        print(f"calibration used: spacing {r0['spacing_m']*100:.1f} cm, offset {r0['offset_us']:+.0f} us "
              f"(source: {r0['calibration']})")
    print("Positive => sign convention OK. Negative => swap channels in code, not in the contract.")


def _countdown(n):
    for i in range(int(n), 0, -1):
        print(f"  recording starts in {i} ...", end="\r", flush=True)
        time.sleep(1)
    print("  RECORDING NOW - clap HARD 5 times, ~1 s apart.   ")


# ------------------------------------------------------------------ one-shot AMD array check
def judge_sides(tau_left_us, tau_right_us):
    """TDOA (us) of claps from the LEFT side and from the RIGHT side. Heuristic thresholds."""
    if len(tau_left_us) < 2 or len(tau_right_us) < 2:
        return [("FAIL", "fewer than 2 claps detected on one side - clap louder and rerun")]
    ml, mr = float(np.median(tau_left_us)), float(np.median(tau_right_us))
    sl = float(np.ptp(tau_left_us)); sr = float(np.ptp(tau_right_us))
    out = [("INFO", f"LEFT claps:  median TDOA {ml:+.0f} us, spread {sl:.0f} us")]
    out.append(("INFO", f"RIGHT claps: median TDOA {mr:+.0f} us, spread {sr:.0f} us"))
    if abs(ml) < 10 and abs(mr) < 10:
        out.append(("FAIL", "no time difference between the channels for either side: they are "
                            "duplicated, summed or beamformed - no usable direction"))
    elif ml < -25 and mr > 25:
        if max(sl, sr) < 40:
            out.append(("PASS", "TDOA flips sign between left and right and is consistent: the array "
                                "gives usable direction (sign convention correct)"))
        else:
            out.append(("WARN", "sign flips correctly but claps scatter by > 40 us: direction is "
                                "usable only coarsely, or processing is adding jitter"))
    elif ml > 25 and mr < -25:
        out.append(("WARN", "direction works but LEFT/RIGHT are reversed: swap channels in code"))
    else:
        out.append(("FAIL", "TDOA does not separate left from right consistently: not usable"))
    return out


def cmd_amdcheck(a):
    sd = _sd()
    devs = []
    for i, d in enumerate(sd.query_devices()):
        host = sd.query_hostapis(d["hostapi"])["name"]
        if (d["max_input_channels"] >= 2 and "WASAPI" in host
                and "array" in d["name"].lower() and "AMD" in d["name"].upper()):
            devs.append(i)
    if not devs:
        print("[FAIL] no AMD Microphone Array (WASAPI, 2 channels) found. Run `probe` and send me the output.")
        return
    dev = devs[0]
    try:
        input("\nStep 1/3: keep talking steadily. Press Enter ... ")
        c = classify_capture(_try_open(sd, dev, FS, 2, 2.0, a.exclusive))
        print(f"Step 1 result: {c['kind']} rms={[round(v, 4) for v in c['rms']]} corr={c['corr']}")
        if c["kind"] in ("SILENT", "DUPLICATED", "MONO"):
            print(f"[FAIL] the array is delivering {c['kind']} audio right now: no direction possible. "
                  "(If it was working before, a Windows setting or app probably changed it back.)")
            return
        taus = {}
        for n, side in ((2, "LEFT"), (3, "RIGHT")):
            input(f"\nStep {n}/3: stand ~1.5 m away, about 45 degrees to the {side} of the laptop "
                  f"(your {side.lower()} hand, sitting at it). Press Enter, then clap hard 3 times, ~1 s apart ... ")
            x = _try_open(sd, dev, FS, 2, 6.0, a.exclusive)
            taus[side] = [r["tdoa_us"] for r in bearing_from_claps(x, FS, d=0.12, offset_us=0.0)
                          if r["tdoa_us"] is not None]
            print(f"  TDOA per clap (us): {[round(t) for t in taus[side]]}")
    except Exception as e:                                  # noqa: BLE001
        print(f"[FAIL] could not record: {e}")
        return
    print()
    for lvl, msg in judge_sides(taus["LEFT"], taus["RIGHT"]):
        print(f"[{lvl}] {msg}")


# ------------------------------------------------------------------ end-fire test (final AMD check)
def find_mic_array(sd):
    """Pick the built-in array BY NAME (device numbers shift when USB devices are plugged in)."""
    apis = sd.query_hostapis()
    for pref in ("WASAPI", "MME"):
        for i, d in enumerate(sd.query_devices()):
            if (d["max_input_channels"] >= 2 and "microphone array" in d["name"].lower()
                    and pref in apis[d["hostapi"]]["name"]):
                return i
    raise RuntimeError("no 2-channel 'Microphone Array' found under WASAPI or MME")


def endfire_verdict(tau_left_us, tau_right_us):
    """tau = t_left - t_right. A source on the line through both mics, on the LEFT, gives
    tau = -d/c; on the RIGHT, +d/c. Returns (list of (LEVEL, msg), spacing_cm, offset_us)."""
    tl, tr = float(np.median(tau_left_us)), float(np.median(tau_right_us))
    d_cm = 343.0 * (tr - tl) / 2.0 * 1e-6 * 100.0
    off = (tl + tr) / 2.0
    spread = max(np.ptp(tau_left_us), np.ptp(tau_right_us))
    out = [("INFO", f"median tau: LEFT side {tl:+.0f} us, RIGHT side {tr:+.0f} us "
                    f"(spread between claps up to {spread:.0f} us; 1 sample = {1e6/FS:.0f} us at 48 kHz)")]
    lb = 343.0 * max(np.max(np.abs(tau_left_us)), np.max(np.abs(tau_right_us))) * 1e-6 * 100.0
    out.append(("INFO", f"spacing is AT LEAST {lb:.1f} cm (largest single |tau|; a clap off the mic line gives a "
                        f"smaller delay, so the median-based estimate below tends to read low)"))
    if tl < -40 and tr > 40:
        if spread <= 60:
            out.append(("PASS", "timing flips sign between the two sides: two real mics, ch0 = left, ch1 = right"))
        else:
            out.append(("WARN", f"sign flips, but claps scatter by {spread:.0f} us: NOT a clean pass. Repeat "
                                "before trusting the spacing or the direction"))
    elif tl > 40 and tr < -40:
        out.append(("WARN", "timing flips but the wrong way: channels are SWAPPED (swap in code)"))
    elif abs(tl) < 20 and abs(tr) < 20:
        out.append(("FAIL", "no timing difference on either side: channels look summed/processed into one signal"))
    else:
        out.append(("WARN", "inconsistent timing: same sign on both sides or too small to trust; repeat"))
    if abs(d_cm) < 2 or abs(d_cm) > 20:
        out.append(("WARN", f"implied spacing {abs(d_cm):.1f} cm is outside the plausible 2-20 cm range"))
    if spread > 60:
        out.append(("WARN", "claps disagree by > 60 us (about 3 samples): noise or a processing driver; repeat"))
    return out, abs(d_cm), off


def cmd_endfire(a):
    import json
    sd = _sd()
    dev = a.device if a.device is not None else find_mic_array(sd)
    taus = {}
    for side in ("LEFT", "RIGHT"):
        input(f"\nStand about 50 cm to your {side} of the laptop (your own left/right when sitting at it), "
              f"level with the top edge where the webcam is, in line with the two mics.\n"
              f"Press Enter, then walk to the spot. Recording starts after a {a.delay:.0f} s countdown ... ")
        _countdown(a.delay)
        x = _try_open(sd, dev, a.rate, 2, 8.0, a.exclusive)
        c = classify_capture(x)
        if c["kind"] in ("SILENT", "MONO", "DUPLICATED"):
            print(f"[FAIL] device returned {c['kind']} - this is not a usable 2-mic stream. "
                  "Run `probe` and check the AMD array still shows CANDIDATE.")
            return
        res = [r for r in bearing_from_claps(x, a.rate, max_claps=5, d=0.12, offset_us=0.0)
               if r["tdoa_us"] is not None]
        if len(res) < 3:
            print(f"[FAIL] only {len(res)} claps detected; clap louder / closer and rerun")
            return
        taus[side] = [r["tdoa_us"] for r in res]
        print("  per clap (tdoa_us, peak_ratio):", [(round(r["tdoa_us"]), round(r["peak_ratio"], 1)) for r in res])
    out, d_cm, off = endfire_verdict(taus["LEFT"], taus["RIGHT"])
    for lvl, msg in out:
        print(f"[{lvl}] {msg}")
    print(f"\nMeasured mic spacing = {d_cm:.1f} cm (contract assumes 12 cm); boresight offset = {off:+.0f} us")
    json.dump(dict(spacing_m=d_cm / 100.0, offset_us=off, left_us=taus["LEFT"], right_us=taus["RIGHT"],
                   rate=a.rate), open("endfire.json", "w"), indent=1)
    print("Saved endfire.json")


# ------------------------------------------------------------------ calibrate spacing
def fit_spacing(angles_deg, taus_us):
    """Fit tau_us = offset + k*sin(theta). Returns effective spacing (cm), offset (us), residual (us)."""
    s = np.sin(np.radians(np.asarray(angles_deg, dtype=float)))
    t = np.asarray(taus_us, dtype=float)
    A = np.stack([np.ones_like(s), s], axis=1)
    (off, k), *_ = np.linalg.lstsq(A, t, rcond=None)
    resid = t - (off + k * s)
    return dict(offset_us=float(off), spacing_cm=float(k * 1e-6 * 343.0 * 100.0),
                resid_us=float(resid.std(ddof=2)) if len(t) > 2 else float("nan"))


def cmd_calibrate(a):
    import json
    sd = _sd()
    angles = [float(v) for v in a.angles.split(",")]
    print("0 deg = straight out from the array's broadside; positive = to the RIGHT as seen from "
          "behind the laptop. Keep the laptop in the SAME orientation you will use in the demo.")
    A, T = [], []
    for ang in angles:
        input(f"\nNEXT MARKER: {a.radius} m away at {ang:+.0f} deg.\n"
              f"Press Enter, then walk to that marker. Recording starts in {a.delay:.0f} s ... ")
        time.sleep(a.delay)
        print("  >>> RECORDING NOW: clap 3 times, about 1 s apart, then stay still <<<")
        x = _try_open(sd, _pick_device(sd, a), a.rate, 2, 6.0, a.exclusive)
        res = [r for r in bearing_from_claps(x, a.rate, d=0.12, offset_us=0.0) if r["tdoa_us"] is not None]
        if not res:
            print("  no claps detected, skipping this angle")
            continue
        taus = [r["tdoa_us"] for r in res]
        print("  tdoa_us per clap:", [round(t) for t in taus])
        A += [ang] * len(taus); T += taus
    if len(set(A)) < 3:
        print("[FAIL] need claps at >= 3 distinct angles"); return
    f = fit_spacing(A, T)
    sig_deg = np.degrees(f["resid_us"] * 1e-6 * 343.0 / (f["spacing_cm"] / 100.0)) if f["spacing_cm"] > 0 else float("nan")
    print(f"\nEffective spacing = {f['spacing_cm']:.1f} cm   (contract assumes 12 cm)")
    print(f"Boresight offset  = {f['offset_us']:+.0f} us   (0 means the array faces where you think)")
    print(f"Residual scatter  = {f['resid_us']:.0f} us  ~ {sig_deg:.1f} deg near broadside")
    json.dump(dict(spacing_m=f["spacing_cm"] / 100.0, offset_us=f["offset_us"], resid_us=f["resid_us"],
                   rate=a.rate, angles=A, taus_us=T), open("calibration.json", "w"), indent=1)
    print("Saved calibration.json. A negative spacing means the channels are swapped.")


# ------------------------------------------------------------------ selftest
def cmd_selftest(_):
    rng = np.random.default_rng(2)
    ang = np.repeat([-45, -30, 0, 30, 45], 3)
    tau = 10 + 0.06 / 343.0 * 1e6 * np.sin(np.radians(ang)) + rng.normal(0, 15, len(ang))
    f = fit_spacing(ang, tau)
    print(f"fit_spacing on synthetic data (true d=6.0 cm, offset=+10 us, noise 15 us): "
          f"d={f['spacing_cm']:.1f} cm, offset={f['offset_us']:+.0f} us, resid={f['resid_us']:.0f} us\n")
    rng = np.random.default_rng(1)
    n = 3 * FS
    amb = rng.standard_normal((n, 2)).astype(np.float32) * 0.01
    mono = np.repeat(amb[:, :1], 2, axis=1)
    tapL = amb.copy(); tapL[:, 0] += rng.standard_normal(n).astype(np.float32) * 0.2
    tapR = amb.copy(); tapR[:, 1] += rng.standard_normal(n).astype(np.float32) * 0.2
    zeros = np.zeros((n, 2), dtype=np.float32)
    print("judge_sides on synthetic TDOAs (us):")
    for name, L, R in (("good", [-90, -85, -95], [88, 92, 85]), ("duplicated", [1, -2, 0], [2, 1, -1]),
                       ("reversed", [90, 85, 95], [-88, -92, -85]), ("noisy", [-90, -20, -140], [88, 10, 150]),
                       ("too few", [-90], [88, 90])):
        print(f"  {name:<10}", [lvl for lvl, _ in judge_sides(L, R)][-1])
    print()
    print("Case A: independent channels, correct order")
    for l, m in judge_channels(amb, tapL, tapR): print(f"  [{l}] {m}")
    print("Case B: driver-duplicated mono")
    for l, m in judge_channels(mono, np.repeat(tapL[:, :1], 2, 1), np.repeat(tapR[:, :1], 2, 1)):
        print(f"  [{l}] {m}")
    print("Case C: channels swapped")
    for l, m in judge_channels(amb, tapR, tapL): print(f"  [{l}] {m}")
    print("Case D: device returns digital silence (the case I previously mislabelled)")
    for l, m in judge_channels(zeros, zeros, zeros): print(f"  [{l}] {m}")

    print("\nclassify_capture on synthetic captures:")
    for name, x in (("independent", amb), ("duplicated", mono), ("silent", zeros),
                    ("1-channel", amb[:, :1]), ("summed", np.stack([amb[:, 0], amb[:, 0] * 1.01], 1))):
        c = classify_capture(x)
        print(f"  {name:<12} -> {c['kind']}")

    print("\nClap detector on synthetic claps at +27 deg (right):")
    sig = np.zeros((6 * FS, 2))
    for t in (1.0, 2.2, 3.4):
        burst = rng.standard_normal(int(0.03 * FS)) * np.exp(-np.arange(int(0.03 * FS)) / (0.005 * FS))
        pad = np.zeros(int(0.2 * FS)); pad[: len(burst)] = burst
        tau = MIC_SPACING_M * np.sin(np.radians(27)) / 343.0
        L, R = _delay(pad, tau, FS), pad
        i = int(t * FS)
        sig[i: i + len(pad), 0] += L; sig[i: i + len(pad), 1] += R
    sig += rng.standard_normal(sig.shape) * 1e-3
    for r in bearing_from_claps(sig, FS, d=MIC_SPACING_M, offset_us=0.0):
        print(f"  raw={r['bearing_raw_deg']:+.1f} deg tdoa={r['tdoa_us']:+.0f} us")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("probe", cmd_probe), ("channels", cmd_channels), ("stream", cmd_stream),
                     ("clap", cmd_clap), ("calibrate", cmd_calibrate), ("amdcheck", cmd_amdcheck),
                     ("endfire", cmd_endfire), ("selftest", cmd_selftest)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        if name == "endfire":
            p.add_argument("--device", type=int, default=None, help="optional; default = find the AMD array by name")
            p.add_argument("--rate", type=int, default=FS)
            p.add_argument("--exclusive", action="store_true")
            p.add_argument("--delay", type=float, default=6.0, help="countdown seconds so you can walk to the spot")
        if name == "amdcheck":
            p.add_argument("--exclusive", action="store_true")
        if name == "clap":
            p.add_argument("--delay", type=float, default=6.0, help="countdown seconds so you can walk to the spot")
        if name == "calibrate":
            p.add_argument("--angles", default="-45,-30,0,30,45")
            p.add_argument("--radius", type=float, default=2.0)
            p.add_argument("--delay", type=float, default=6.0, help="seconds to walk to the marker")
        if name in ("channels", "stream", "clap", "calibrate"):
            p.add_argument("--device", type=int, default=None,
                           help="optional device number; default = find the AMD 'Microphone Array' by name")
            p.add_argument("--rate", type=int, default=FS)
            p.add_argument("--exclusive", action="store_true", help="WASAPI exclusive mode (bypass Windows effects)")
        if name == "stream":
            p.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    args.fn(args)
