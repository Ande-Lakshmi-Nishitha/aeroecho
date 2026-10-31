"""Run from member2_audio_ml/:  python -m pytest -q test_aeroecho_sender.py   (no microphone or TensorFlow needed)"""
import json
import math
import os
import sys
import threading
import time

import numpy as np
import pytest

from aeroecho_sender import (EnergyClassifier, Poster, Processor, SenderConfig, bearing_from_tau, gcc_phat,
                             resample_48k_to_16k)
from make_test_wav import frac_delay, make, write_wav
from yamnet_baseline import build_group_index, scores_to_probs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FS = 48000


def run_windows(x, cfg=None, clf=None):
    p = Processor(cfg or SenderConfig(), clf or EnergyClassifier())
    return [p.process(x[s:s + FS], time.time()).record for s in range(0, len(x) - FS + 1, FS // 2)]


# ---------------------------------------------------------------- GCC-PHAT
@pytest.mark.parametrize("ang", [-55, -30, -10, 0, 15, 30, 50])
def test_gcc_phat_recovers_known_bearing_with_offset(ang):
    rng = np.random.default_rng(abs(ang) + 7)
    s = rng.normal(0, 1, FS)
    tau = 0.034 * math.sin(math.radians(ang)) / 343.0
    left = frac_delay(s, max(tau, 0) + 20e-6, FS) + rng.normal(0, 0.05, FS)
    right = frac_delay(s, max(-tau, 0), FS) + rng.normal(0, 0.05, FS)
    t, pk = gcc_phat(left, right, FS, 0.034 / 343 + 40e-6, (300, 4000))
    b, tau_us = bearing_from_tau(t, 0.034, 20.0, 60.0)
    assert abs(b - ang) < 1.0 and pk > 5
    assert abs(tau_us - tau * 1e6) < 3.0  # payload tdoa_us is offset-corrected


def test_sign_convention_right_is_positive():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, FS)
    left, right = frac_delay(s, 80e-6, FS), s  # right mic hears it first
    t, _ = gcc_phat(left, right, FS, 150e-6, (300, 4000))
    assert t > 0 and bearing_from_tau(t, 0.034, 0.0, 60.0)[0] > 0


def test_clamp_at_contract_limit():
    assert bearing_from_tau(200e-6, 0.034, 0.0, 60.0)[0] == 60.0
    assert bearing_from_tau(-200e-6, 0.034, 0.0, 60.0)[0] == -60.0


@pytest.mark.parametrize("ang", [-40, 0, 25, 45])
def test_voice_with_echoes(ang):
    recs = run_windows(make(ang, 14, seed=ang + 100))
    b = [r["bearing_deg"] for r in recs if r["bearing_deg"] is not None]
    assert len(b) >= 5 and abs(np.median(b) - ang) < 3.0  # SIMULATED; expect ~10 deg in a real room


def test_silence_gives_null_bearing_and_low_score():
    recs = run_windows(make(0, 12, silent=True))
    assert all(r["bearing_deg"] is None for r in recs)
    assert max(r["detect_score"] for r in recs) < 0.5


def test_mono_mode_has_no_bearing_but_snr():
    cfg = SenderConfig(mic_channels=1)
    recs = run_windows(make(20, 10), cfg=cfg)
    assert all(r["bearing_deg"] is None and r["mic_channels"] == 1 and "snr_db" in r for r in recs)


def test_calibration_file_and_swapped_channels(tmp_path):
    p = tmp_path / "endfire.json"
    p.write_text(json.dumps({"spacing_m": -0.036, "offset_us": 18.0}))
    cfg = SenderConfig()
    cfg.load_calibration(str(p))
    assert cfg.swap_channels and cfg.spacing_m == pytest.approx(0.036) and cfg.offset_us == 18.0
    p.write_text(json.dumps({"spacing_m": 0.0017}))
    with pytest.raises(SystemExit):
        SenderConfig().load_calibration(str(p))


def test_resampler():
    t = np.arange(FS) / FS
    low = resample_48k_to_16k(np.sin(2 * np.pi * 1000 * t))
    high = resample_48k_to_16k(np.sin(2 * np.pi * 10000 * t))  # above the new 8 kHz Nyquist
    assert len(low) == 16000 and np.std(low) > 0.6 and np.std(high) < 0.05


# ---------------------------------------------------------------- contract
def test_records_match_contract_schema_and_core_model():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.load(open(os.path.join(ROOT, "contracts", "perception_schema.json")))
    recs = run_windows(make(25, 8)) + run_windows(make(0, 6, silent=True)) + run_windows(make(25, 6), cfg=SenderConfig(mic_channels=1))
    for r in recs:
        jsonschema.validate(r, schema)
    sys.path.insert(0, os.path.join(ROOT, "member3_core_ui"))
    try:
        from main import PerceptionRecord
    except Exception:  # noqa: BLE001 - Core deps not installed on this machine
        return
    for r in recs:
        PerceptionRecord(**r)


# ---------------------------------------------------------------- YAMNet mapping (no TensorFlow)
def test_yamnet_group_mapping():
    names = ["Speech", "Shout", "Cough", "Breathing", "Silence", "Engine", "Throat clearing", "Gasp"]
    idx, missing = build_group_index(names)
    assert set(idx["Voice"]) == {0, 1} and set(idx["Cough"]) == {2, 6} and set(idx["Breathing"]) == {3, 7}
    assert set(idx["Machinery"]) == {4, 5} and missing  # not every AudioSet label is in this toy list
    speech = scores_to_probs(np.array([[0.9, 0.1, 0.0, 0.0, 0.02, 0.05, 0.0, 0.0]]), idx)
    silence = scores_to_probs(np.array([[0.01, 0.0, 0.0, 0.0, 0.95, 0.02, 0.0, 0.0]]), idx)
    assert max(speech, key=speech.get) == "Voice" and speech["Voice"] > 0.8
    assert max(silence, key=silence.get) == "Machinery" and silence["Voice"] < 0.05
    assert abs(sum(speech.values()) - 1) < 1e-6


# ---------------------------------------------------------------- transport
def test_poster_never_blocks_and_drops_oldest():
    gate = threading.Event()
    got = []

    def slow_send(rec):
        gate.wait(2)
        got.append(rec["n"])
        return {}

    p = Poster("http://x", maxsize=3, send=slow_send)
    t0 = time.time()
    for i in range(10):
        p.submit({"n": i})
    assert time.time() - t0 < 0.1 and p.dropped >= 6
    gate.set()
    p.close()
    assert got[-1] == 9


def test_end_to_end_into_core(tmp_path):
    """Synthetic victim at +25 deg from W0 -> sender -> real Core API -> grid points along +25 deg."""
    pytest.importorskip("fastapi")
    sys.path.insert(0, os.path.join(ROOT, "member3_core_ui"))
    from fastapi.testclient import TestClient
    from main import Settings, create_app
    import config as CC

    app = create_app(Settings(pi_url="mock", wifi="off", audit_path=str(tmp_path / "a.jsonl")))
    with TestClient(app) as client:
        eng = app.state.core.engine
        eng.placed_at -= CC.AUDIO_SETTLE_S + 1  # skip the settle wait
        replies = []

        def send(rec):
            r = client.post("/ingest", json=rec)
            assert r.status_code == 200, r.text
            replies.append(r.json())
            return r.json()

        poster = Poster("http://unused", send=send)
        x = make(25, 14, seed=3)
        proc = Processor(SenderConfig(), EnergyClassifier())
        for s in range(0, len(x) - FS + 1, FS // 2):
            poster.submit(proc.process(x[s:s + FS], time.time()).record)
        poster.close(5)
        assert len(replies) >= 20 and sum(r["used"] for r in replies) >= 10
        g = eng.grid
        px, py = g.probe_xy
        cells = g.prob()
        on = sum(cells[g.world_to_cell(px + d * math.sin(math.radians(25)), py + d * math.cos(math.radians(25)))] for d in (1, 2, 3))
        off = sum(cells[g.world_to_cell(px + d * math.sin(math.radians(-25)), py + d * math.cos(math.radians(-25)))] for d in (1, 2, 3))
        assert on > 5 * off  # the Core drew the wedge on the correct (right) side
