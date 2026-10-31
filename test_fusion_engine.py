"""Run from member3_core_ui/:  python -m pytest -q test_fusion_engine.py"""
import json
import math

import pytest

import config as C
from fusion_engine import (AWAITING_HITL, ESTOP, HOLD, MOVING, SCAN_DONE, SCANNING, SEARCHING, SETTLING,
                           FusionEngine)

T0 = 1_793_442_000.0


def audio(ts, s=0.9, q=0.9, b=10.0):
    return {"sensor_id": C.SENSOR_AUDIO, "timestamp": ts, "detect_score": s, "bearing_deg": b,
            "listening_quality": q, "primary_class": "Voice"}


def wifi(ts, s=0.6, q=0.7, rng=None):
    return {"sensor_id": C.SENSOR_WIFI, "timestamp": ts, "detect_score": s, "bearing_deg": None,
            "listening_quality": q, "primary_class": "Respiration", "range_m": rng}


def bearing(probe, target):
    return math.degrees(math.atan2(target[0] - probe[0], target[1] - probe[1]))


def new_engine(events=None):
    return FusionEngine(on_event=(lambda k, d: events.append((k, d))) if events is not None else None, now=T0)


def scan_stop(eng, wp, victim, t, n=30, s=0.92):
    """Place probe at wp at time t, wait out settling, stream n audio records toward victim. Returns (t, actions)."""
    eng.probe_moving(t)
    eng.probe_placed(t + 1, waypoint=wp)
    t = t + 1 + C.AUDIO_SETTLE_S + 0.1
    acts = []
    b = bearing(eng.grid.waypoints[wp], victim)
    for _ in range(n):
        acts += eng.ingest(audio(t, s=s, b=b), now=t).actions
        t += 0.5
    return t, acts


# --------------------------------------------------------------- gating
def test_starts_placed_at_start_waypoint_and_settling():
    eng = new_engine()
    assert eng.probe_state == SETTLING and eng.probe_waypoint == C.START_WAYPOINT
    assert eng.grid.probe_xy == C.WAYPOINTS[C.START_WAYPOINT]


def test_records_gated_by_probe_state():
    eng = new_engine()
    r = eng.ingest(audio(T0 + 1), now=T0 + 1)
    assert not r.used and r.reason == "audio settling"
    t = T0 + C.AUDIO_SETTLE_S + 0.5
    assert eng.ingest(audio(t), now=t).used
    w = eng.ingest(wifi(t), now=t)
    assert not w.used and w.reason == "wifi settling"
    t2 = T0 + C.WIFI_SETTLE_S + 0.5
    assert eng.ingest(wifi(t2), now=t2).used
    eng.probe_moving(t2 + 1)
    assert eng.ingest(audio(t2 + 1.5), now=t2 + 1.5).reason == "probe moving"


def test_stale_and_pre_placement_records_rejected():
    eng = new_engine()
    t = T0 + 10
    assert eng.ingest(audio(t - 5), now=t).reason == "stale record"
    eng.probe_placed(t, waypoint="W3")
    t2 = t + C.AUDIO_SETTLE_S + 1
    assert eng.ingest(audio(t - 0.2), now=t2).reason in ("stale record", "recorded before placement")


def test_unused_records_do_not_touch_grid():
    eng = new_engine()
    before = eng.grid.prob().copy()
    eng.ingest(audio(T0 + 1), now=T0 + 1)  # settling
    assert (eng.grid.prob() == before).all()


# --------------------------------------------------------------- fusion math
def test_quality_weighted_fusion_and_ema():
    eng = new_engine()
    t = T0 + C.WIFI_SETTLE_S + 1
    eng.ingest(audio(t, s=0.8, q=0.9), now=t)
    eng.ingest(wifi(t, s=0.4, q=0.3), now=t)
    expected = (0.9 * 0.8 + 0.3 * 0.4) / (0.9 + 0.3)
    assert math.isclose(eng.state.s_master_raw, expected, rel_tol=1e-4)
    assert 0 < eng.state.s_master < expected  # EMA lags the raw value


def test_noisy_audio_defers_to_wifi():
    eng = new_engine()
    t = T0 + C.WIFI_SETTLE_S + 1
    eng.ingest(audio(t, s=0.95, q=0.0), now=t)  # machinery noise: Q_audio -> 0
    eng.ingest(wifi(t, s=0.3, q=0.8), now=t)
    assert math.isclose(eng.state.s_master_raw, 0.3, rel_tol=1e-4)


def test_stale_modality_drops_out():
    eng = new_engine()
    t = T0 + C.WIFI_SETTLE_S + 1
    eng.ingest(wifi(t, s=0.2, q=0.9), now=t)
    t2 = t + C.STALE_S + 0.5
    eng.ingest(audio(t2, s=0.8, q=0.9), now=t2)
    assert eng.state.q_wifi == 0.0
    assert math.isclose(eng.state.s_master_raw, 0.8, rel_tol=1e-4)


def test_degraded_when_all_quality_zero():
    eng = new_engine()
    t = T0 + C.AUDIO_SETTLE_S + 1
    eng.ingest(audio(t, s=0.9, q=0.0), now=t)
    assert eng.state.degraded and eng.state.s_master_raw == 0.0


# --------------------------------------------------------------- tiers
def test_full_loop_triangulates_raises_candidate_and_points_camera():
    events = []
    eng = new_engine(events)
    victim = (0.0, 4.5)
    t, acts = scan_stop(eng, "W1", victim, T0 + 1)
    if eng.mode != AWAITING_HITL:
        t, acts2 = scan_stop(eng, "W2", victim, t)
        acts += acts2
    assert eng.mode == AWAITING_HITL
    cand = [a for a in acts if a.type == "CANDIDATE"]
    assert len(cand) == 1 and cand[0].tier == 2
    c = eng.pending
    assert math.hypot(c.x_m - victim[0], c.y_m - victim[1]) <= 0.7  # 12 deg bearing sigma (3.4 cm array)
    pan = [a for a in acts if a.type == "PAN"][-1]
    assert abs(pan.payload["angle_deg"] - eng.grid.turret_angle_to(c.row, c.col)) <= C.PAN_STEP_DEG / 2
    assert c.evidence_record_ids and len(c.evidence_record_ids) <= C.EVIDENCE_IDS_KEEP
    assert any(k == "CANDIDATE" for k, _ in events)


def _engine_with_candidate():
    eng = new_engine()
    victim = (0.0, 4.5)
    t, _ = scan_stop(eng, "W1", victim, T0 + 1)
    if eng.mode != AWAITING_HITL:
        t, _ = scan_stop(eng, "W2", victim, t)
    assert eng.mode == AWAITING_HITL
    return eng, t


def test_awaiting_hitl_blocks_new_actions_then_times_out_to_hold():
    eng, t = _engine_with_candidate()
    acts = eng.ingest(audio(t, b=0.0), now=t).actions
    assert not any(a.type in ("PAN", "CANDIDATE", "DWELL") for a in acts)
    eng.tick(eng.hitl_deadline + 0.1)
    assert eng.mode == HOLD and eng.pending is not None
    out = eng.hitl_decision(eng.pending.candidate_id, "APPROVE", now=eng.hitl_deadline + 5)
    assert out["mode"] == SEARCHING and out["found"] == 1


def test_approve_marks_found_and_suppresses_region():
    eng, t = _engine_with_candidate()
    c = eng.pending
    eng.hitl_decision(c.candidate_id, "approve", now=t)
    assert eng.found[0]["candidate_id"] == c.candidate_id
    assert eng.grid.region_mass(c.row, c.col) < 1e-6
    assert eng._suppressed(c.row, c.col)


def test_reject_lowers_region():
    eng, t = _engine_with_candidate()
    c = eng.pending
    before = eng.grid.region_mass(c.row, c.col)
    eng.hitl_decision(c.candidate_id, "REJECT", now=t)
    assert eng.grid.region_mass(c.row, c.col) < before and eng.mode == SEARCHING
    assert eng.state.consecutive_high == 0


def test_defer_cools_down_region():
    eng, t = _engine_with_candidate()
    c = eng.pending
    eng.hitl_decision(c.candidate_id, "DEFER", now=t)
    assert eng._suppressed(c.row, c.col)
    eng.tick(t + C.DEFER_COOLDOWN_S + 1)
    assert not eng._suppressed(c.row, c.col)


def test_bad_decisions_rejected():
    eng, t = _engine_with_candidate()
    with pytest.raises(ValueError):
        eng.hitl_decision("c-9999", "APPROVE", now=t)
    with pytest.raises(ValueError):
        eng.hitl_decision(eng.pending.candidate_id, "MAYBE", now=t)


def test_dwell_extends_scan_once():
    eng = new_engine()
    t = T0 + C.AUDIO_SETTLE_S + 1
    first_done = eng.scan_done_at
    dwells = []
    for i in range(20):
        dwells += [a for a in eng.ingest(audio(t, s=0.7, b=None), now=t).actions if a.type == "DWELL"]
        t += 0.5
    assert len(dwells) == C.MAX_DWELL_EXTENSIONS
    assert eng.scan_done_at >= first_done + C.DWELL_S - 1e-9


def test_move_probe_suggested_once_after_scan_window():
    eng = new_engine()
    t = eng.scan_done_at + 0.1
    acts = eng.tick(t)
    moves = [a for a in acts if a.type == "MOVE_PROBE"]
    assert len(moves) == 1 and moves[0].payload["waypoint"] != C.START_WAYPOINT
    assert eng.probe_state == SCAN_DONE
    assert not [a for a in eng.tick(t + 1) if a.type == "MOVE_PROBE"]


def test_no_move_suggestion_while_awaiting_rescuer():
    eng, t = _engine_with_candidate()
    eng.move_suggested = False
    acts = eng.tick(eng.scan_done_at + 1)
    assert not [a for a in acts if a.type == "MOVE_PROBE"]


def test_pan_rate_limited_and_on_grid_steps():
    eng = new_engine()
    t = T0 + C.AUDIO_SETTLE_S + 1
    pans = []
    for _ in range(10):
        pans += [a for a in eng.ingest(audio(t, s=0.6, b=40.0), now=t).actions if a.type == "PAN"]
        t += 0.25
    assert pans, "camera should turn toward the evidence"
    assert all(a.payload["angle_deg"] % C.PAN_STEP_DEG == 0 for a in pans)
    times = [a.timestamp for a in pans]
    assert all(b - a >= C.PAN_MIN_INTERVAL_S for a, b in zip(times, times[1:]))


# --------------------------------------------------------------- safety
def test_estop_blocks_turret_and_restores_mode():
    eng = new_engine()
    eng.set_estop(True, now=T0 + 1)
    assert eng.mode == ESTOP
    t = T0 + C.AUDIO_SETTLE_S + 1
    acts = []
    for _ in range(10):
        acts += eng.ingest(audio(t, b=40.0), now=t).actions
        t += 0.5
    assert not [a for a in acts if a.type == "PAN"]
    assert eng.ingest(audio(t), now=t).used  # sensing continues
    assert eng.set_estop(False, now=t) == SEARCHING


def test_estop_during_candidate_returns_to_awaiting():
    eng, t = _engine_with_candidate()
    eng.set_estop(True, now=t)
    assert eng.set_estop(False, now=t + 1) == AWAITING_HITL


# --------------------------------------------------------------- probe + wifi
def test_probe_placed_variants_and_validation():
    eng = new_engine()
    assert eng.probe_placed(T0, row=10, col=5)["waypoint"] is None
    assert eng.probe_placed(T0, x_m=1.5, y_m=1.0)["waypoint"] == "W2"
    assert eng.probe_placed(T0, waypoint="W3")["x_m"] == 0.0
    for bad in ({}, {"waypoint": "W9"}, {"row": 3}, {"x_m": 9.0, "y_m": 1.0}, {"waypoint": "W1", "row": 1, "col": 1}):
        with pytest.raises(ValueError):
            eng.probe_placed(T0, **bad)


def test_wifi_range_only_used_at_calibrated_stop():
    a, b = new_engine(), new_engine()
    b.probe_placed(T0, waypoint="W3")
    for eng in (a, b):
        t = T0 + C.WIFI_SETTLE_S + 1
        for _ in range(10):
            eng.ingest(wifi(t, s=0.8, q=0.9, rng=2.0), now=t)
            t += 1
    ring_cell = a.grid.world_to_cell(0.0, 0.5 + 2.0)
    near_cell = a.grid.world_to_cell(0.0, 0.7)  # 0.2 m from W0, far off the 2 m ring
    assert a.grid.prob()[ring_cell] > 1.5 * a.grid.prob()[near_cell]  # ring at W0
    rb = b.grid.prob()
    r_on = b.grid.world_to_cell(0.0, 5.0)  # 2 m from W3 along +y
    r_in = b.grid.world_to_cell(0.0, 3.2)  # right next to W3
    assert rb[r_in] >= rb[r_on]  # no ring at W3: plain presence falloff


# --------------------------------------------------------------- outputs
def test_snapshot_is_json_and_complete():
    eng = new_engine()
    snap = eng.snapshot(now=T0 + 1)
    s = json.loads(json.dumps(snap))
    for key in ("mode", "fusion", "probe", "next_waypoint", "best_region", "p_absent", "entropy_bits", "grid", "found"):
        assert key in s
    assert len(s["grid"]) == 30 and len(s["grid"][0]) == 30
    assert "grid" not in eng.snapshot(now=T0 + 1, include_grid=False)


def test_events_emitted_for_audit():
    events = []
    eng = new_engine(events)
    t = T0 + C.AUDIO_SETTLE_S + 1
    eng.ingest(audio(t), now=t)
    kinds = {k for k, _ in events}
    assert {"PROBE", "INGEST"} <= kinds
    json.dumps([d for _, d in events])  # all audit payloads serialisable


# --------------------------------------------------------------- mono fallback
def test_mono_ignores_bearing_and_localises_by_loudness():
    import numpy as np
    rng = np.random.default_rng(0)
    eng = new_engine()
    victim, l1 = (1.2, 3.0), 30.0
    t = T0 + 1
    for wp in ("W0", "W2", "W3", "W1", "W5"):
        eng.probe_moving(t)
        eng.probe_placed(t + 1, waypoint=wp)
        t += 1 + C.AUDIO_SETTLE_S + 0.1
        px, py = eng.grid.waypoints[wp]
        d = max(math.hypot(victim[0] - px, victim[1] - py), 0.3)
        for _ in range(30):
            rec = audio(t, s=0.9, b=45.0)  # bearing present but must be ignored in mono
            rec.update(mic_channels=1, snr_db=l1 - 20 * math.log10(d) + rng.normal(0, 3))
            eng.ingest(rec, now=t)
            t += 0.5
        if eng.mode == AWAITING_HITL:
            break
    assert eng.grid.audio_mono and eng.snapshot(now=t)["audio_mode"] == "mono"
    assert eng.mode == AWAITING_HITL
    c = eng.pending
    assert math.hypot(c.x_m - victim[0], c.y_m - victim[1]) < 1.2
    assert eng.snapshot(now=t, include_grid=False)["best_region"]["size_m"] == pytest.approx(1.8)


def test_snr_ignored_in_stereo():
    eng = new_engine()
    t = T0 + C.AUDIO_SETTLE_S + 1
    rec = audio(t)
    rec["snr_db"] = 20.0
    eng.ingest(rec, now=t)
    assert not eng.grid.audio_mono and eng.grid.levels  # recorded ...
    assert not eng.grid._level_active()  # ... but not used in stereo


# --------------------------------------------------------------- silent victim (Wi-Fi only)
def test_silent_victim_found_by_wifi_breathing():
    import numpy as np
    rng = np.random.default_rng(0)
    eng = new_engine()
    victim = (1.2, 3.8)
    t, end = T0, T0 + 600
    while t < end and eng.mode != AWAITING_HITL:
        px, py = eng.grid.probe_xy
        d = math.hypot(victim[0] - px, victim[1] - py)
        acts = list(eng.ingest(audio(t, s=float(np.clip(0.12 + rng.normal(0, 0.05), 0, 1)), b=None), now=t).actions)
        if abs((t - T0) % 1.0) < 1e-6:
            ws = float(np.clip(0.15 + 0.7 * math.exp(-0.5 * (d / 2.5) ** 2) + rng.normal(0, 0.05), 0, 1))
            acts += eng.ingest(wifi(t, s=ws, q=0.75), now=t).actions
        acts += eng.tick(t)
        for a in acts:
            if a.type == "MOVE_PROBE":
                eng.probe_moving(t)
                t += 8
                eng.probe_placed(t, waypoint=a.payload["waypoint"])
        t += 0.5
    assert eng.mode == AWAITING_HITL
    c = eng.pending
    assert c.basis == "wifi_breathing" and c.radius_cells == C.CANDIDATE_RADIUS_CELLS_WIFI
    assert math.hypot(c.x_m - victim[0], c.y_m - victim[1]) < 1.5
