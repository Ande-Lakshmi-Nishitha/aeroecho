"""API + audit + Wi-Fi adapter tests. Run from member3_core_ui/:  python -m pytest -q test_core_api.py"""
import json
import time

import pytest
from fastapi.testclient import TestClient

import config as C
from audit_logger import AuditLogger, verify_file
from main import Settings, create_app
from wifi_adapter import telemetry_to_record


@pytest.fixture()
def client(tmp_path):
    app = create_app(Settings(pi_url="mock", wifi="off", audit_path=str(tmp_path / "audit.jsonl")))
    with TestClient(app) as c:
        yield c


def audio(ts=None, **kw):
    r = {"schema_version": 1, "sensor_id": C.SENSOR_AUDIO, "timestamp": ts or time.time(), "detect_score": 0.9,
         "bearing_deg": 20.0, "listening_quality": 0.9, "primary_class": "Voice"}
    r.update(kw)
    return r


# ---------------------------------------------------------------- API
def test_dashboard_and_health(client):
    assert client.get("/").status_code == 200 and "AEROECHO" in client.get("/").text
    h = client.get("/health").json()
    assert h["ok"] and h["pi_mock"] and h["mode"] == "SEARCHING"


def test_ingest_validates_contract(client):
    assert client.post("/ingest", json=audio()).json()["ok"]
    bad = [audio(primary_class="Respiration"), audio(bearing_deg=75.0), audio(detect_score=1.5),
           audio(mic_channels=1),  # mono needs snr_db
           {**audio(), "sensor_id": C.SENSOR_WIFI, "primary_class": "Respiration"},  # Wi-Fi with a bearing
           {**audio(), "sensor_id": "phone"}]
    for rec in bad:
        assert client.post("/ingest", json=rec).status_code == 422, rec


def test_settling_then_used(client):
    r = client.post("/ingest", json=audio()).json()
    assert r["used"] is False and "settling" in r["reason"]


def test_probe_flow_and_errors(client):
    assert client.post("/probe/moving").json()["probe"] == "MOVING"
    assert client.post("/probe/placed", json={"waypoint": "W3"}).json()["waypoint"] == "W3"
    assert client.post("/probe/placed", json={"row": 10, "col": 5}).json()["ok"]
    assert client.post("/probe/placed", json={"waypoint": "W9"}).status_code == 400
    assert client.post("/probe/placed", json={}).status_code == 400
    nw = client.get("/next_waypoint").json()
    assert nw["name"] in C.WAYPOINTS


def test_state_snapshot_and_websocket(client):
    s = client.get("/grid").json()
    assert len(s["grid"]) == 30 and s["waypoints"]["W0"] == list(C.WAYPOINTS["W0"])
    with client.websocket_connect("/ws/state") as ws:
        msg = ws.receive_json()
        assert msg["mode"] == "SEARCHING" and "fusion" in msg


def test_estop_camera_reset_snapshot(client):
    assert client.post("/estop", json={"engaged": True}).json()["mode"] == "ESTOP"
    assert client.post("/estop", json={"engaged": False}).json()["mode"] == "SEARCHING"
    assert client.post("/camera/observation", json={"person_visible": False}).json()["ok"]
    assert client.post("/reset").json()["ok"]
    r = client.get("/camera/snapshot")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")


def test_hitl_errors_and_audit_chain(client):
    assert client.post("/hitl/decision", json={"candidate_id": "c-9999", "decision": "APPROVE"}).status_code == 400
    v = client.get("/audit/verify").json()
    assert v["ok"] and v["entries"] >= 2
    assert all("hash" in e for e in client.get("/audit/recent").json())


def test_full_candidate_cycle_through_api(client):
    """Drive the real HTTP API with records from two stops; expect a candidate, then APPROVE it."""
    import math
    core = client.app.state.core
    eng = core.engine
    victim = (0.0, 4.5)
    t = time.time()
    for wp in ("W1", "W2"):
        client.post("/probe/moving")
        client.post("/probe/placed", json={"waypoint": wp})
        eng.placed_at -= C.AUDIO_SETTLE_S + 1  # skip the settle wait in a unit test
        x, y = C.WAYPOINTS[wp]
        b = math.degrees(math.atan2(victim[0] - x, victim[1] - y))
        for _ in range(25):
            client.post("/ingest", json=audio(ts=time.time(), bearing_deg=round(b, 1), detect_score=0.92))
        if eng.mode == "AWAITING_HITL":
            break
    s = client.get("/grid").json()
    assert s["mode"] == "AWAITING_HITL", s["fusion"]
    pc = s["pending_candidate"]
    assert math.hypot(pc["x_m"] - victim[0], pc["y_m"] - victim[1]) < 0.8
    assert s["evidence"]["candidate_id"] == pc["candidate_id"]
    out = client.post("/hitl/decision", json={"candidate_id": pc["candidate_id"], "decision": "APPROVE"}).json()
    assert out["mode"] == "SEARCHING" and out["found"] == 1
    assert out["evidence"]["decision"]["decision"] == "APPROVE"
    assert client.get("/audit/verify").json()["ok"]


# ---------------------------------------------------------------- audit logger
def test_audit_chain_detects_tampering(tmp_path):
    p = tmp_path / "a.jsonl"
    log = AuditLogger(str(p))
    for i in range(5):
        log.log("TEST", {"i": i, "f": 0.1 * i})
    assert verify_file(str(p))[0]
    lines = p.read_text().splitlines()
    e = json.loads(lines[2]); e["data"]["i"] = 99; lines[2] = json.dumps(e)
    p.write_text("\n".join(lines) + "\n")
    ok, n, _, msg = verify_file(str(p))
    assert not ok and n == 2 and "seq 3" in msg


def test_audit_resumes_chain(tmp_path):
    p = str(tmp_path / "a.jsonl")
    a = AuditLogger(p); a.log("X", {"k": 1})
    b = AuditLogger(p); b.log("Y", {"k": 2})
    assert b.seq == 2 and verify_file(p)[0]


# ---------------------------------------------------------------- Wi-Fi adapter mapping
def test_wifi_mapping():
    base = {"type": "telemetry", "server_time": 1793442000.0, "confidence": 0.7, "link": {"packet_rate_hz": 100.0}}
    r = telemetry_to_record({**base, "state": "breathing"})
    assert r["primary_class"] == "Respiration" and r["listening_quality"] == 1.0 and r["bearing_deg"] is None
    m = telemetry_to_record({**base, "state": "motion", "confidence": 0.2})
    assert m["primary_class"] == "Presence" and m["detect_score"] == C.WIFI_MOTION_SCORE and m["listening_quality"] <= C.WIFI_MOTION_QUALITY_CAP
    assert telemetry_to_record({**base, "state": "warming_up"}) is None
    assert telemetry_to_record({"type": "alert_ack"}) is None
    cal = telemetry_to_record({**base, "state": "breathing", "distance_source": "calibrated", "distance_m": 2.1, "distance_uncertainty_m": 0.4})
    assert cal["range_m"] == 2.1 and cal["range_sigma_m"] == 0.4
