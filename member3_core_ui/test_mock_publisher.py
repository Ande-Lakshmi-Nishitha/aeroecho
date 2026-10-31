"""Hardware-free demo: a simulated victim + simulated audio and Wi-Fi sensors feeding the real Core.

Start the Core first (python main.py), open http://127.0.0.1:8000, then:

    python test_mock_publisher.py                         # voice victim at (1.2, 3.8); YOU move the probe on the dashboard
    python test_mock_publisher.py --auto-move             # simulated operator follows the agent's "move probe" suggestions
    python test_mock_publisher.py --victim -2.0 2.5       # put the victim somewhere else (metres, world frame)
    python test_mock_publisher.py --mode silent           # unconscious victim: no voice, Wi-Fi breathing only
    python test_mock_publisher.py --mode none             # empty room (should never raise a candidate)
    python test_mock_publisher.py --mono                  # mono-mic fallback (no bearings, loudness only)
    python test_mock_publisher.py --noise 20 40           # machinery noise from t=20 s to t=40 s (Q_audio drops)
    python test_mock_publisher.py --auto-move --auto-decide APPROVE   # fully unattended run

It reads the probe's position from GET /grid, so the readings always match where the Core thinks
the probe is. Records follow contracts/perception_schema.json exactly; the Core cannot tell them
from Member 2's real sender. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.error
import urllib.request

AUDIO_ID, WIFI_ID = "laptop_earphone_mics", "esp32_csi_node"


def http(core: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(core + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "detail": e.read().decode(errors="replace")[:300]}


def bearing_reading(probe: tuple[float, float], victim: tuple[float, float]) -> float:
    """What a 2-mic broadside array facing +y reports: rear sources fold to 180 - theta."""
    th = math.degrees(math.atan2(victim[0] - probe[0], victim[1] - probe[1]))
    if abs(th) > 90:
        th = math.copysign(180 - abs(th), th)
    return th


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default="http://127.0.0.1:8000")
    ap.add_argument("--victim", nargs=2, type=float, default=[1.2, 3.8], metavar=("X", "Y"))
    ap.add_argument("--mode", choices=["voice", "silent", "none"], default="voice")
    ap.add_argument("--mono", action="store_true")
    ap.add_argument("--noise", nargs=2, type=float, metavar=("FROM_S", "TO_S"), help="machinery noise window (seconds from start)")
    ap.add_argument("--auto-move", action="store_true", help="simulated operator follows MOVE_PROBE suggestions")
    ap.add_argument("--walk-s", type=float, default=6.0, help="seconds the simulated operator takes to move the probe")
    ap.add_argument("--auto-decide", choices=["APPROVE", "REJECT", "DEFER"], help="answer candidates automatically after 5 s")
    ap.add_argument("--bearing-noise", type=float, default=10.0, help="deg, per 1 s window (3.4 cm laptop array ~10)")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    victim = (a.victim[0], a.victim[1])
    voice_level_db = rng.uniform(26, 34)  # SNR the victim's voice would have at 1 m

    h = http(a.core, "GET", "/health")
    if not h.get("ok"):
        raise SystemExit(f"Core not reachable at {a.core}: start it with  python main.py")
    print(f"Mock sensors -> {a.core}  victim={victim} mode={a.mode} {'MONO' if a.mono else 'stereo'}"
          f"{' auto-move' if a.auto_move else ''}{' auto-decide ' + a.auto_decide if a.auto_decide else ''}")

    t0 = time.time()
    next_audio = next_wifi = t0
    state: dict = {}
    last_poll = 0.0
    moving_until = None
    pending_seen_at: dict[str, float] = {}

    while True:
        now = time.time()
        if now - last_poll >= 0.5:
            state = http(a.core, "GET", "/grid") or state
            last_poll = now
        probe = state.get("probe", {})
        px, py = probe.get("x_m", 0.0), probe.get("y_m", 0.5)
        d = max(math.hypot(victim[0] - px, victim[1] - py), 0.3)
        elapsed = now - t0
        noisy = bool(a.noise and a.noise[0] <= elapsed <= a.noise[1])

        # ---------------- simulated operator
        if a.auto_move:
            sug = state.get("move_suggestion")
            if moving_until is None and sug and probe.get("state") == "SCAN_DONE" and state.get("mode") == "SEARCHING":
                print(f"[operator] moving probe to {sug['waypoint']} ...")
                http(a.core, "POST", "/probe/moving")
                moving_until = (now + a.walk_s, sug["waypoint"])
            elif moving_until is not None and now >= moving_until[0]:
                r = http(a.core, "POST", "/probe/placed", {"waypoint": moving_until[1]})
                print(f"[operator] placed at {moving_until[1]}: {r.get('ok')}")
                moving_until = None
        pc = state.get("pending_candidate")
        if a.auto_decide and pc:
            cid = pc["candidate_id"]
            pending_seen_at.setdefault(cid, now)
            if now - pending_seen_at[cid] >= 5:
                r = http(a.core, "POST", "/hitl/decision", {"candidate_id": cid, "decision": a.auto_decide, "operator": "auto"})
                print(f"[rescuer] {a.auto_decide} {cid} at ({pc['x_m']:+.1f}, {pc['y_m']:.1f}): {r.get('mode')}")
                pending_seen_at[cid] = float("inf")

        # ---------------- audio, 2 records/s
        if now >= next_audio:
            next_audio += 0.5
            calling = a.mode == "voice" and (int(elapsed) % 7) < 4  # calls ~4 s out of every 7
            heard = math.exp(-0.5 * (d / 4.0) ** 2)
            noise_db = 22.0 if noisy else 0.0
            snr = (voice_level_db - 20 * math.log10(d) if calling else 2.0) - noise_db + rng.gauss(0, 3)
            q = max(0.0, min(1.0, (snr + 10) / 30)) if calling else max(0.0, min(1.0, (12 - noise_db + 10) / 30))
            if calling and not noisy:
                s = min(1.0, max(0.0, 0.45 + 0.4 * heard + rng.gauss(0, 0.08)))
                cls = rng.choice(["Voice", "Voice", "Voice", "Cough"])
            else:
                s = min(1.0, max(0.0, (0.25 if noisy else 0.1) + rng.gauss(0, 0.05)))
                cls = "Machinery"
            bearing = None
            if not a.mono and s >= 0.5:
                bearing = max(-60.0, min(60.0, bearing_reading((px, py), victim) + rng.gauss(0, a.bearing_noise)))
            rec = {"schema_version": 1, "sensor_id": AUDIO_ID, "timestamp": time.time(), "detect_score": round(s, 3),
                   "bearing_deg": None if bearing is None else round(bearing, 1), "listening_quality": round(q, 3),
                   "primary_class": cls, "snr_db": round(snr, 1), "mic_channels": 1 if a.mono else 2}
            r = http(a.core, "POST", "/ingest", rec)
            if not r.get("ok"):
                print("audio rejected:", r)

        # ---------------- Wi-Fi, 1 record/s
        if now >= next_wifi:
            next_wifi += 1.0
            if a.mode == "none":
                ws, cls = 0.15 + rng.gauss(0, 0.05), "None"
            else:
                ws = 0.15 + 0.7 * math.exp(-0.5 * (d / 2.5) ** 2) + rng.gauss(0, 0.05)
                cls = "Respiration" if ws >= 0.55 else "None"
            rec = {"schema_version": 1, "sensor_id": WIFI_ID, "timestamp": time.time(), "detect_score": round(min(1, max(0, ws)), 3),
                   "bearing_deg": None, "listening_quality": 0.75, "primary_class": cls}
            http(a.core, "POST", "/ingest", rec)

        time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
