"""AeroEcho Wi-Fi adapter: ghop-ghop telemetry -> PerceptionRecord (sensor_id = esp32_csi_node).

ghop-ghop (github.com/Chunnu-Munnu/ghop-ghop) owns the ESP32 serial port and does all CSI DSP.
Run it unchanged on port 8001, then this adapter reads its WebSocket (~1 message/s):

    cd ghop-ghop/backend && python app.py --port 8001              # real ESP32
    cd ghop-ghop/backend && python app.py --port 8001 --simulate   # no hardware (badged SIMULATION)

main.py runs the adapter in-process (python main.py --wifi ghop). Standalone, it POSTs to the Core:

    python wifi_adapter.py --core http://127.0.0.1:8000 --ghop ws://127.0.0.1:8001/ws

Mapping (ghop-ghop `state` -> record):
    breathing  -> Respiration, S = confidence,             Q = link quality
    motion     -> Presence,    S = max(confidence, 0.6),   Q = min(link, 0.5)
    empty      -> None,        S = confidence,             Q = link quality
    no_signal / warming_up -> no record (Core treats Wi-Fi as stale, Q = 0)
    link quality = clip(packet_rate_hz / WIFI_FULL_PACKET_RATE_HZ, 0, 1)
    range_m only when ghop-ghop reports distance_source == "calibrated"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from typing import Any, Awaitable, Callable

import config as C


def telemetry_to_record(t: dict[str, Any]) -> dict[str, Any] | None:
    """One ghop-ghop telemetry message -> PerceptionRecord dict, or None if it carries no evidence."""
    if t.get("type") != "telemetry":
        return None
    state = t.get("state")
    if state in (None, "no_signal", "warming_up"):
        return None
    link = t.get("link") or {}
    rate = float(link.get("packet_rate_hz") or 0.0)
    q_link = max(0.0, min(1.0, rate / C.WIFI_FULL_PACKET_RATE_HZ))
    conf = float(t.get("confidence") or 0.0)
    if state == "breathing":
        cls, s, q = "Respiration", conf, q_link
    elif state == "motion":
        cls, s, q = "Presence", max(conf, C.WIFI_MOTION_SCORE), min(q_link, C.WIFI_MOTION_QUALITY_CAP)
    else:  # "empty" or anything unexpected
        cls, s, q = "None", conf, q_link
    rec: dict[str, Any] = {
        "schema_version": C.CONTRACT_VERSION,
        "sensor_id": C.SENSOR_WIFI,
        "timestamp": float(t.get("server_time") or time.time()),
        "detect_score": round(max(0.0, min(1.0, s)), 4),
        "bearing_deg": None,
        "listening_quality": round(q, 4),
        "primary_class": cls,
    }
    if t.get("distance_source") == "calibrated" and t.get("distance_m") is not None:
        rec["range_m"] = float(t["distance_m"])
        rec["range_sigma_m"] = float(t.get("distance_uncertainty_m") or 0.5)
    rec["ghop_source"] = t.get("source")  # "serial" or "simulation" (extra key, ignored by the schema)
    rec["bpm"] = t.get("bpm")
    return rec


class WifiAdapter:
    def __init__(self, ghop_ws: str = C.GHOP_WS_URL) -> None:
        self.ghop_ws = ghop_ws
        self.connected = False
        self.last_state: str | None = None
        self.last_message_at: float | None = None
        self.records_sent = 0
        self.source: str | None = None

    def status(self) -> dict[str, Any]:
        return {"connected": self.connected, "url": self.ghop_ws, "state": self.last_state,
                "source": self.source, "last_message_at": self.last_message_at, "records": self.records_sent}

    async def run(self, on_record: Callable[[dict[str, Any]], Awaitable[None] | None]) -> None:
        """Connect, map, forward forever; reconnect every 2 s if ghop-ghop is down."""
        import websockets  # installed with uvicorn[standard]

        while True:
            try:
                async with websockets.connect(self.ghop_ws, open_timeout=3, ping_interval=10) as ws:
                    self.connected = True
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self.last_message_at = time.time()
                        self.last_state = msg.get("state", self.last_state)
                        self.source = msg.get("source", self.source)
                        rec = telemetry_to_record(msg)
                        if rec is None:
                            continue
                        out = on_record(rec)
                        if asyncio.iscoroutine(out):
                            await out
                        self.records_sent += 1
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - ghop-ghop not running / restarted
                self.connected = False
                await asyncio.sleep(2.0)


def _post(core: str, rec: dict[str, Any]) -> None:
    req = urllib.request.Request(f"{core}/ingest", data=json.dumps(rec).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        r.read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default=f"http://127.0.0.1:{C.CORE_PORT}")
    ap.add_argument("--ghop", default=C.GHOP_WS_URL)
    a = ap.parse_args()
    adapter = WifiAdapter(a.ghop)

    def forward(rec: dict[str, Any]) -> None:
        try:
            _post(a.core, rec)
            print(f"{rec['primary_class']:<11} S={rec['detect_score']:.2f} Q={rec['listening_quality']:.2f}")
        except Exception as e:  # noqa: BLE001
            print("Core not reachable:", e)

    print(f"ghop-ghop {a.ghop} -> Core {a.core}  (Ctrl+C to stop)")
    asyncio.run(adapter.run(forward))


if __name__ == "__main__":
    main()
