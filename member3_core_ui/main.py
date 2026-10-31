"""AeroEcho Core server (Member 3): FastAPI + fusion engine + search map + audit + dashboard.

Run (from member3_core_ui/):
    python main.py                                   # demo: mock Pi turret, no Wi-Fi adapter
    python main.py --pi http://192.168.1.50:5000     # real Pi turret
    python main.py --wifi ghop                       # read ghop-ghop on ws://127.0.0.1:8001/ws
Then open http://127.0.0.1:8000 (or http://<laptop-ip>:8000 from a phone).
Simulated sensors for a hardware-free demo:  python test_mock_publisher.py --auto-move

Endpoints (contracts/CONTRACT.md):
    POST /ingest               PerceptionRecord -> {"ok", "record_id", "used", "reason", "mode"}
    POST /probe/moving         operator picked the probe up
    POST /probe/placed         {"waypoint": "W3"} | {"row": 10, "col": 5} | {"x_m": 0.4, "y_m": 2.1}
    GET  /next_waypoint        agent's suggested stop + expected information gain
    POST /hitl/decision        {"candidate_id", "decision": APPROVE|REJECT|DEFER, "operator"?, "note"?}
    POST /camera/observation   {"person_visible": true|false}  (operator looked at the snapshot)
    POST /estop                {"engaged": true|false}
    POST /reset                belief back to prior (audit chain kept)
    GET  /grid                 full state snapshot (same as the WebSocket)
    GET  /camera/snapshot      proxied from the Pi (or a mock image)
    GET  /audit/verify         recompute the SHA-256 chain
    GET  /audit/recent         last audit entries (no INGEST spam)
    GET  /evidence/{id}        evidence card for a candidate
    GET  /health               liveness + sensor freshness
    WS   /ws/state             state snapshot pushed after every change (<= 4 Hz) and every second
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

import config as C
from audit_logger import AuditLogger
from fusion_engine import FusionEngine
from wifi_adapter import WifiAdapter

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


# ---------------------------------------------------------------- settings
@dataclass
class Settings:
    pi_url: str = "mock"  # "mock" or http://<pi-ip>:5000
    wifi: str = "off"  # "off" or "ghop"
    ghop_ws: str = C.GHOP_WS_URL
    audit_path: str = C.AUDIT_LOG_PATH

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            pi_url=os.getenv("AEROECHO_PI_URL", "mock"),
            wifi=os.getenv("AEROECHO_WIFI", "off"),
            ghop_ws=os.getenv("AEROECHO_GHOP_WS", C.GHOP_WS_URL),
            audit_path=os.getenv("AEROECHO_AUDIT_LOG", C.AUDIT_LOG_PATH),
        )


# ---------------------------------------------------------------- schemas
AUDIO_CLASSES = Literal["Voice", "Cough", "Breathing", "Machinery"]
ALL_CLASSES = Literal["Voice", "Cough", "Breathing", "Machinery", "Respiration", "Presence", "None"]


class PerceptionRecord(BaseModel):
    """contracts/perception_schema.json v1. Extra keys are allowed and kept."""
    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    sensor_id: Literal["laptop_earphone_mics", "esp32_csi_node"]
    timestamp: float
    detect_score: float = Field(ge=0.0, le=1.0)
    bearing_deg: Optional[float] = Field(default=None, ge=-60.0, le=60.0)
    listening_quality: float = Field(ge=0.0, le=1.0)
    primary_class: ALL_CLASSES
    mic_channels: Optional[Literal[1, 2]] = None
    snr_db: Optional[float] = None
    tdoa_us: Optional[float] = None
    class_probs: Optional[dict[str, float]] = None
    range_m: Optional[float] = Field(default=None, ge=0.0, le=5.0)
    range_sigma_m: Optional[float] = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _pairing(self) -> "PerceptionRecord":
        audio_cls = {"Voice", "Cough", "Breathing", "Machinery"}
        if self.sensor_id == C.SENSOR_WIFI:
            if self.bearing_deg is not None:
                raise ValueError("esp32_csi_node must send bearing_deg = null")
            if self.primary_class in audio_cls:
                raise ValueError("Wi-Fi primary_class must be Respiration | Presence | None")
        else:
            if self.primary_class not in audio_cls:
                raise ValueError("audio primary_class must be Voice | Cough | Breathing | Machinery")
            if self.mic_channels == 1 and self.snr_db is None:
                raise ValueError("snr_db is required when mic_channels = 1")
        return self


class Placed(BaseModel):
    waypoint: Optional[str] = None
    row: Optional[int] = None
    col: Optional[int] = None
    x_m: Optional[float] = None
    y_m: Optional[float] = None


class Decision(BaseModel):
    candidate_id: str
    decision: Literal["APPROVE", "REJECT", "DEFER", "approve", "reject", "defer"]
    operator: str = "rescuer"
    note: str = ""


class CameraObs(BaseModel):
    person_visible: bool
    angle_deg: Optional[float] = None


class EStop(BaseModel):
    engaged: bool


# ---------------------------------------------------------------- Pi turret client
class PiClient:
    """Talks to Member 1's turret server, or pretends to (url == "mock")."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.mock = url == "mock"
        self.ok = self.mock
        self.angle_deg = 0.0
        self.last_error: Optional[str] = None
        self._client = None

    async def _http(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=C.PI_TIMEOUT_S)
        return self._client

    async def pan(self, angle: float) -> bool:
        if self.mock:
            await asyncio.sleep(0.3)  # pretend the servo moves and settles
            self.angle_deg = float(angle)
            return True
        try:
            r = await (await self._http()).post(f"{self.url}/pan", params={"angle": int(round(angle))})
            r.raise_for_status()
            self.angle_deg = float(r.json().get("angle_deg", angle))
            self.ok, self.last_error = True, None
            return True
        except Exception as e:  # noqa: BLE001
            self.ok, self.last_error = False, f"pan failed: {e}"
            return False

    async def health(self) -> None:
        if self.mock:
            return
        try:
            r = await (await self._http()).get(f"{self.url}/health")
            r.raise_for_status()
            self.angle_deg = float(r.json().get("angle_deg", self.angle_deg))
            self.ok, self.last_error = True, None
        except Exception as e:  # noqa: BLE001
            self.ok, self.last_error = False, f"health failed: {e}"

    async def snapshot(self) -> tuple[bytes, str]:
        if not self.mock:
            try:
                r = await (await self._http()).get(f"{self.url}/snapshot")
                r.raise_for_status()
                return r.content, r.headers.get("content-type", "image/jpeg")
            except Exception as e:  # noqa: BLE001
                self.ok, self.last_error = False, f"snapshot failed: {e}"
        return self._mock_image().encode(), "image/svg+xml"

    def _mock_image(self) -> str:
        t = time.strftime("%H:%M:%S")
        note = "MOCK CAMERA" if self.mock else "PI UNREACHABLE"
        return (f"<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360' viewBox='0 0 640 360'>"
                f"<rect width='640' height='360' fill='#1c2230'/>"
                f"<g stroke='#3a4560' stroke-width='1'>" + "".join(f"<line x1='{x}' y1='0' x2='{x}' y2='360'/>" for x in range(0, 641, 80)) + "</g>"
                f"<circle cx='320' cy='180' r='46' fill='none' stroke='#e8b04b' stroke-width='3'/>"
                f"<line x1='320' y1='110' x2='320' y2='250' stroke='#e8b04b'/><line x1='250' y1='180' x2='390' y2='180' stroke='#e8b04b'/>"
                f"<text x='20' y='34' fill='#c9d3e6' font-family='monospace' font-size='20'>{note}</text>"
                f"<text x='20' y='336' fill='#c9d3e6' font-family='monospace' font-size='18'>pan {self.angle_deg:+.0f} deg  {t}</text></svg>")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()


# ---------------------------------------------------------------- application
class Core:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audit = AuditLogger(settings.audit_path)
        self.engine = FusionEngine(on_event=self.audit)
        self.pi = PiClient(settings.pi_url)
        self.wifi = WifiAdapter(settings.ghop_ws) if settings.wifi == "ghop" else None
        self.clients: set[WebSocket] = set()
        self.messages: deque[dict[str, Any]] = deque(maxlen=40)  # agent messages for the UI log
        self.evidence: dict[str, dict[str, Any]] = {}
        self.move_suggestion: Optional[dict[str, Any]] = None
        self.last_ingest: dict[str, float] = {}
        self.ingest_counts = {"audio": 0, "wifi": 0, "used": 0}
        self._dirty = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self.started_at = time.time()
        self.audit.log("SYSTEM", {"event": "core_started", "pi": settings.pi_url, "wifi": settings.wifi})
        self.say("system", f"Core started. Turret: {'MOCK' if self.pi.mock else settings.pi_url}. "
                           f"Wi-Fi: {'ghop-ghop ' + settings.ghop_ws if self.wifi else 'off'}.")

    # -------------------------------------------------------------- helpers
    def say(self, kind: str, text: str) -> None:
        self.messages.append({"t": time.time(), "kind": kind, "text": text})
        self.mark_dirty()

    def mark_dirty(self) -> None:
        with contextlib.suppress(RuntimeError):
            self._dirty.set()

    def handle_actions(self, actions: list) -> None:
        for a in actions:
            if a is None:
                continue
            p = a.payload
            if a.type == "PAN":
                self._tasks.append(asyncio.create_task(self._do_pan(p["angle_deg"])))
            elif a.type == "MOVE_PROBE":
                self.move_suggestion = dict(p, at=a.timestamp)
                self.say("agent", f"Move probe to {p['waypoint']} ({p['x_m']:+.1f}, {p['y_m']:.1f}) m: "
                                  f"expected information gain {p['expected_info_gain_bits']:.2f} bits.")
            elif a.type == "DWELL":
                self.say("agent", f"Interesting signal: listening {p['seconds']:.0f} s longer at this stop.")
            elif a.type == "CANDIDATE":
                self.evidence[p["candidate_id"]] = self.audit.evidence_card(p)
                where = f"({p['x_m']:+.1f}, {p['y_m']:.1f}) m"
                self.say("alert", f"CANDIDATE {p['candidate_id']} at {where}, {p['basis'].replace('_', ' ')}, "
                                  f"P(region) {p['region_mass']:.2f}. Waiting for rescuer decision.")
        self.mark_dirty()

    async def _do_pan(self, angle: float) -> None:
        if await self.pi.pan(angle):
            self.engine.pan_confirmed(self.pi.angle_deg)
        else:
            self.say("error", self.pi.last_error or "pan failed")
        self.mark_dirty()

    def ingest(self, rec: dict[str, Any]) -> dict[str, Any]:
        rec = {k: v for k, v in rec.items() if v is not None or k == "bearing_deg"}  # drop unset optionals
        res = self.engine.ingest(rec)
        mod = "audio" if rec["sensor_id"] == C.SENSOR_AUDIO else "wifi"
        self.last_ingest[mod] = time.time()
        self.ingest_counts[mod] += 1
        self.ingest_counts["used"] += int(res.used)
        self.handle_actions(res.actions)
        return {"ok": True, "record_id": res.record_id, "used": res.used, "reason": res.reason, "mode": self.engine.mode}

    def state(self, include_grid: bool = True) -> dict[str, Any]:
        s = self.engine.snapshot(include_grid=include_grid)
        now = time.time()
        s["waypoints"] = {k: list(v) for k, v in C.WAYPOINTS.items()}
        s["turret_xy"] = list(C.TURRET_XY)
        s["grid_n"], s["cell_m"] = C.GRID_N, C.CELL_M
        s["pi"] = {"mock": self.pi.mock, "ok": self.pi.ok, "angle_deg": self.pi.angle_deg, "error": self.pi.last_error}
        s["wifi_adapter"] = self.wifi.status() if self.wifi else None
        s["sensors"] = {m: (round(now - t, 1) if t else None) for m, t in
                        (("audio_age_s", self.last_ingest.get("audio")), ("wifi_age_s", self.last_ingest.get("wifi")))}
        s["ingest_counts"] = dict(self.ingest_counts)
        s["messages"] = list(self.messages)[-15:]
        s["move_suggestion"] = self.move_suggestion if self.engine.probe_state == "SCAN_DONE" else None
        s["audit"] = {"length": self.audit.seq, "head": self.audit.last_hash[:16]}
        pc = s.get("pending_candidate")
        s["evidence"] = self.evidence.get(pc["candidate_id"]) if pc else None
        s["thresholds"] = {"t2_s_master": C.T2_THRESHOLD, "t2_region": C.T2_MIN_REGION_P, "t1": C.T1_THRESHOLD}
        return s

    # -------------------------------------------------------------- loops
    async def broadcast(self) -> None:
        if not self.clients:
            return
        payload = self.state()
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def broadcaster(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._dirty.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self._dirty.clear()
            await self.broadcast()
            await asyncio.sleep(0.25)  # <= 4 Hz

    async def ticker(self) -> None:
        last_mode, n = self.engine.mode, 0
        while True:
            self.handle_actions(self.engine.tick(time.time()))
            if self.engine.mode != last_mode:
                if self.engine.mode == "HOLD":
                    self.say("alert", "No rescuer decision in time: search on HOLD until a decision is made.")
                last_mode = self.engine.mode
            n += 1
            if n % 5 == 0:
                await self.pi.health()
            self.mark_dirty()
            await asyncio.sleep(1.0)

    async def wifi_loop(self) -> None:
        async def on_record(rec: dict[str, Any]) -> None:
            try:
                self.ingest(PerceptionRecord(**rec).model_dump())
            except Exception as e:  # noqa: BLE001
                self.say("error", f"Wi-Fi record rejected: {e}")
        await self.wifi.run(on_record)

    async def start(self) -> None:
        self._tasks += [asyncio.create_task(self.broadcaster()), asyncio.create_task(self.ticker())]
        if self.wifi:
            self._tasks.append(asyncio.create_task(self.wifi_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await self.pi.close()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    holder: dict[str, Core] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        core = Core(settings)
        holder["core"] = core
        _app.state.core = core
        await core.start()
        try:
            yield
        finally:
            await core.stop()

    app = FastAPI(title="AeroEcho Core", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def core() -> Core:
        return holder["core"]

    def bad(e: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(e))

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    async def health():
        c = core()
        now = time.time()
        return {"ok": True, "mode": c.engine.mode, "uptime_s": round(now - c.started_at, 1),
                "last_audio_ts": c.last_ingest.get("audio"), "last_wifi_ts": c.last_ingest.get("wifi"),
                "pi_ok": c.pi.ok, "pi_mock": c.pi.mock, "wifi_adapter": c.wifi.status() if c.wifi else None,
                "audit_entries": c.audit.seq}

    @app.post("/ingest")
    async def ingest(rec: PerceptionRecord):
        return core().ingest(rec.model_dump())

    @app.post("/probe/moving")
    async def probe_moving():
        c = core()
        c.engine.probe_moving(time.time())
        c.move_suggestion = None
        c.say("operator", "Probe picked up: evidence paused while it moves.")
        return {"ok": True, "probe": c.engine.probe_state}

    @app.post("/probe/placed")
    async def probe_placed(p: Placed):
        c = core()
        try:
            info = c.engine.probe_placed(time.time(), **p.model_dump(exclude_none=True))
        except ValueError as e:
            raise bad(e)
        c.move_suggestion = None
        where = info["waypoint"] or f"({info['x_m']:+.1f}, {info['y_m']:.1f}) m"
        c.say("operator", f"Probe placed at {where}. Audio live in {C.AUDIO_SETTLE_S:.0f} s, Wi-Fi in {C.WIFI_SETTLE_S:.0f} s.")
        return {"ok": True, **info}

    @app.get("/next_waypoint")
    async def next_waypoint():
        s = core().engine.snapshot(include_grid=False)
        return s["next_waypoint"]

    @app.post("/hitl/decision")
    async def hitl_decision(d: Decision):
        c = core()
        cand = c.engine.pending.to_dict() if c.engine.pending else None
        try:
            out = c.engine.hitl_decision(d.candidate_id, d.decision, operator=d.operator, note=d.note)
        except ValueError as e:
            raise bad(e)
        if cand:
            c.evidence[d.candidate_id] = c.audit.evidence_card(cand, {"decision": d.decision.upper(), "operator": d.operator, "at": time.time()})
        c.say("operator", f"Rescuer {d.decision.upper()} {d.candidate_id}.")
        return {**out, "evidence": c.evidence.get(d.candidate_id)}

    @app.post("/camera/observation")
    async def camera_observation(o: CameraObs):
        c = core()
        c.engine.camera_observation(o.person_visible, angle_deg=o.angle_deg)
        c.say("operator", f"Camera at {c.engine.pan_deg:+.0f} deg: {'person visible' if o.person_visible else 'nobody visible'}.")
        return {"ok": True}

    @app.post("/estop")
    async def estop(e: EStop):
        c = core()
        mode = c.engine.set_estop(e.engaged)
        c.say("alert" if e.engaged else "system", "E-STOP ENGAGED: turret frozen." if e.engaged else "E-stop released.")
        return {"ok": True, "mode": mode}

    @app.post("/reset")
    async def reset():
        c = core()
        c.engine.reset()
        c.evidence.clear()
        c.move_suggestion = None
        c.say("system", "Belief map reset to prior (audit chain kept).")
        return {"ok": True}

    @app.get("/grid")
    async def grid():
        return JSONResponse(core().state())

    @app.get("/camera/snapshot")
    async def camera_snapshot():
        body, ctype = await core().pi.snapshot()
        return Response(content=body, media_type=ctype, headers={"Cache-Control": "no-store"})

    @app.get("/audit/verify")
    async def audit_verify():
        return core().audit.verify()

    @app.get("/audit/recent")
    async def audit_recent(n: int = 50):
        return core().audit.recent(n)

    @app.get("/evidence/{candidate_id}")
    async def evidence(candidate_id: str):
        card = core().evidence.get(candidate_id)
        if card is None:
            raise HTTPException(404, f"no evidence card for {candidate_id}")
        return card

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        c = core()
        await ws.accept()
        c.clients.add(ws)
        try:
            await ws.send_json(c.state())
            while True:
                await ws.receive_text()  # keepalive / ignore client messages
        except WebSocketDisconnect:
            pass
        finally:
            c.clients.discard(ws)

    return app


app = create_app()  # for: uvicorn main:app


def main() -> None:
    ap = argparse.ArgumentParser(description="AeroEcho Core server")
    ap.add_argument("--pi", default=os.getenv("AEROECHO_PI_URL", "mock"), help='"mock" or http://<pi-ip>:5000')
    ap.add_argument("--wifi", default=os.getenv("AEROECHO_WIFI", "off"), choices=["off", "ghop"])
    ap.add_argument("--ghop", default=C.GHOP_WS_URL, help="ghop-ghop WebSocket URL")
    ap.add_argument("--host", default=C.CORE_HOST)
    ap.add_argument("--port", type=int, default=C.CORE_PORT)
    ap.add_argument("--audit", default=C.AUDIT_LOG_PATH)
    a = ap.parse_args()
    s = Settings(pi_url=a.pi, wifi=a.wifi, ghop_ws=a.ghop, audit_path=a.audit)
    print(f"\n  AeroEcho Core  |  turret: {s.pi_url}  |  wifi: {s.wifi}\n"
          f"  Dashboard: http://127.0.0.1:{a.port}   (phone: http://<laptop-ip>:{a.port})\n  Ctrl+C to stop\n")
    uvicorn.run(create_app(s), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
