"""AeroEcho fusion engine: the agent loop between /ingest and the search map.

Owns, per contracts/CONTRACT.md:
  * Probe cycle      MOVING -> SETTLING -> SCANNING -> SCAN_DONE (suggest next stop)
                     Records are only used once the probe is placed and settled.
  * Late fusion      S_master = (Qa*Sa + Qw*Sw) / (Qa + Qw + eps); stale -> Q = 0; EMA smoothing.
  * Safety tiers     Tier 0 PAN / MOVE_PROBE (auto or advisory), Tier 1 DWELL (auto),
                     Tier 2 CANDIDATE (halts, waits for rescuer APPROVE / REJECT / DEFER;
                     no answer in HITL_TIMEOUT_S -> HOLD).
  * E-stop           no turret commands while engaged; sensing continues.
  * Mono fallback    mic_channels = 1: bearings ignored, per-stop mean SNR of detections
                     localises by loudness, candidate box widens to 1.8 m.

Pure Python + the BayesGrid: no I/O. main.py feeds it records and clock ticks and
executes the Actions it returns. Every method takes `now` (Unix seconds) so tests
can drive time explicitly. Events for the audit chain go to `on_event`.
"""
from __future__ import annotations

import itertools
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import config as C
from bayes_grid import BayesGrid

# Modes (dashboard + gatekeeper)
SEARCHING, AWAITING_HITL, HOLD, ESTOP = "SEARCHING", "AWAITING_HITL", "HOLD", "ESTOP"
# Probe states
UNPLACED, MOVING, SETTLING, SCANNING, SCAN_DONE = "UNPLACED", "MOVING", "SETTLING", "SCANNING", "SCAN_DONE"
DECISIONS = ("APPROVE", "REJECT", "DEFER")


@dataclass
class Action:
    action_id: str
    timestamp: float
    type: str  # PAN | MOVE_PROBE | DWELL | CANDIDATE
    tier: int
    payload: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FusionState:
    timestamp: float = 0.0
    s_master: float = 0.0  # EMA-smoothed
    s_master_raw: float = 0.0
    s_audio: float = 0.0
    q_audio: float = 0.0
    s_wifi: float = 0.0
    q_wifi: float = 0.0
    degraded: bool = True  # both modalities stale or Q = 0
    consecutive_high: int = 0


@dataclass
class IngestResult:
    record_id: str
    used: bool
    reason: str  # "ok" or why it was discarded
    state: FusionState
    actions: list[Action] = field(default_factory=list)


@dataclass
class Candidate:
    candidate_id: str
    created_at: float
    row: int
    col: int
    x_m: float
    y_m: float
    region_mass: float
    s_master: float
    turret_angle_deg: float
    probe_xy: tuple[float, float]
    evidence_record_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _modality(sensor_id: str) -> str:
    if sensor_id == C.SENSOR_AUDIO:
        return "audio"
    if sensor_id == C.SENSOR_WIFI:
        return "wifi"
    raise ValueError(f"unknown sensor_id {sensor_id!r}")


class FusionEngine:
    def __init__(
        self,
        grid: BayesGrid | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        self.grid = grid or BayesGrid()
        self.on_event = on_event or (lambda kind, data: None)
        self._rec_ids = itertools.count(1)
        self._act_ids = itertools.count(1)
        self._cand_ids = itertools.count(1)

        self.mode = SEARCHING
        self.state = FusionState(timestamp=now)
        self.latest: dict[str, tuple[float, float, float]] = {}  # modality -> (timestamp, score, q)
        self.recent_used_ids: deque[str] = deque(maxlen=C.EVIDENCE_IDS_KEEP)

        self.probe_state = UNPLACED
        self.probe_waypoint: str | None = None
        self.placed_at: float | None = None
        self.scan_done_at: float | None = None
        self.dwell_extensions = 0
        self.move_suggested = False

        self.pending: Candidate | None = None
        self.hitl_deadline: float | None = None
        self.deferred: list[tuple[int, int, float]] = []  # (row, col, until)
        self.found: list[dict[str, Any]] = []

        self.stop_index = 0
        self._stop_snr: list[float] = []
        self.stops_heard: set[int] = set()  # stops where audio detections were used

        self.pan_deg = 0.0
        self.last_pan_at = -math.inf
        self.estop_engaged = False
        self._mode_before_estop = SEARCHING

        if C.AUTO_PLACE_AT_START:
            self.probe_placed(now, waypoint=C.START_WAYPOINT)

    # ------------------------------------------------------------- helpers
    def _action(self, now: float, type_: str, tier: int, payload: dict[str, Any], reason: str) -> Action:
        a = Action(f"a-{next(self._act_ids):06d}", now, type_, tier, payload, reason)
        self.on_event("ACTION", a.to_dict())
        return a

    def _set_mode(self, mode: str, now: float, why: str) -> None:
        if mode != self.mode:
            self.on_event("MODE", {"from": self.mode, "to": mode, "reason": why, "timestamp": now})
            self.mode = mode

    # ---------------------------------------------------------- probe cycle
    def probe_moving(self, now: float) -> None:
        """Operator picked the probe up: stop using evidence until it is placed again."""
        self.probe_state = MOVING
        self.on_event("PROBE", {"event": "moving", "timestamp": now})

    def probe_placed(
        self,
        now: float,
        waypoint: str | None = None,
        row: int | None = None,
        col: int | None = None,
        x_m: float | None = None,
        y_m: float | None = None,
    ) -> dict[str, Any]:
        """Probe set down. Exactly one of: waypoint name, grid cell (row, col), or metres (x_m, y_m)."""
        given = [waypoint is not None, row is not None or col is not None, x_m is not None or y_m is not None]
        if sum(given) != 1:
            raise ValueError("give exactly one of waypoint, (row, col) or (x_m, y_m)")
        if waypoint is not None:
            if waypoint not in self.grid.waypoints:
                raise ValueError(f"unknown waypoint {waypoint!r}")
            x, y = self.grid.set_probe_waypoint(waypoint)
        elif row is not None or col is not None:
            if row is None or col is None:
                raise ValueError("row and col are both required")
            x, y = self.grid.set_probe_cell(int(row), int(col))
        else:
            if x_m is None or y_m is None:
                raise ValueError("x_m and y_m are both required")
            if self.grid.world_to_cell(float(x_m), float(y_m)) is None:
                raise ValueError(f"({x_m}, {y_m}) is outside the search area")
            x, y = float(x_m), float(y_m)
            self.grid.set_probe(x, y)
        self.probe_waypoint = self._nearest_waypoint(x, y)
        self.probe_state = SETTLING
        self.placed_at = now
        self.scan_done_at = now + max(C.PROBE_DWELL_S, C.WIFI_SETTLE_S + C.SCAN_WINDOW_S)
        self.dwell_extensions = 0
        self.move_suggested = False
        self.latest.clear()  # readings from the previous spot no longer describe this one
        self.stop_index += 1
        self._stop_snr = []
        info = {"event": "placed", "x_m": x, "y_m": y, "waypoint": self.probe_waypoint, "timestamp": now}
        self.on_event("PROBE", info)
        return info

    def _nearest_waypoint(self, x: float, y: float) -> str | None:
        for name, (wx, wy) in self.grid.waypoints.items():
            if math.hypot(wx - x, wy - y) <= 0.5:
                return name
        return None

    def _gate(self, modality: str, rec_ts: float, now: float) -> str:
        """'ok' or the reason a record is not used."""
        if self.probe_state in (UNPLACED, MOVING):
            return f"probe {self.probe_state.lower()}"
        settle = C.AUDIO_SETTLE_S if modality == "audio" else C.WIFI_SETTLE_S
        if now - self.placed_at < settle:
            return f"{modality} settling"
        if now - rec_ts > C.STALE_S:
            return "stale record"
        if self.placed_at is not None and rec_ts < self.placed_at:
            return "recorded before placement"
        return "ok"

    # --------------------------------------------------------------- clock
    def tick(self, now: float) -> list[Action]:
        """Advance timers. main.py calls this ~1 Hz and on every ingest."""
        actions: list[Action] = []
        if self.probe_state == SETTLING and now - self.placed_at >= C.AUDIO_SETTLE_S:
            self.probe_state = SCANNING
        if self.probe_state == SCANNING and now >= self.scan_done_at:
            self.probe_state = SCAN_DONE
        if self.probe_state == SCAN_DONE and not self.move_suggested and self.mode == SEARCHING:
            name, ig = self.grid.best_waypoint()
            x, y = self.grid.waypoints[name]
            actions.append(self._action(now, "MOVE_PROBE", 0, {"waypoint": name, "x_m": x, "y_m": y, "expected_info_gain_bits": round(ig, 4)},
                                        "scan window complete; max expected information gain"))
            self.move_suggested = True
        if self.mode == AWAITING_HITL and self.hitl_deadline is not None and now >= self.hitl_deadline:
            self._set_mode(HOLD, now, "no rescuer decision within timeout")
            self.on_event("HOLD_TIMEOUT", {"candidate_id": self.pending.candidate_id if self.pending else None, "timestamp": now})
        self.deferred = [d for d in self.deferred if d[2] > now]
        self._refresh_staleness(now)
        return actions

    # -------------------------------------------------------------- fusion
    def _refresh_staleness(self, now: float) -> None:
        st = self.state
        sa, qa = self._current("audio", now)
        sw, qw = self._current("wifi", now)
        st.s_audio, st.q_audio, st.s_wifi, st.q_wifi = sa, qa, sw, qw
        st.degraded = qa + qw <= 0.0
        st.timestamp = now

    def _current(self, modality: str, now: float) -> tuple[float, float]:
        if modality not in self.latest:
            return 0.0, 0.0
        ts, s, q = self.latest[modality]
        if now - ts > C.STALE_S:
            return s, 0.0
        return s, q

    def _fuse(self, now: float) -> None:
        self._refresh_staleness(now)
        st = self.state
        raw = (st.q_audio * st.s_audio + st.q_wifi * st.s_wifi) / (st.q_audio + st.q_wifi + C.QUALITY_EPS)
        st.s_master_raw = 0.0 if st.degraded else raw
        st.s_master = C.EMA_ALPHA * st.s_master_raw + (1.0 - C.EMA_ALPHA) * st.s_master
        st.consecutive_high = st.consecutive_high + 1 if st.s_master > C.T2_THRESHOLD else 0

    # -------------------------------------------------------------- ingest
    def ingest(self, rec: Mapping[str, Any], now: float | None = None) -> IngestResult:
        """One PerceptionRecord (already schema-validated). Returns what happened and any actions."""
        now = time.time() if now is None else now
        modality = _modality(rec["sensor_id"])
        record_id = f"r-{next(self._rec_ids):06d}"
        actions = self.tick(now)
        reason = self._gate(modality, float(rec["timestamp"]), now)
        used = reason == "ok"
        if used:
            s, q = float(rec["detect_score"]), float(rec["listening_quality"])
            self.latest[modality] = (float(rec["timestamp"]), s, q)
            if modality == "audio":
                self.grid.audio_mono = int(rec.get("mic_channels", 2)) == 1
                bearing = None if self.grid.audio_mono else rec.get("bearing_deg")
                self.grid.update_audio(s, q, bearing)
                if s >= 0.5:
                    self.stops_heard.add(self.stop_index)
                if rec.get("snr_db") is not None and s >= 0.5:  # loudness of detections at this stop
                    self._stop_snr.append(float(rec["snr_db"]))
                    self.grid.set_stop_level(self.stop_index, self.grid.probe_xy, sum(self._stop_snr) / len(self._stop_snr))
            else:
                rng = rec.get("range_m") if self.probe_waypoint == C.WIFI_CALIBRATED_AT else None
                self.grid.update_wifi(s, q, rng, rec.get("range_sigma_m") if rng is not None else None)
            self.recent_used_ids.append(record_id)
            self._fuse(now)
            actions += self.decide(now)
        self.on_event("INGEST", {"record_id": record_id, "used": used, "reason": reason, "record": dict(rec), "s_master": round(self.state.s_master, 4)})
        return IngestResult(record_id, used, reason, self.state, actions)

    # ------------------------------------------------------------ decisions
    def decide(self, now: float) -> list[Action]:
        """Gatekeeper: pick at most one tier-2/1 action plus an optional PAN."""
        actions: list[Action] = []
        if self.mode in (AWAITING_HITL, HOLD):
            return actions
        st = self.state
        row, col, mass = self.grid.best_region(self.candidate_radius)

        # Tier 2: candidate -> halt and ask the rescuer.
        min_stops = C.T2_MIN_STOPS_MONO if self.grid.audio_mono else C.T2_MIN_STOPS
        if (st.consecutive_high >= C.T2_CONSECUTIVE and mass >= C.T2_MIN_REGION_P
                and len(self.stops_heard) >= min_stops and not self._suppressed(row, col)):
            x, y = self.grid.cell_to_world(row, col)
            angle = self.grid.turret_angle_to(row, col)
            self.pending = Candidate(f"c-{next(self._cand_ids):04d}", now, row, col, x, y, round(mass, 4),
                                     round(st.s_master, 4), angle, self.grid.probe_xy, list(self.recent_used_ids))
            self.hitl_deadline = now + C.HITL_TIMEOUT_S
            if not self.estop_engaged:
                actions.append(self._pan(now, angle, "point camera at candidate for verification", force=True))
            actions.append(self._action(now, "CANDIDATE", 2, self.pending.to_dict(),
                                        f"S_master {st.s_master:.2f} > {C.T2_THRESHOLD} x{st.consecutive_high}, region mass {mass:.2f}"))
            self.on_event("CANDIDATE", self.pending.to_dict())
            self._set_mode(AWAITING_HITL, now, "tier 2 candidate")
            return [a for a in actions if a is not None]

        # Tier 1: interesting but not conclusive -> listen longer here.
        if (st.s_master > C.T1_THRESHOLD and self.dwell_extensions < C.MAX_DWELL_EXTENSIONS
                and self.probe_state in (SCANNING, SCAN_DONE)):
            self.dwell_extensions += 1
            self.scan_done_at = max(self.scan_done_at, now) + C.DWELL_S
            if self.probe_state == SCAN_DONE:
                self.probe_state, self.move_suggested = SCANNING, False
            actions.append(self._action(now, "DWELL", 1, {"seconds": C.DWELL_S, "scan_done_at": self.scan_done_at},
                                        f"S_master {st.s_master:.2f} > {C.T1_THRESHOLD}"))

        # Tier 0: keep the camera on the most likely region.
        if not self.estop_engaged:
            angle, _ = self.grid.best_pan(self.pan_deg)
            pan = self._pan(now, angle, "max P(victim in view)")
            if pan is not None:
                actions.append(pan)
        return actions

    def _pan(self, now: float, angle: float, reason: str, force: bool = False) -> Action | None:
        angle = float(min(max(round(angle / C.PAN_STEP_DEG) * C.PAN_STEP_DEG, C.PAN_MIN_DEG), C.PAN_MAX_DEG))
        if not force and (angle == self.pan_deg or now - self.last_pan_at < C.PAN_MIN_INTERVAL_S):
            return None
        self.pan_deg, self.last_pan_at = angle, now
        return self._action(now, "PAN", 0, {"angle_deg": angle}, reason)

    @property
    def candidate_radius(self) -> int:
        return C.CANDIDATE_RADIUS_CELLS_MONO if self.grid.audio_mono else C.CANDIDATE_RADIUS_CELLS

    def _suppressed(self, row: int, col: int) -> bool:
        r = self.candidate_radius
        near = lambda a, b: abs(a[0] - b[0]) <= r and abs(a[1] - b[1]) <= r
        if any(near((row, col), (d[0], d[1])) for d in self.deferred):
            return True
        return any(near((row, col), (f["row"], f["col"])) for f in self.found)

    # --------------------------------------------------------- human input
    def hitl_decision(self, candidate_id: str, decision: str, now: float | None = None, operator: str = "", note: str = "") -> dict[str, Any]:
        now = time.time() if now is None else now
        decision = decision.upper()
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if self.pending is None or self.pending.candidate_id != candidate_id:
            raise ValueError(f"no pending candidate {candidate_id!r}")
        c = self.pending
        if decision == "APPROVE":
            self.grid.mark_found(c.row, c.col, self.candidate_radius)
            self.found.append({**c.to_dict(), "approved_at": now, "operator": operator})
        elif decision == "REJECT":
            self.grid.penalize_region(c.row, c.col, radius=self.candidate_radius)
        else:
            self.deferred.append((c.row, c.col, now + C.DEFER_COOLDOWN_S))
        self.on_event("HITL_DECISION", {"candidate_id": candidate_id, "decision": decision, "operator": operator, "note": note, "timestamp": now})
        self.pending, self.hitl_deadline = None, None
        self.state.consecutive_high = 0
        self._set_mode(ESTOP if self.estop_engaged else SEARCHING, now, f"rescuer {decision}")
        return {"ok": True, "mode": self.mode, "found": len(self.found)}

    def camera_observation(self, person_visible: bool, now: float | None = None, angle_deg: float | None = None) -> None:
        """Operator looked at the latest snapshot: person visible or not (camera as a sensor)."""
        now = time.time() if now is None else now
        angle = self.pan_deg if angle_deg is None else float(angle_deg)
        self.grid.apply_camera_observation(angle, person_visible)
        self.on_event("CAMERA_OBSERVATION", {"angle_deg": angle, "person_visible": person_visible, "timestamp": now})

    def set_estop(self, engaged: bool, now: float | None = None) -> str:
        now = time.time() if now is None else now
        if engaged and not self.estop_engaged:
            self._mode_before_estop = self.mode
            self.estop_engaged = True
            self._set_mode(ESTOP, now, "E-stop engaged")
        elif not engaged and self.estop_engaged:
            self.estop_engaged = False
            back = self._mode_before_estop
            if back == ESTOP or (back in (AWAITING_HITL, HOLD) and self.pending is None):
                back = SEARCHING
            self._set_mode(back, now, "E-stop released")
        return self.mode

    def pan_confirmed(self, angle_deg: float, now: float | None = None) -> None:
        """Pi reported the turret's real angle (after /pan or /health)."""
        self.pan_deg = float(angle_deg)

    def reset(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self.grid.reset()
        self.state = FusionState(timestamp=now)
        self.latest.clear()
        self.pending, self.hitl_deadline = None, None
        self.deferred.clear()
        self.found.clear()
        self.stops_heard.clear()
        self._set_mode(ESTOP if self.estop_engaged else SEARCHING, now, "reset")
        self.on_event("RESET", {"timestamp": now})

    # ------------------------------------------------------------ snapshot
    def snapshot(self, now: float | None = None, include_grid: bool = True) -> dict[str, Any]:
        """StateSnapshot for WS /ws/state and GET /grid."""
        now = time.time() if now is None else now
        self._refresh_staleness(now)
        row, col, mass = self.grid.best_region(self.candidate_radius)
        bx, by = self.grid.cell_to_world(row, col)
        wp_scores = self.grid.waypoint_info_gain()
        nxt = max(wp_scores, key=wp_scores.get) if wp_scores else None
        st = self.state
        snap = {
            "timestamp": now,
            "mode": self.mode,
            "estop": self.estop_engaged,
            "fusion": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(st).items()},
            "probe": {
                "state": self.probe_state,
                "x_m": self.grid.probe_xy[0],
                "y_m": self.grid.probe_xy[1],
                "waypoint": self.probe_waypoint,
                "placed_at": self.placed_at,
                "audio_live_at": None if self.placed_at is None else self.placed_at + C.AUDIO_SETTLE_S,
                "wifi_live_at": None if self.placed_at is None else self.placed_at + C.WIFI_SETTLE_S,
                "scan_done_at": self.scan_done_at,
                "visits": dict(self.grid.visits),
            },
            "next_waypoint": {"name": nxt, "expected_info_gain_bits": round(wp_scores[nxt], 4) if nxt else None,
                              "all": {k: round(v, 4) for k, v in wp_scores.items()}},
            "best_region": {"row": row, "col": col, "x_m": bx, "y_m": by, "mass": round(mass, 4),
                            "size_m": (2 * self.candidate_radius + 1) * C.CELL_M},
            "audio_mode": "mono" if self.grid.audio_mono else "stereo",
            "stops_heard": len(self.stops_heard),
            "p_absent": round(self.grid.p_absent(), 4),
            "entropy_bits": round(self.grid.entropy(), 4),
            "pan_deg": self.pan_deg,
            "pending_candidate": self.pending.to_dict() if self.pending else None,
            "hitl_deadline": self.hitl_deadline,
            "found": list(self.found),
        }
        if include_grid:
            snap["grid"] = self.grid.to_list()
        return snap
