"""AeroEcho Evidence & Audit Agent: append-only, SHA-256 hash-chained event log.

    H_n = SHA256( H_{n-1} || canonical_json(entry_n without "hash") )
    canonical_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    genesis H_0 = "0" * 64

Each line of the JSONL file is one entry:
    {"seq", "timestamp", "event_type", "data", "prev_hash", "hash"}
Editing, deleting or reordering any line breaks every hash after it, and verify() reports
the first broken sequence number.

Run standalone to check a log:  python audit_logger.py audit_log.jsonl
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any

import config as C


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def _default(o: Any) -> Any:
    """Make numpy scalars / tuples / sets JSON-safe."""
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(o: Any) -> Any:
    """Round-trip through JSON so stored data equals what is hashed (floats, NaN, numpy)."""
    def fix(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {str(k): fix(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        return v
    return json.loads(canonical_json(fix(o)))


def entry_hash(prev_hash: str, entry_wo_hash: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(entry_wo_hash)).encode("utf-8")).hexdigest()


class AuditLogger:
    # INGEST events are very frequent (2-3 Hz); keep them out of the in-memory tail but in the file.
    QUIET = {"INGEST"}

    def __init__(self, path: str = C.AUDIT_LOG_PATH, resume: bool = True) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.seq = 0
        self.last_hash = C.GENESIS_HASH
        self.tail: deque[dict[str, Any]] = deque(maxlen=200)  # recent non-INGEST entries for the UI
        self.by_candidate: dict[str, list[dict[str, Any]]] = {}
        if resume and os.path.exists(path):
            ok, n, last, msg = verify_file(path)
            if not ok:
                # Never append to a broken chain: rotate it aside and start fresh.
                os.replace(path, f"{path}.broken-{int(time.time())}")
            else:
                self.seq, self.last_hash = n, last
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def log(self, event_type: str, data: dict[str, Any] | None = None, timestamp: float | None = None) -> dict[str, Any]:
        with self._lock:
            entry = {
                "seq": self.seq + 1,
                "timestamp": round(time.time() if timestamp is None else float(timestamp), 3),
                "event_type": str(event_type),
                "data": _clean(data or {}),
                "prev_hash": self.last_hash,
            }
            entry["hash"] = entry_hash(self.last_hash, entry)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(canonical_json(entry) + "\n")
            self.seq, self.last_hash = entry["seq"], entry["hash"]
            if event_type not in self.QUIET:
                self.tail.append(entry)
            cid = (entry["data"] or {}).get("candidate_id")
            if cid:
                self.by_candidate.setdefault(cid, []).append(entry)
            return entry

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        """So it can be passed straight to FusionEngine(on_event=...)."""
        self.log(event_type, data)

    def verify(self) -> dict[str, Any]:
        ok, n, last, msg = verify_file(self.path)
        return {"ok": ok, "entries": n, "head_hash": last, "message": msg}

    def recent(self, n: int = 50) -> list[dict[str, Any]]:
        return list(self.tail)[-n:]

    def evidence_card(self, candidate: dict[str, Any], decision: dict[str, Any] | None = None) -> dict[str, Any]:
        """Deterministic summary a rescuer can verify: what fired, from which records, the chain head."""
        cid = candidate.get("candidate_id")
        events = self.by_candidate.get(cid, [])
        basis_text = {
            "acoustic": "Acoustic bearings from multiple probe stops intersect here",
            "acoustic_mono": "Loudness compared across probe stops (mono audio, coarse)",
            "wifi_breathing": "Wi-Fi CSI breathing detected at multiple stops, no voice (coarse)",
        }.get(candidate.get("basis", "acoustic"), candidate.get("basis", ""))
        size_m = (2 * int(candidate.get("radius_cells", C.CANDIDATE_RADIUS_CELLS)) + 1) * C.CELL_M
        return {
            "candidate_id": cid,
            "location_m": {"x": candidate.get("x_m"), "y": candidate.get("y_m")},
            "region_size_m": round(size_m, 2),
            "region_probability": candidate.get("region_mass"),
            "s_master": candidate.get("s_master"),
            "basis": basis_text,
            "camera_angle_deg": candidate.get("turret_angle_deg"),
            "probe_at_alert_m": candidate.get("probe_xy"),
            "evidence_record_ids": candidate.get("evidence_record_ids", []),
            "decision": decision,
            "audit_entries": [{"seq": e["seq"], "event_type": e["event_type"], "hash": e["hash"][:16]} for e in events],
            "chain_head": self.last_hash,
            "chain_length": self.seq,
        }


def verify_file(path: str) -> tuple[bool, int, str, str]:
    """(ok, entries, last_hash, message). Recomputes every hash from genesis."""
    prev = C.GENESIS_HASH
    n = 0
    if not os.path.exists(path):
        return True, 0, prev, "no log yet"
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                return False, n, prev, f"line {lineno}: not JSON"
            if e.get("seq") != n + 1:
                return False, n, prev, f"line {lineno}: sequence {e.get('seq')} expected {n + 1} (deleted or reordered)"
            if e.get("prev_hash") != prev:
                return False, n, prev, f"seq {e['seq']}: prev_hash does not match (chain broken)"
            body = {k: v for k, v in e.items() if k != "hash"}
            if entry_hash(prev, body) != e.get("hash"):
                return False, n, prev, f"seq {e['seq']}: hash mismatch (entry was edited)"
            prev, n = e["hash"], n + 1
    return True, n, prev, f"chain intact: {n} entries"


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else C.AUDIT_LOG_PATH
    ok, n, last, msg = verify_file(p)
    print(("PASS " if ok else "FAIL ") + msg)
    print("head:", last)
    sys.exit(0 if ok else 1)
