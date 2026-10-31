"""[MOCK] AeroEcho perception receiver -- NOT Member 3's real FastAPI server.

Purpose: let Member 2 develop and test the audio publisher without waiting for
Member 3. It accepts the agreed six-field payload, REJECTS anything else
(extra or missing keys -> HTTP 422), prints what it received, and acks.

[MOCK] The path and reply shape follow CONTRACT.md section 3 (POST /ingest ->
{"ok": true, "record_id": ...}), but this is NOT Member 3's server, so nothing
here proves integration with the real Core.

Accepted: the six agreed fields, plus OPTIONAL `schema_version` (int), because
CONTRACT.md s3 says "optional but send it" and its own s6 curl example sends it.
Still rejected (422) on purpose: class_probs, snr_db, tdoa_us, mic_channels and
any other key. Those are not yet approved for the audio sender, so this mock is
a tripwire against emitting them before Member 3 agrees. It is NOT evidence that
the real Core rejects them.

Run:
    uvicorn publisher.mock_server:app --port 8000
"""

from __future__ import annotations

import json
import uuid
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

MOCK_ENDPOINT_PATH = "/ingest"  # per CONTRACT.md s3

app = FastAPI(title="[MOCK] AeroEcho perception receiver")

# Kept in memory so tests can inspect what arrived. [MOCK] only.
RECEIVED: List[dict] = []


class PerceptionPayload(BaseModel):
    """Mirror of the agreed contract. extra='forbid' makes unknown keys a 422."""

    model_config = ConfigDict(extra="forbid")

    sensor_id: StrictStr
    timestamp: float
    detect_score: float
    bearing_deg: Optional[float]  # required key, but null allowed (CONTRACT.md s3)
    listening_quality: float
    primary_class: StrictStr
    schema_version: Optional[StrictInt] = None  # optional per CONTRACT.md s3


@app.post(MOCK_ENDPOINT_PATH)
def receive(payload: PerceptionPayload) -> dict:
    data = payload.model_dump(exclude_unset=True)  # record only what was sent
    RECEIVED.append(data)
    print("[MOCK] received:", json.dumps(data))
    return {"ok": True, "record_id": f"mock-{uuid.uuid4().hex[:8]}", "mock": True}


@app.get("/mock/received")
def received() -> dict:
    return {"mock": True, "count": len(RECEIVED), "items": RECEIVED}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
