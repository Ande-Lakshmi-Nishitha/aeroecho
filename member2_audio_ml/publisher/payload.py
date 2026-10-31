"""AeroEcho audio perception payload builder (Member 2).

Implements the AGREED team interface contract. Do NOT add, rename, or retype
fields here without Member 3's approval (see the technical documentation file).

Agreed contract:
{
  "sensor_id": "laptop_earphone_mics",   # str
  "timestamp": 1793442000.0,             # float, epoch seconds
  "detect_score": 0.85,                  # float
  "bearing_deg": 25.0,                   # float
  "listening_quality": 0.90,             # float
  "primary_class": "Voice"               # str
}

Per CONTRACT.md section 3 (rev. 2026-11-01), bearing_deg is `null` when the
GCC-PHAT correlation peak is unreliable or the mic channels are duplicated, so
it is Optional[float] here. All other fields are required and non-null.

Deliberately NOT enforced here (not yet pinned down in the contract text this
module was written against; Member 3's schema/Core is the authority):
  * range of detect_score / listening_quality (0..1 per the formulas)
  * bearing_deg clamp to [-60, +60] (the sender's gcc_phat.py should clamp)
  * the allowed set of primary_class labels (Voice/Cough/Breathing/Machinery)
  * optional extras (schema_version, class_probs, snr_db, tdoa_us, mic_channels)
    -- NOT emitted by make_payload until the team decides; see docs update.
"""

from __future__ import annotations

import math
import time
from numbers import Real
from typing import Any, Dict, Optional

SENSOR_ID_DEFAULT = "laptop_earphone_mics"

# Order matches the agreed contract. The set is the single source of truth.
PAYLOAD_FIELDS = (
    "sensor_id",
    "timestamp",
    "detect_score",
    "bearing_deg",
    "listening_quality",
    "primary_class",
)

_FLOAT_FIELDS = ("timestamp", "detect_score", "listening_quality")
_NULLABLE_FLOAT_FIELDS = ("bearing_deg",)  # null = unreliable / no direction
_STR_FIELDS = ("sensor_id", "primary_class")


def _as_float(name: str, value: Any) -> float:
    # bool is a subclass of int; reject it so True does not silently become 1.0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        # NaN/inf are not valid JSON and would break the consumer
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Check a dict against the agreed contract. Raises on extra/missing keys,
    wrong types, or non-finite numbers. Returns the payload unchanged."""
    keys = set(payload.keys())
    expected = set(PAYLOAD_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing or extra:
        raise ValueError(
            f"payload keys do not match contract; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name in _FLOAT_FIELDS:
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(f"{name} must be float, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    for name in _NULLABLE_FLOAT_FIELDS:
        value = payload[name]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, float):
            raise TypeError(f"{name} must be float or None, got {type(value).__name__}")
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    for name in _STR_FIELDS:
        _as_str(name, payload[name])
    return payload


def make_payload(
    *,
    detect_score: float,
    bearing_deg: Optional[float],
    listening_quality: float,
    primary_class: str,
    timestamp: Optional[float] = None,
    sensor_id: str = SENSOR_ID_DEFAULT,
) -> Dict[str, Any]:
    """Build exactly the six agreed fields.

    Keyword-only on purpose: positional mix-ups between the three float scores
    would be silent bugs. Any unexpected keyword raises TypeError.
    `timestamp=None` uses time.time() (epoch seconds, float).
    `bearing_deg=None` means "no reliable direction" and is sent as JSON null.
    It is still a required argument so it cannot be omitted by accident.
    """
    if timestamp is None:
        timestamp = time.time()
    payload = {
        "sensor_id": _as_str("sensor_id", sensor_id),
        "timestamp": _as_float("timestamp", timestamp),
        "detect_score": _as_float("detect_score", detect_score),
        "bearing_deg": None if bearing_deg is None else _as_float("bearing_deg", bearing_deg),
        "listening_quality": _as_float("listening_quality", listening_quality),
        "primary_class": _as_str("primary_class", primary_class),
    }
    return validate_payload(payload)
