import json
import math

import pytest

from publisher.payload import PAYLOAD_FIELDS, make_payload, validate_payload

AGREED_KEYS = {
    "sensor_id",
    "timestamp",
    "detect_score",
    "bearing_deg",
    "listening_quality",
    "primary_class",
}


def _good(**overrides):
    kwargs = dict(
        sensor_id="laptop_earphone_mics",
        timestamp=1793442000.0,
        detect_score=0.85,
        bearing_deg=25.0,
        listening_quality=0.90,
        primary_class="Voice",
    )
    kwargs.update(overrides)
    return kwargs


def test_field_list_matches_contract():
    assert set(PAYLOAD_FIELDS) == AGREED_KEYS
    assert len(PAYLOAD_FIELDS) == 6


def test_exact_keys_and_types():
    p = make_payload(**_good())
    assert set(p.keys()) == AGREED_KEYS  # fails on any extra OR missing key
    assert isinstance(p["sensor_id"], str)
    assert isinstance(p["timestamp"], float)
    assert isinstance(p["detect_score"], float)
    assert isinstance(p["bearing_deg"], float)
    assert isinstance(p["listening_quality"], float)
    assert isinstance(p["primary_class"], str)


def test_matches_agreed_example_exactly():
    p = make_payload(**_good())
    assert p == {
        "sensor_id": "laptop_earphone_mics",
        "timestamp": 1793442000.0,
        "detect_score": 0.85,
        "bearing_deg": 25.0,
        "listening_quality": 0.90,
        "primary_class": "Voice",
    }


def test_json_roundtrip_keeps_keys_and_types():
    p = make_payload(**_good())
    back = json.loads(json.dumps(p))
    assert set(back.keys()) == AGREED_KEYS
    assert back == p


def test_int_inputs_become_float():
    p = make_payload(**_good(detect_score=1, bearing_deg=0, timestamp=1793442000))
    assert isinstance(p["detect_score"], float)
    assert isinstance(p["bearing_deg"], float)
    assert isinstance(p["timestamp"], float)


def test_default_timestamp_is_float_epoch():
    p = make_payload(**{k: v for k, v in _good().items() if k != "timestamp"})
    assert isinstance(p["timestamp"], float)
    assert p["timestamp"] > 1.6e9  # sanity: epoch seconds, not ms or monotonic


def test_null_bearing_is_allowed_per_contract():
    # CONTRACT.md s3: bearing_deg is null when the correlation peak is unreliable
    p = make_payload(**_good(bearing_deg=None))
    assert set(p.keys()) == AGREED_KEYS
    assert p["bearing_deg"] is None
    assert json.loads(json.dumps(p))["bearing_deg"] is None  # serialises as JSON null


def test_extra_kwarg_is_rejected():
    with pytest.raises(TypeError):
        make_payload(**_good(), class_probs={"Voice": 0.9})


@pytest.mark.parametrize("name", ["detect_score", "bearing_deg", "listening_quality", "primary_class"])
def test_missing_required_arg_is_rejected(name):
    kwargs = _good()
    kwargs.pop(name)
    with pytest.raises(TypeError):
        make_payload(**kwargs)


def test_validate_rejects_extra_key():
    p = make_payload(**_good())
    p["extra"] = 1
    with pytest.raises(ValueError, match="extra"):
        validate_payload(p)


@pytest.mark.parametrize("name", sorted(AGREED_KEYS))
def test_validate_rejects_missing_key(name):
    p = make_payload(**_good())
    del p[name]
    with pytest.raises(ValueError, match="missing"):
        validate_payload(p)


@pytest.mark.parametrize(
    "bad",
    [
        dict(detect_score="0.85"),
        dict(detect_score=None),
        dict(bearing_deg="25"),
        dict(bearing_deg=True),
        dict(listening_quality=True),
        dict(primary_class=5),
        dict(primary_class=""),
        dict(sensor_id=None),
        dict(timestamp="1793442000"),
    ],
)
def test_wrong_types_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        make_payload(**_good(**bad))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(value):
    with pytest.raises(ValueError):
        make_payload(**_good(bearing_deg=value))
