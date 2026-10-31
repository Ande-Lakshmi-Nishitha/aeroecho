"""[MOCK] server tests: they only show the mock enforces the agreed contract."""

import pytest
from fastapi.testclient import TestClient

from publisher.mock_server import MOCK_ENDPOINT_PATH, RECEIVED, app
from publisher.payload import make_payload

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear():
    RECEIVED.clear()


def _p():
    return make_payload(
        detect_score=0.85,
        bearing_deg=25.0,
        listening_quality=0.90,
        primary_class="Voice",
        timestamp=1793442000.0,
    )


def test_accepts_make_payload_output():
    r = client.post(MOCK_ENDPOINT_PATH, json=_p())
    assert r.status_code == 200
    assert len(RECEIVED) == 1
    assert RECEIVED[0] == _p()


def test_accepts_null_bearing():
    body = make_payload(
        detect_score=0.1,
        bearing_deg=None,
        listening_quality=0.3,
        primary_class="Machinery",
        timestamp=1793442000.0,
    )
    r = client.post(MOCK_ENDPOINT_PATH, json=body)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert RECEIVED[0]["bearing_deg"] is None


def test_rejects_extra_key():
    body = _p()
    body["class_probs"] = {"Voice": 0.9}
    assert client.post(MOCK_ENDPOINT_PATH, json=body).status_code == 422
    assert RECEIVED == []


def test_rejects_missing_key():
    body = _p()
    del body["bearing_deg"]
    assert client.post(MOCK_ENDPOINT_PATH, json=body).status_code == 422
    assert RECEIVED == []


def test_accepts_optional_schema_version():
    # CONTRACT.md s6 curl example sends schema_version: 1
    body = _p()
    body["schema_version"] = 1
    r = client.post(MOCK_ENDPOINT_PATH, json=body)
    assert r.status_code == 200
    assert RECEIVED[0] == body


def test_rejects_wrong_type_schema_version():
    body = _p()
    body["schema_version"] = "1"
    assert client.post(MOCK_ENDPOINT_PATH, json=body).status_code == 422


@pytest.mark.parametrize("key", ["snr_db", "mic_channels", "tdoa_us"])
def test_still_rejects_unapproved_extras(key):
    body = _p()
    body[key] = 1
    assert client.post(MOCK_ENDPOINT_PATH, json=body).status_code == 422
    assert RECEIVED == []


def test_rejects_wrong_type():
    body = _p()
    body["primary_class"] = 7
    assert client.post(MOCK_ENDPOINT_PATH, json=body).status_code == 422
