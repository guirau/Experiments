"""Unit tests for the FastAPI trigger service (api.py).

No network: we test the HTTP surface + the running-guard, not the live job runs (those call
the same discover/enrich functions covered elsewhere).
"""

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_status_shape():
    body = client.get("/status").json()
    assert set(body) == {"discover", "enrich"}
    for job in body.values():
        assert {"status", "result", "error", "started_at", "finished_at"} <= set(job)


def test_discover_returns_409_when_already_running():
    api.JOBS["discover"] = {"status": "running", "result": None, "error": None,
                            "started_at": "x", "finished_at": None}
    try:
        assert client.post("/discover", json={}).status_code == 409
    finally:
        api.JOBS["discover"] = api._idle()


def test_enrich_returns_409_when_already_running():
    api.JOBS["enrich"] = {"status": "running", "result": None, "error": None,
                          "started_at": "x", "finished_at": None}
    try:
        assert client.post("/enrich").status_code == 409
    finally:
        api.JOBS["enrich"] = api._idle()
