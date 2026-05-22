"""Tests for /healthz (no DB, no model load).

Uses TERRA_QUERY_SKIP_MODEL_LOAD=1 + lifespan + TestClient so the
lifespan still runs (sets app.state.cfg, .storage, etc.) without
loading 1+ GB of CLIP weights.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from terra_query.api.app import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("TERRA_QUERY_SKIP_MODEL_LOAD", "1")
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_healthz_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["embed_dim"] > 0
    assert isinstance(body["model_id"], str) and body["model_id"]
    assert isinstance(body["experiment_id"], str) and body["experiment_id"]


def test_search_503_when_model_skipped(client):
    r = client.post("/search", json={"text": "waterfall", "top_k": 5})
    assert r.status_code == 503, r.text


def test_search_validation_empty_text(client):
    r = client.post("/search", json={"text": "", "top_k": 5})
    assert r.status_code == 422


def test_cors_allows_configured_origin(client):
    r = client.get(
        "/healthz", headers={"Origin": "http://localhost:5173"}
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_blocks_unconfigured_origin(client):
    r = client.get("/healthz", headers={"Origin": "http://evil.example"})
    # request still 200, but no CORS header is granted to a non-allowed origin
    assert r.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in r.headers}
