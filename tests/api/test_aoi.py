"""Tests for /aoi endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from terra_query.api.app import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("TERRA_QUERY_SKIP_MODEL_LOAD", "1")
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_aoi_returns_geojson(client):
    r = client.get("/aoi")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["aoi_id"]
    gj = body["geojson"]
    assert gj["type"] == "FeatureCollection"
    assert gj["features"], "AOI must have at least one feature"
    geom = gj["features"][0]["geometry"]
    assert geom["type"] == "Polygon"
    # Bond Falls block: ring lives near lon -89, lat 46
    ring = geom["coordinates"][0]
    lon, lat = ring[0][:2]
    assert -90 < lon < -88
    assert 46 < lat < 47
