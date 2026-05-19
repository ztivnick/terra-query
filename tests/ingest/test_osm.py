"""Tests for OSM Overpass ingest functions."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import responses as responses_lib

from terra_query.ingest.osm import (
    build_overpass_query,
    fetch_osm_features,
    save_osm,
)

_BOUNDS = (-89.13, 46.38, -89.099, 46.409)
_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# minimal Overpass JSON: one node, one closed-way building, one open-way road
_MOCK_RESPONSE = {
    "elements": [
        {
            "type": "node",
            "id": 1001,
            "lat": 46.395,
            "lon": -89.115,
            "tags": {"natural": "tree"},
        },
        {
            "type": "way",
            "id": 2001,
            "tags": {"highway": "track"},
            "geometry": [
                {"lat": 46.390, "lon": -89.120},
                {"lat": 46.392, "lon": -89.110},
                {"lat": 46.394, "lon": -89.115},
            ],
        },
        {
            "type": "way",
            "id": 2002,
            "tags": {"building": "cabin"},
            "geometry": [
                {"lat": 46.395, "lon": -89.110},
                {"lat": 46.396, "lon": -89.110},
                {"lat": 46.396, "lon": -89.109},
                {"lat": 46.395, "lon": -89.109},
                {"lat": 46.395, "lon": -89.110},  # closed
            ],
        },
    ]
}


def test_build_overpass_query_contains_bbox() -> None:
    """build_overpass_query includes south,west,north,east bbox in output."""
    w, s, e, n = _BOUNDS
    query = build_overpass_query(_BOUNDS, ["highway"])
    assert f"{s},{w},{n},{e}" in query
    assert "highway" in query
    assert "[out:json]" in query


def test_build_overpass_query_all_element_types() -> None:
    """build_overpass_query generates node/way/relation for each tag."""
    query = build_overpass_query(_BOUNDS, ["natural", "highway"])
    for el in ["node", "way", "relation"]:
        assert el in query


@responses_lib.activate
def test_fetch_osm_features_crs_and_geometry_types() -> None:
    """fetch_osm_features returns GeoDataFrame in EPSG:26916 with Point/Line/Polygon."""
    responses_lib.add(
        responses_lib.POST,
        _OVERPASS_URL,
        json=_MOCK_RESPONSE,
        status=200,
    )

    gdf = fetch_osm_features(_BOUNDS, tags=["highway", "natural", "building"])

    assert not gdf.empty
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 26916
    geom_types = set(gdf.geometry.geom_type.unique())
    # expect at least point and linestring from our mock
    assert "Point" in geom_types
    assert "LineString" in geom_types
    assert "Polygon" in geom_types


@responses_lib.activate
def test_fetch_osm_features_warns_on_empty(caplog: pytest.LogCaptureFixture) -> None:
    """fetch_osm_features warns when Overpass returns zero elements."""
    import logging

    responses_lib.add(
        responses_lib.POST,
        _OVERPASS_URL,
        json={"elements": []},
        status=200,
    )

    with caplog.at_level(logging.WARNING, logger="terra_query.ingest.osm"):
        gdf = fetch_osm_features(_BOUNDS)

    assert gdf.empty
    assert any("no elements" in msg.lower() for msg in caplog.messages)


def test_save_osm_writes_parquet() -> None:
    """save_osm writes a readable GeoParquet file."""
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"osm_id": [1], "osm_type": ["node"], "natural": ["tree"]},
        geometry=[Point(-89.115, 46.395)],
        crs="EPSG:26916",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "osm_aoi.parquet"
        result = save_osm(gdf, out_path)

        assert result == out_path
        assert out_path.exists()

        loaded = gpd.read_parquet(out_path)
        assert len(loaded) == 1


def test_save_osm_dry_run() -> None:
    """save_osm dry_run=True does not create the file."""
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)], crs="EPSG:26916")
    out_path = Path("/tmp/osm_dry_run.parquet")
    save_osm(gdf, out_path, dry_run=True)
    assert not out_path.exists()
