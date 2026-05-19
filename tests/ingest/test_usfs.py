"""Tests for USFS boundary ingest functions."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import responses as responses_lib
from shapely.geometry import box, mapping

from terra_query.ingest.usfs import fetch_usfs_boundary, save_usfs

_BOUNDS = (-89.13, 46.38, -89.099, 46.409)

# minimal polygon that covers the Ottawa NF test AOI
_MOCK_POLY = box(*_BOUNDS)

_MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": mapping(_MOCK_POLY),
            "properties": {"FORESTNAME": "Ottawa National Forest", "OBJECTID": 1},
        }
    ],
}

_MOCK_URL = (
    "https://apps.fs.usda.gov/arcgis/rest/services/EDW"
    "/EDW_ForestSystemBoundaries_01/MapServer/0/query"
)


@responses_lib.activate
def test_fetch_usfs_boundary_returns_polygon() -> None:
    """fetch_usfs_boundary returns a single valid polygon in EPSG:26916."""
    responses_lib.add(
        responses_lib.GET,
        _MOCK_URL,
        json=_MOCK_GEOJSON,
        status=200,
    )

    gdf = fetch_usfs_boundary("Ottawa National Forest")

    assert not gdf.empty
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 26916
    assert all(gdf.geometry.is_valid), "returned geometries are not valid"
    assert len(gdf) == 1


@responses_lib.activate
def test_fetch_usfs_boundary_warns_on_empty(caplog: pytest.LogCaptureFixture) -> None:
    """fetch_usfs_boundary warns when no features returned."""
    import logging

    empty_geojson = {"type": "FeatureCollection", "features": []}
    responses_lib.add(
        responses_lib.GET,
        _MOCK_URL,
        json=empty_geojson,
        status=200,
    )

    with caplog.at_level(logging.WARNING, logger="terra_query.ingest.usfs"):
        gdf = fetch_usfs_boundary()

    assert gdf.empty
    assert any("no features" in msg.lower() for msg in caplog.messages)


def test_save_usfs_writes_parquet() -> None:
    """save_usfs writes a readable GeoParquet file."""
    gdf = gpd.GeoDataFrame(
        {"FORESTNAME": ["Ottawa National Forest"]},
        geometry=[_MOCK_POLY],
        crs="EPSG:26916",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "ottawa_nf_boundary.parquet"
        result = save_usfs(gdf, out_path)

        assert result == out_path
        assert out_path.exists()

        loaded = gpd.read_parquet(out_path)
        assert len(loaded) == 1
        assert "FORESTNAME" in loaded.columns


def test_save_usfs_dry_run() -> None:
    """save_usfs dry_run=True does not create the file."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[_MOCK_POLY], crs="EPSG:26916")
    out_path = Path("/tmp/usfs_dry_run.parquet")
    save_usfs(gdf, out_path, dry_run=True)
    assert not out_path.exists()
