"""Tests for AOI helper functions."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from terra_query.ingest.aoi import aoi_bounds_4326, aoi_bounds_26916, load_aoi

# Bond Falls AOI coordinates used in all tests
_POLY = Polygon(
    [(-89.13, 46.38), (-89.099, 46.38), (-89.099, 46.409), (-89.13, 46.409), (-89.13, 46.38)]
)


def _write_tmp_geojson(poly: Polygon) -> Path:
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    tmp = tempfile.NamedTemporaryFile(suffix=".geojson", delete=False)
    tmp.close()
    gdf.to_file(tmp.name, driver="GeoJSON")
    return Path(tmp.name)


def test_load_aoi_crs() -> None:
    """load_aoi reprojects to EPSG:26916."""
    path = _write_tmp_geojson(_POLY)
    try:
        gdf = load_aoi(path)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 26916
    finally:
        path.unlink()


def test_aoi_bounds_4326() -> None:
    """aoi_bounds_4326 returns values in expected lat/lon range."""
    path = _write_tmp_geojson(_POLY)
    try:
        w, s, e, n = aoi_bounds_4326(path)
        # values should be close to the polygon's own coords
        assert w == pytest.approx(-89.13, abs=1e-6)
        assert s == pytest.approx(46.38, abs=1e-6)
        assert e == pytest.approx(-89.099, abs=1e-6)
        assert n == pytest.approx(46.409, abs=1e-6)
    finally:
        path.unlink()


def test_aoi_bounds_26916() -> None:
    """aoi_bounds_26916 returns plausible UTM coords for UP Michigan."""
    path = _write_tmp_geojson(_POLY)
    try:
        minx, miny, maxx, maxy = aoi_bounds_26916(path)
        # Ottawa NF AOI: easting ~330000-345000, northing ~5135000-5140000
        assert 330000 < minx < 345000, f"minx out of range: {minx}"
        assert 330000 < maxx < 345000, f"maxx out of range: {maxx}"
        assert 5130000 < miny < 5145000, f"miny out of range: {miny}"
        assert 5130000 < maxy < 5145000, f"maxy out of range: {maxy}"
    finally:
        path.unlink()
