import json

import pytest
from pyproj import Geod
from shapely.geometry import Polygon

from terra_query.core.crs import (
    WORKING_CRS_EPSG,
    area_of_use_bounds,
    area_of_use_covers,
    reproject_polygon,
    to_wgs84,
    to_working,
    working_crs,
)
from terra_query.core.paths import AOI_WGS84, EVAL_WGS84


@pytest.fixture(scope="module")
def aoi():
    return json.loads(AOI_WGS84.read_text())


@pytest.fixture(scope="module")
def eval_fc():
    return json.loads(EVAL_WGS84.read_text())


def _aoi_rings(aoi):
    return aoi["features"][0]["geometry"]["coordinates"]


def _aoi_bbox(aoi):
    ring = _aoi_rings(aoi)[0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def test_working_crs_is_projected_metric():
    crs = working_crs()
    assert crs.to_epsg() == WORKING_CRS_EPSG
    assert crs.is_projected
    for axis in crs.axis_info[:2]:
        assert axis.unit_name in ("metre", "meter"), f"axis {axis.name} unit {axis.unit_name}"
    bounds = area_of_use_bounds()
    assert len(bounds) == 4
    assert all(isinstance(b, float) for b in bounds)


def test_area_of_use_strictly_covers_aoi(aoi):
    bbox = _aoi_bbox(aoi)
    assert area_of_use_covers(bbox)
    w, s, e, n = bbox
    aw, as_, ae, an = area_of_use_bounds()
    assert w - aw > 0
    assert s - as_ > 0
    assert ae - e > 0
    assert an - n > 0


def test_round_trip_under_1e7_deg(aoi, eval_fc):
    fwd = to_working()
    back = to_wgs84()
    coords: list[tuple[float, float]] = []
    for ring in _aoi_rings(aoi):
        coords.extend((p[0], p[1]) for p in ring)
    for feat in eval_fc["features"]:
        c = feat["geometry"]["coordinates"]
        coords.append((c[0], c[1]))
    max_err = 0.0
    for lon, lat in coords:
        x, y = fwd.transform(lon, lat)
        rl, rt = back.transform(x, y)
        err = max(abs(rl - lon), abs(rt - lat))
        max_err = max(max_err, err)
    assert max_err < 1e-7, f"max round-trip error {max_err:.2e} deg"


def test_aoi_area_in_working_crs_matches_s1(aoi):
    rings_proj = reproject_polygon(_aoi_rings(aoi))
    poly = Polygon(rings_proj[0], rings_proj[1:])
    area_km2 = poly.area / 1_000_000
    assert 24.875 <= area_km2 <= 25.125, f"AOI area {area_km2:.4f} km^2 outside [24.875, 25.125]"


def test_euclidean_in_working_matches_geod(eval_fc):
    feats = {f["properties"]["id"]: f for f in eval_fc["features"]}
    a = feats["bond-falls"]["geometry"]["coordinates"]
    b = feats["unnamed-pond-e-of-flowage"]["geometry"]["coordinates"]
    fwd = to_working()
    ax, ay = fwd.transform(a[0], a[1])
    bx, by = fwd.transform(b[0], b[1])
    euc = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    geod = Geod(ellps="GRS80")
    _, _, ell = geod.inv(a[0], a[1], b[0], b[1])
    rel = abs(euc - ell) / ell
    assert rel < 0.001, f"relative diff {rel:.4%}, euc={euc:.3f} m, ell={ell:.3f} m"
