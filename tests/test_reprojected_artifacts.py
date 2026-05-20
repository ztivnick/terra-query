import json
from pathlib import Path

import pytest
from shapely.geometry import Point, Polygon

REPO = Path(__file__).resolve().parent.parent
AOI_PROJ = REPO / "data" / "aoi" / "bond_falls_block_26916.geojson"
EVAL_PROJ = REPO / "data" / "eval" / "known_features_26916.geojson"
REPORT = REPO / "data" / "crs_verification.txt"


def _declared_epsg(fc: dict) -> int:
    return int(fc["crs"]["properties"]["name"].split("::")[-1])


@pytest.fixture(scope="module")
def aoi_proj() -> dict:
    return json.loads(AOI_PROJ.read_text())


@pytest.fixture(scope="module")
def eval_proj() -> dict:
    return json.loads(EVAL_PROJ.read_text())


def test_aoi_projected_artifact_shape(aoi_proj):
    assert aoi_proj["type"] == "FeatureCollection"
    assert _declared_epsg(aoi_proj) == 26916
    assert len(aoi_proj["features"]) == 1
    feat = aoi_proj["features"][0]
    assert feat["geometry"]["type"] == "Polygon"
    ring = feat["geometry"]["coordinates"][0]
    eastings = [p[0] for p in ring]
    northings = [p[1] for p in ring]
    assert all(300_000 < e < 400_000 for e in eastings), f"easting out of range: {eastings}"
    assert all(5_100_000 < n < 5_200_000 for n in northings), f"northing out of range: {northings}"


def test_eval_projected_artifact_shape(eval_proj):
    assert eval_proj["type"] == "FeatureCollection"
    assert _declared_epsg(eval_proj) == 26916
    assert len(eval_proj["features"]) == 11
    for f in eval_proj["features"]:
        assert f["geometry"]["type"] == "Point"
        assert "id" in f["properties"]


def test_all_eval_points_inside_reprojected_aoi(aoi_proj, eval_proj):
    rings = aoi_proj["features"][0]["geometry"]["coordinates"]
    poly = Polygon(rings[0], rings[1:])
    for f in eval_proj["features"]:
        x, y = f["geometry"]["coordinates"][:2]
        assert poly.covers(Point(x, y)), f"{f['properties']['id']} outside reprojected AOI"


def test_verification_report_contents():
    text = REPORT.read_text()
    for needle in ("EPSG:26916", "Area of use", "AOI area", "Round-trip", "tolerance 1e-7"):
        assert needle in text, f"missing from report: {needle!r}"
