"""Reproject S1 vector artifacts into the working CRS and write a verification report.

Reads the AOI + eval set from `human_authored/`, writes their 26916
counterparts to `pipeline_outputs/`, and a verification report to
`verification/`. Idempotent: rerunning produces byte-identical output.

    uv run python -m terra_query.ingest.cli.reproject_inputs
"""

from __future__ import annotations

import json

from shapely.geometry import Polygon

from terra_query.core.crs import (
    WORKING_CRS,
    WORKING_CRS_EPSG,
    area_of_use_bounds,
    reproject_polygon,
    to_wgs84,
    to_working,
)
from terra_query.core.paths import (
    AOI_26916,
    AOI_WGS84,
    CRS_VERIFICATION,
    EVAL_26916,
    EVAL_WGS84,
    REPO_ROOT,
)

CRS_BLOCK_26916 = {
    "type": "name",
    "properties": {"name": f"urn:ogc:def:crs:EPSG::{WORKING_CRS_EPSG}"},
}


def _round_coord(v: float, ndigits: int = 3) -> float:
    return round(v, ndigits)


def reproject_aoi(fc: dict) -> tuple[dict, float, tuple[float, float, float, float]]:
    feat = fc["features"][0]
    rings_wgs = feat["geometry"]["coordinates"]
    rings_proj = reproject_polygon(rings_wgs)
    rings_proj = [[[_round_coord(x), _round_coord(y)] for x, y in r] for r in rings_proj]
    poly = Polygon(rings_proj[0], rings_proj[1:])
    area_m2 = poly.area
    new_props = dict(feat["properties"])
    new_props["area_m2_projected"] = round(area_m2, 3)
    new_props["working_crs"] = WORKING_CRS
    new_feat = {
        "type": "Feature",
        "properties": new_props,
        "geometry": {"type": "Polygon", "coordinates": rings_proj},
    }
    out_fc = {
        "type": "FeatureCollection",
        "name": fc.get("name"),
        "crs": CRS_BLOCK_26916,
        "features": [new_feat],
    }
    return out_fc, area_m2, poly.bounds


def reproject_eval(fc: dict) -> dict:
    fwd = to_working()
    new_features = []
    for feat in fc["features"]:
        lon, lat = feat["geometry"]["coordinates"][:2]
        x, y = fwd.transform(lon, lat)
        new_features.append(
            {
                "type": "Feature",
                "properties": dict(feat["properties"]),
                "geometry": {
                    "type": "Point",
                    "coordinates": [_round_coord(x), _round_coord(y)],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": fc.get("name"),
        "crs": CRS_BLOCK_26916,
        "features": new_features,
    }


def max_round_trip(fc_aoi: dict, fc_eval: dict) -> tuple[float, int, int]:
    fwd = to_working()
    back = to_wgs84()
    max_err = 0.0
    n_vert = 0
    for ring in fc_aoi["features"][0]["geometry"]["coordinates"]:
        for lon, lat in ring:
            x, y = fwd.transform(lon, lat)
            rl, rt = back.transform(x, y)
            max_err = max(max_err, abs(rl - lon), abs(rt - lat))
            n_vert += 1
    n_pts = 0
    for f in fc_eval["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        x, y = fwd.transform(lon, lat)
        rl, rt = back.transform(x, y)
        max_err = max(max_err, abs(rl - lon), abs(rt - lat))
        n_pts += 1
    return max_err, n_vert, n_pts


def aoi_bbox_wgs84(fc: dict) -> tuple[float, float, float, float]:
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def write_report(
    aoi_bbox_wgs: tuple[float, float, float, float],
    aoi_bbox_proj: tuple[float, float, float, float],
    area_m2: float,
    max_err: float,
    n_vert: int,
    n_pts: int,
) -> str:
    aou = area_of_use_bounds()
    s1_area_km2 = 25.0
    rel = abs(area_m2 / 1e6 - s1_area_km2) / s1_area_km2
    text = (
        "terra-query working CRS verification\n"
        "====================================\n"
        "\n"
        f"Pinned CRS                  : {WORKING_CRS} (NAD83 / UTM Zone 16N, meters)\n"
        "\n"
        f"Area of use (WGS84 W,S,E,N) : {aou}\n"
        f"AOI bbox    (WGS84 W,S,E,N) : {aoi_bbox_wgs}\n"
        f"AOI bbox    (26916 E,N,E,N) : ({aoi_bbox_proj[0]:.3f}, {aoi_bbox_proj[1]:.3f}, "
        f"{aoi_bbox_proj[2]:.3f}, {aoi_bbox_proj[3]:.3f})\n"
        "\n"
        f"AOI area (26916)            : {area_m2:.3f} m^2 ({area_m2 / 1e6:.5f} km^2)\n"
        f"AOI area (S1 haversine)     : {s1_area_km2} km^2\n"
        f"Relative diff               : {rel:.4%}\n"
        "\n"
        f"Round-trip 4326 -> 26916 -> 4326 across {n_vert} AOI vertices "
        f"+ {n_pts} eval points:\n"
        f"Max coordinate error        : {max_err:.2e} deg (tolerance 1e-7)\n"
        "\n"
        "Artifacts:\n"
        f"- {AOI_26916.relative_to(REPO_ROOT)}\n"
        f"- {EVAL_26916.relative_to(REPO_ROOT)}\n"
    )
    CRS_VERIFICATION.parent.mkdir(parents=True, exist_ok=True)
    CRS_VERIFICATION.write_text(text)
    return text


def main() -> None:
    fc_aoi = json.loads(AOI_WGS84.read_text())
    fc_eval = json.loads(EVAL_WGS84.read_text())

    out_aoi, area_m2, bbox_proj = reproject_aoi(fc_aoi)
    out_eval = reproject_eval(fc_eval)
    err, n_vert, n_pts = max_round_trip(fc_aoi, fc_eval)
    bbox_wgs = aoi_bbox_wgs84(fc_aoi)

    AOI_26916.parent.mkdir(parents=True, exist_ok=True)
    EVAL_26916.parent.mkdir(parents=True, exist_ok=True)
    AOI_26916.write_text(json.dumps(out_aoi, indent=2) + "\n")
    EVAL_26916.write_text(json.dumps(out_eval, indent=2) + "\n")

    print(write_report(bbox_wgs, bbox_proj, area_m2, err, n_vert, n_pts))


if __name__ == "__main__":
    main()
