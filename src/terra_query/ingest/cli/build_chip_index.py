"""CLI: build the S4 chip index, eval-chip PNGs, and grid-overview PNG.

Idempotent: each output gets regenerated only if it is missing or older than
its inputs. Re-run safely after a NAIP refresh, an eval-set edit, or a code
change.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import shape

from terra_query.core.paths import (
    AOI_26916,
    CHIP_EVAL_DIR,
    CHIP_GRID_OVERVIEW_PNG,
    CHIP_INDEX_JSON,
    EVAL_26916,
    NAIP_MANIFEST,
    chip_eval,
    naip_cog,
)
from terra_query.ingest.chips import (
    ChipBox,
    assemble_chip_index,
    build_chip_grid,
    render_chip_grid_overview,
    render_eval_chip_png,
)

OVERVIEW_CYCLE = "2022"  # latest NAIP cycle used for visual artifacts


def _newer(a: Path, b: Path) -> bool:
    """True if a is missing or older than b (by mtime)."""
    if not a.exists():
        return True
    return a.stat().st_mtime < b.stat().st_mtime


def _load_inputs():
    aoi_gj = json.loads(AOI_26916.read_text())
    aoi_poly = shape(aoi_gj["features"][0]["geometry"])
    aoi_ring = aoi_gj["features"][0]["geometry"]["coordinates"][0]

    ev_gj = json.loads(EVAL_26916.read_text())
    ev_pts: list[tuple[str, tuple[float, float]]] = []
    for f in ev_gj["features"]:
        x, y = f["geometry"]["coordinates"]
        ev_pts.append((f["properties"]["id"], (float(x), float(y))))

    manifest = json.loads(NAIP_MANIFEST.read_text())
    cycle_cogs = [(c["year"], naip_cog(c["year"])) for c in manifest["cycles"]]

    # AOI bounds in 26916
    xs = [p[0] for p in aoi_ring]
    ys = [p[1] for p in aoi_ring]
    aoi_bounds = (min(xs), min(ys), max(xs), max(ys))

    # buffered AOI / COG extent: read off any one cycle's COG (they all share it)
    import rasterio

    with rasterio.open(cycle_cogs[0][1]) as ds:
        buffered = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    return aoi_poly, aoi_ring, ev_pts, cycle_cogs, aoi_bounds, buffered


def _newest_input_mtime(cycle_cogs: list[tuple[str, Path]]) -> float:
    inputs = [AOI_26916, EVAL_26916, NAIP_MANIFEST] + [c for _, c in cycle_cogs]
    return max(p.stat().st_mtime for p in inputs)


def build_index_if_stale(force: bool = False) -> dict:
    aoi_poly, _aoi_ring, ev_pts, cycle_cogs, aoi_bounds, buffered = _load_inputs()

    rebuild = force or not CHIP_INDEX_JSON.exists()
    if not rebuild:
        rebuild = CHIP_INDEX_JSON.stat().st_mtime < _newest_input_mtime(cycle_cogs)

    if rebuild:
        idx = assemble_chip_index(
            aoi_polygon_26916=aoi_poly,
            eval_points_26916=ev_pts,
            naip_cycle_cogs=cycle_cogs,
            aoi_bounds_26916=aoi_bounds,
            aoi_buffered_bounds_26916=buffered,
        )
        CHIP_INDEX_JSON.parent.mkdir(parents=True, exist_ok=True)
        CHIP_INDEX_JSON.write_text(json.dumps(idx, indent=2))
        print(f"wrote {CHIP_INDEX_JSON} ({sum(len(c['chips']) for c in idx['cycles'])} chips)")
    else:
        idx = json.loads(CHIP_INDEX_JSON.read_text())
        print(f"chip index up to date: {CHIP_INDEX_JSON}")
    return idx


def _chip_box_from_record(ch: dict) -> ChipBox:
    return ChipBox(
        row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1]
    )


def render_eval_chips(idx: dict, force: bool = False) -> None:
    by_year = {c["year"]: c for c in idx["cycles"]}
    overview_cycle = by_year[OVERVIEW_CYCLE]
    chips_by_id = {ch["chip_id"]: ch for ch in overview_cycle["chips"]}

    ev_gj = json.loads(EVAL_26916.read_text())
    ev_pts_by_id = {
        f["properties"]["id"]: tuple(f["geometry"]["coordinates"]) for f in ev_gj["features"]
    }

    cog = naip_cog(OVERVIEW_CYCLE)
    CHIP_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for fid, entries in idx["eval_lookup"].items():
        chip_id_2022 = next(e["chip_id"] for e in entries if e["year"] == OVERVIEW_CYCLE)
        ch_rec = chips_by_id[chip_id_2022]
        out = chip_eval(fid)
        if not force and out.exists() and out.stat().st_mtime >= cog.stat().st_mtime:
            continue
        render_eval_chip_png(
            chip=_chip_box_from_record(ch_rec),
            source_cog_path=cog,
            feature_point_26916=ev_pts_by_id[fid],
            out_path=out,
        )
        print(f"wrote {out}")


def render_overview(idx: dict, force: bool = False) -> None:
    cog = naip_cog(OVERVIEW_CYCLE)
    out = CHIP_GRID_OVERVIEW_PNG
    if not force and out.exists():
        if out.stat().st_mtime >= max(cog.stat().st_mtime, CHIP_INDEX_JSON.stat().st_mtime):
            print(f"grid overview up to date: {out}")
            return

    by_year = {c["year"]: c for c in idx["cycles"]}
    grid = [_chip_box_from_record(ch) for ch in by_year[OVERVIEW_CYCLE]["chips"]]
    aoi_gj = json.loads(AOI_26916.read_text())
    aoi_ring = aoi_gj["features"][0]["geometry"]["coordinates"][0]
    ev_gj = json.loads(EVAL_26916.read_text())
    ev_pts = [
        (f["properties"]["id"], tuple(f["geometry"]["coordinates"]))
        for f in ev_gj["features"]
    ]
    render_chip_grid_overview(
        grid=grid,
        source_cog_path=cog,
        aoi_polygon_26916_ring=aoi_ring,
        eval_points_26916=ev_pts,
        out_path=out,
    )
    print(f"wrote {out}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build S4 chip index + verification PNGs.")
    parser.add_argument("--force", action="store_true", help="Regenerate every artifact.")
    args = parser.parse_args()

    idx = build_index_if_stale(force=args.force)
    render_eval_chips(idx, force=args.force)
    render_overview(idx, force=args.force)


if __name__ == "__main__":
    main()
