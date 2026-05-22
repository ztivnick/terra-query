"""CLI: build the NAIP chip index, eval-chip PNGs, and grid-overview PNG.

Idempotent: each output gets regenerated only if it is missing or older than
its inputs. Re-run safely after a NAIP refresh, an eval-set edit, or a code
change.

    uv run python -m terra_query.ingest.cli.build_chip_index
    uv run python -m terra_query.ingest.cli.build_chip_index --force
    uv run python -m terra_query.ingest.cli.build_chip_index --experiment /path/to/cfg.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shapely.geometry import shape

from terra_query.core import config
from terra_query.core.paths import (
    aoi_26916,
    chip_eval,
    chip_eval_dir,
    chip_grid_overview_png,
    chip_index_json,
    eval_26916,
    naip_cog,
    naip_manifest,
)
from terra_query.ingest.chips import (
    ChipBox,
    assemble_chip_index,
    render_chip_grid_overview,
    render_eval_chip_png,
)


def _newer(a: Path, b: Path) -> bool:
    """True if a is missing or older than b (by mtime)."""
    if not a.exists():
        return True
    return a.stat().st_mtime < b.stat().st_mtime


def _load_inputs(aoi_id: str, eval_set_id: str):
    aoi_gj = json.loads(aoi_26916(aoi_id).read_text())
    aoi_poly = shape(aoi_gj["features"][0]["geometry"])
    aoi_ring = aoi_gj["features"][0]["geometry"]["coordinates"][0]

    ev_gj = json.loads(eval_26916(eval_set_id).read_text())
    ev_pts: list[tuple[str, tuple[float, float]]] = []
    for f in ev_gj["features"]:
        x, y = f["geometry"]["coordinates"]
        ev_pts.append((f["properties"]["id"], (float(x), float(y))))

    manifest = json.loads(naip_manifest(aoi_id).read_text())
    cycle_cogs = [(c["year"], naip_cog(aoi_id, c["year"])) for c in manifest["cycles"]]

    # AOI bounds in 26916
    xs = [p[0] for p in aoi_ring]
    ys = [p[1] for p in aoi_ring]
    aoi_bounds = (min(xs), min(ys), max(xs), max(ys))

    # buffered AOI / COG extent: read off any one cycle's COG (they all share it)
    import rasterio

    with rasterio.open(cycle_cogs[0][1]) as ds:
        buffered = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    return aoi_poly, aoi_ring, ev_pts, cycle_cogs, aoi_bounds, buffered


def _newest_input_mtime(
    aoi_id: str,
    eval_set_id: str,
    cycle_cogs: list[tuple[str, Path]],
) -> float:
    inputs = [
        aoi_26916(aoi_id),
        eval_26916(eval_set_id),
        naip_manifest(aoi_id),
    ] + [c for _, c in cycle_cogs]
    return max(p.stat().st_mtime for p in inputs)


def build_index_if_stale(
    experiment_id: str,
    aoi_id: str,
    eval_set_id: str,
    chip_size_m: int,
    stride_m: int,
    force: bool = False,
) -> dict:
    aoi_poly, _aoi_ring, ev_pts, cycle_cogs, aoi_bounds, buffered = _load_inputs(
        aoi_id, eval_set_id,
    )

    out_path = chip_index_json(experiment_id)
    rebuild = force or not out_path.exists()
    if not rebuild:
        rebuild = out_path.stat().st_mtime < _newest_input_mtime(
            aoi_id, eval_set_id, cycle_cogs,
        )

    if rebuild:
        idx = assemble_chip_index(
            aoi_polygon_26916=aoi_poly,
            eval_points_26916=ev_pts,
            naip_cycle_cogs=cycle_cogs,
            aoi_bounds_26916=aoi_bounds,
            aoi_buffered_bounds_26916=buffered,
            chip_size_m=chip_size_m,
            stride_m=stride_m,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(idx, indent=2))
        print(f"wrote {out_path} ({sum(len(c['chips']) for c in idx['cycles'])} chips)")
    else:
        idx = json.loads(out_path.read_text())
        print(f"chip index up to date: {out_path}")
    return idx


def _chip_box_from_record(ch: dict) -> ChipBox:
    return ChipBox(
        row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1]
    )


def _overview_cycle(idx: dict) -> str:
    """Latest cycle in the chip index (used for rendering)."""
    return sorted(c["year"] for c in idx["cycles"])[-1]


def render_eval_chips(
    experiment_id: str,
    aoi_id: str,
    eval_set_id: str,
    idx: dict,
    force: bool = False,
) -> None:
    by_year = {c["year"]: c for c in idx["cycles"]}
    overview_year = _overview_cycle(idx)
    overview_cycle = by_year[overview_year]
    chips_by_id = {ch["chip_id"]: ch for ch in overview_cycle["chips"]}

    ev_gj = json.loads(eval_26916(eval_set_id).read_text())
    ev_pts_by_id = {
        f["properties"]["id"]: tuple(f["geometry"]["coordinates"]) for f in ev_gj["features"]
    }

    cog = naip_cog(aoi_id, overview_year)
    chip_eval_dir(experiment_id).mkdir(parents=True, exist_ok=True)
    for fid, entries in idx["eval_lookup"].items():
        chip_id_latest = next(e["chip_id"] for e in entries if e["year"] == overview_year)
        ch_rec = chips_by_id[chip_id_latest]
        out = chip_eval(experiment_id, fid)
        if not force and out.exists() and out.stat().st_mtime >= cog.stat().st_mtime:
            continue
        render_eval_chip_png(
            chip=_chip_box_from_record(ch_rec),
            source_cog_path=cog,
            feature_point_26916=ev_pts_by_id[fid],
            out_path=out,
        )
        print(f"wrote {out}")


def render_overview(
    experiment_id: str,
    aoi_id: str,
    eval_set_id: str,
    idx: dict,
    force: bool = False,
) -> None:
    overview_year = _overview_cycle(idx)
    cog = naip_cog(aoi_id, overview_year)
    out = chip_grid_overview_png(experiment_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not force and out.exists():
        if out.stat().st_mtime >= max(
            cog.stat().st_mtime, chip_index_json(experiment_id).stat().st_mtime
        ):
            print(f"grid overview up to date: {out}")
            return

    by_year = {c["year"]: c for c in idx["cycles"]}
    grid = [_chip_box_from_record(ch) for ch in by_year[overview_year]["chips"]]
    aoi_gj = json.loads(aoi_26916(aoi_id).read_text())
    aoi_ring = aoi_gj["features"][0]["geometry"]["coordinates"][0]
    ev_gj = json.loads(eval_26916(eval_set_id).read_text())
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
    parser = argparse.ArgumentParser(description="Build the chip index + verification PNGs.")
    parser.add_argument("--force", action="store_true", help="Regenerate every artifact.")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    args = parser.parse_args()

    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    aoi_id = config.aoi_id_of(cfg)
    eval_set_id = config.eval_set_id_of(cfg)
    cp = config.chip_params_of(cfg)

    idx = build_index_if_stale(
        experiment_id=experiment_id,
        aoi_id=aoi_id,
        eval_set_id=eval_set_id,
        chip_size_m=cp["chip_size_m"],
        stride_m=cp["stride_m"],
        force=args.force,
    )
    render_eval_chips(experiment_id, aoi_id, eval_set_id, idx, force=args.force)
    render_overview(experiment_id, aoi_id, eval_set_id, idx, force=args.force)


if __name__ == "__main__":
    main()
