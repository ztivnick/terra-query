"""CLI: fetch the per-cycle NAIP COGs picked by `discover_aerial`.

Reads the per-cycle picks from the per-AOI NAIP manifest, fetches each
one (idempotent), and updates the manifest with asset URLs and fetch
timestamps. AOI is resolved via the experiment config's `aoi_id`.

    uv run python -m terra_query.ingest.cli.fetch_naip                # all cycles
    uv run python -m terra_query.ingest.cli.fetch_naip --year 2022
    uv run python -m terra_query.ingest.cli.fetch_naip --experiment /path/to/cfg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from terra_query.core import config
from terra_query.core.paths import (
    REPO_ROOT,
    aoi_wgs84,
    naip_cog,
    naip_manifest,
)
from terra_query.ingest.aerial import (
    NAIP_COLLECTION,
    StacPick,
    aoi_bounds_buffered_26916,
    fetch_naip_cycle,
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _picks_from_cycle(cycle: dict) -> list[StacPick]:
    return [
        StacPick(
            collection=NAIP_COLLECTION,
            item_id=it["item_id"],
            datetime=it["datetime"],
            cloud_cover=None,
            assets=it.get("assets", {}),
        )
        for it in cycle["items"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="only fetch this cycle year")
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    args = parser.parse_args()
    cfg = config.load_experiment(args.experiment)
    aoi_id = config.aoi_id_of(cfg)

    aoi = json.loads(aoi_wgs84(aoi_id).read_text())
    bounds = aoi_bounds_buffered_26916(aoi)
    print(f"AOI id: {aoi_id}")
    print(f"AOI 26916 buffered bounds: {bounds}")

    manifest_path = naip_manifest(aoi_id)
    manifest = json.loads(manifest_path.read_text())
    cycles = manifest["cycles"]
    if args.year:
        cycles = [c for c in cycles if c["year"] == args.year]
        if not cycles:
            print(f"no cycle for year {args.year} in manifest", file=sys.stderr)
            return 1

    for cycle in cycles:
        year = cycle["year"]
        out = naip_cog(aoi_id, year)
        picks = _picks_from_cycle(cycle)
        print(f"[NAIP {year}] {len(picks)} items -> {out.relative_to(REPO_ROOT)}")
        t0 = time.monotonic()
        urls, skipped = fetch_naip_cycle(picks, out, bounds)
        elapsed = time.monotonic() - t0
        for it in cycle["items"]:
            if it["item_id"] not in urls:
                continue
            it["assets"] = {"image": urls[it["item_id"]]}
            if skipped:
                if not it.get("fetched_at"):
                    it["fetched_at"] = datetime.fromtimestamp(
                        out.stat().st_mtime, tz=timezone.utc
                    ).isoformat()
            else:
                it["fetched_at"] = _now_utc()
        if not skipped:
            cycle["fetch_elapsed_s"] = round(elapsed, 2)
        print(f"  {'skipped (valid COG)' if skipped else 'wrote COG'}; {elapsed:.2f}s")

    # Re-save manifest (with updated assets / fetched_at)
    manifest["last_fetch_at"] = _now_utc()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
