"""CLI: fetch the per-scene Sentinel-2 COGs picked by `discover_aerial`.

Reads scene picks from the per-AOI S2 manifest, fetches each (idempotent),
and updates the manifest.

    uv run python -m terra_query.ingest.cli.fetch_sentinel2
    uv run python -m terra_query.ingest.cli.fetch_sentinel2 --experiment /path/to/cfg.yaml
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
    s2_cog,
    s2_manifest,
)
from terra_query.ingest.aerial import (
    S2_COLLECTION,
    StacPick,
    aoi_bounds_buffered_26916,
    fetch_sentinel2_scene,
)


def _date_only(iso: str) -> str:
    return iso[:10]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_from_scene(scene: dict) -> StacPick:
    return StacPick(
        collection=S2_COLLECTION,
        item_id=scene["item_id"],
        datetime=scene["datetime"],
        cloud_cover=scene.get("cloud_cover"),
        assets=scene.get("assets", {}),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
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

    manifest_path = s2_manifest(aoi_id)
    manifest = json.loads(manifest_path.read_text())
    for scene in manifest["scenes"]:
        date_str = _date_only(scene["datetime"])
        out = s2_cog(aoi_id, date_str)
        pick = _pick_from_scene(scene)
        print(f"[S2 {date_str}] {pick.item_id} -> {out.relative_to(REPO_ROOT)}")
        t0 = time.monotonic()
        urls, skipped = fetch_sentinel2_scene(pick, out, bounds)
        elapsed = time.monotonic() - t0
        scene["assets"] = urls
        if skipped:
            if not scene.get("fetched_at"):
                scene["fetched_at"] = datetime.fromtimestamp(
                    out.stat().st_mtime, tz=timezone.utc
                ).isoformat()
        else:
            scene["fetched_at"] = _now_utc()
            scene["fetch_elapsed_s"] = round(elapsed, 2)
        print(f"  {'skipped (valid COG)' if skipped else 'wrote COG'}; {elapsed:.2f}s")

    manifest["last_fetch_at"] = _now_utc()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
