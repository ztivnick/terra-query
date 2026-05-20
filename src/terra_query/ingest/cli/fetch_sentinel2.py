"""CLI for the Sentinel-2 ingest step (S3 BUILD step 5).

Reads scene picks from the S2 manifest, fetches each (idempotent), and
updates the manifest. Run as:

    uv run python -m terra_query.ingest.cli.fetch_sentinel2
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone

from terra_query.core.paths import (
    AOI_WGS84,
    REPO_ROOT,
    S2_MANIFEST,
    s2_cog,
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
    aoi = json.loads(AOI_WGS84.read_text())
    bounds = aoi_bounds_buffered_26916(aoi)
    print(f"AOI 26916 buffered bounds: {bounds}")

    manifest = json.loads(S2_MANIFEST.read_text())
    for scene in manifest["scenes"]:
        date_str = _date_only(scene["datetime"])
        out = s2_cog(date_str)
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
    S2_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
