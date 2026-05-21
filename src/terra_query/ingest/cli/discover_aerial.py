"""STAC discovery + cycle selection for aerial ingest.

Lists available NAIP cycles and Sentinel-2 winter scenes for the AOI,
prints a summary, and writes per-source manifest files with the picks.
No downloads happen here. Run once via:

    uv run python -m terra_query.ingest.cli.discover_aerial
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from terra_query.core.paths import (
    AOI_WGS84,
    NAIP_MANIFEST,
    REPO_ROOT,
    S2_MANIFEST,
)
from terra_query.ingest.aerial import (
    NAIP_COLLECTION,
    PC_STAC_URL,
    S2_COLLECTION,
    StacPick,
    aoi_bounds_wgs84,
    search_naip,
    search_sentinel2_winter,
)

NAIP_CYCLES_TO_PICK = 6  # all available cycles for max retrieval signal
S2_SCENES_TO_PICK = 2


def _year(iso: str) -> str:
    return iso[:4]


def _winter_year(iso: str) -> int:
    y = int(iso[:4])
    m = int(iso[5:7])
    return y + 1 if m == 12 else y


def pick_naip_cycles(years_sorted: list[str], n: int) -> list[str]:
    """Pick up to n cycles with max temporal spread (latest, earliest, middle)."""
    if not years_sorted:
        return []
    if len(years_sorted) <= n:
        return list(years_sorted)
    if n == 1:
        return [years_sorted[-1]]
    if n == 2:
        return [years_sorted[0], years_sorted[-1]]
    picks = {years_sorted[0], years_sorted[-1]}
    remaining = n - len(picks)
    inner = years_sorted[1:-1]
    if inner and remaining > 0:
        step = max(1, len(inner) // (remaining + 1))
        for i in range(1, remaining + 1):
            idx = min(len(inner) - 1, i * step)
            picks.add(inner[idx])
    return sorted(picks)


def pick_s2_scenes(picks: list[StacPick], n: int) -> list[StacPick]:
    """Pick up to n scenes: lowest-cloud per winter, from the n most recent winters."""
    if not picks:
        return []
    groups: dict[int, list[StacPick]] = defaultdict(list)
    for p in picks:
        groups[_winter_year(p.datetime)].append(p)
    out: list[StacPick] = []
    for wy in sorted(groups.keys(), reverse=True):
        best = min(groups[wy], key=lambda p: p.cloud_cover or 100.0)
        out.append(best)
        if len(out) >= n:
            break
    return out


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _existing_naip_assets(existing: dict) -> dict[str, dict]:
    """item_id -> {'assets': ..., 'fetched_at': ...} from a prior manifest."""
    out: dict[str, dict] = {}
    for cycle in existing.get("cycles", []):
        for it in cycle.get("items", []):
            out[it["item_id"]] = {
                "assets": it.get("assets", {}),
                "fetched_at": it.get("fetched_at"),
            }
    return out


def _existing_s2_assets(existing: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sc in existing.get("scenes", []):
        out[sc["item_id"]] = {
            "assets": sc.get("assets", {}),
            "fetched_at": sc.get("fetched_at"),
        }
    return out


def _write_naip_manifest(path: Path, picks: list[StacPick], picked_years: list[str]) -> None:
    prior = _existing_naip_assets(_load_existing(path))
    by_year: dict[str, list[StacPick]] = defaultdict(list)
    for p in picks:
        by_year[_year(p.datetime)].append(p)
    cycles = []
    for y in picked_years:
        items = []
        for p in sorted(by_year[y], key=lambda x: x.item_id):
            entry = {
                "item_id": p.item_id,
                "datetime": p.datetime,
                "assets": prior.get(p.item_id, {}).get("assets") or p.assets,
            }
            if prior.get(p.item_id, {}).get("fetched_at"):
                entry["fetched_at"] = prior[p.item_id]["fetched_at"]
            items.append(entry)
        cycles.append({"year": y, "items": items})
    payload = {
        "generated_at": _now_utc(),
        "stac_endpoint": PC_STAC_URL,
        "collection": NAIP_COLLECTION,
        "picked_years": picked_years,
        "cycles": cycles,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_s2_manifest(path: Path, picks: list[StacPick]) -> None:
    prior = _existing_s2_assets(_load_existing(path))
    scenes = []
    for p in sorted(picks, key=lambda x: x.datetime):
        entry = {
            "item_id": p.item_id,
            "datetime": p.datetime,
            "cloud_cover": p.cloud_cover,
            "assets": prior.get(p.item_id, {}).get("assets") or p.assets,
        }
        if prior.get(p.item_id, {}).get("fetched_at"):
            entry["fetched_at"] = prior[p.item_id]["fetched_at"]
        scenes.append(entry)
    payload = {
        "generated_at": _now_utc(),
        "stac_endpoint": PC_STAC_URL,
        "collection": S2_COLLECTION,
        "scenes": scenes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    aoi = json.loads(AOI_WGS84.read_text())
    bbox = aoi_bounds_wgs84(aoi)
    print(f"AOI WGS84 bbox: ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f}, {bbox[3]:.4f})")
    print(f"STAC endpoint: {PC_STAC_URL}")
    print()

    print(f"[NAIP] searching {NAIP_COLLECTION}...")
    naip = search_naip(bbox)
    naip_by_year: dict[str, list[StacPick]] = defaultdict(list)
    for p in naip:
        naip_by_year[_year(p.datetime)].append(p)
    print(f"  found {len(naip)} items across {len(naip_by_year)} cycle(s)")
    for y in sorted(naip_by_year):
        print(f"    {y}: {len(naip_by_year[y])} item(s)")
    years_sorted = sorted(naip_by_year.keys())
    naip_picked = pick_naip_cycles(years_sorted, NAIP_CYCLES_TO_PICK)
    print(f"  picked cycles: {naip_picked}")
    naip_picked_items = [p for p in naip if _year(p.datetime) in naip_picked]
    _write_naip_manifest(NAIP_MANIFEST, naip_picked_items, naip_picked)
    print(f"  wrote {NAIP_MANIFEST.relative_to(REPO_ROOT)}")
    print()

    print(f"[S2] searching {S2_COLLECTION} (winter, cloud<{20})...")
    s2 = search_sentinel2_winter(bbox)
    print(f"  found {len(s2)} winter scenes under cloud cap")
    s2_by_winter: dict[int, list[StacPick]] = defaultdict(list)
    for p in s2:
        s2_by_winter[_winter_year(p.datetime)].append(p)
    for wy in sorted(s2_by_winter):
        ccs = sorted([p.cloud_cover for p in s2_by_winter[wy] if p.cloud_cover is not None])
        ccs_str = ", ".join(f"{c:.1f}" for c in ccs[:5])
        more = "" if len(ccs) <= 5 else f" (+{len(ccs) - 5} more)"
        print(f"    winter {wy}: {len(s2_by_winter[wy])} scenes; cloud%: [{ccs_str}]{more}")
    s2_picked = pick_s2_scenes(s2, S2_SCENES_TO_PICK)
    print(f"  picked scenes:")
    for p in s2_picked:
        print(f"    {p.datetime} cloud={p.cloud_cover:.1f}% id={p.item_id}")
    _write_s2_manifest(S2_MANIFEST, s2_picked)
    print(f"  wrote {S2_MANIFEST.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
