"""OSM feature ingest via Overpass API: query, parse, reproject, write GeoParquet.

NOTE: overpass-api.de and mirrors may rate-limit or block programmatic access
on some networks. If you get repeated 429/406, try the kumi.systems mirror or
wait before retrying. The --overpass-url CLI flag accepts alternate mirrors.
"""

import logging
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon

log = logging.getLogger(__name__)

_DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM feature tags to collect: roads/trails, buildings, natural features
DEFAULT_TAGS = [
    "highway",
    "building",
    "natural",
    "leisure",
    "landuse",
    "waterway",
]


def build_overpass_query(
    bounds_4326: tuple[float, float, float, float],
    tags: list[str],
) -> str:
    """Build an Overpass QL union query for all listed tag keys over the bbox.

    Overpass bbox order is: south,west,north,east.
    """
    w, s, e, n = bounds_4326
    bbox = f"{s},{w},{n},{e}"
    parts: list[str] = []
    for tag in tags:
        parts.append(f'node["{tag}"]({bbox});')
        parts.append(f'way["{tag}"]({bbox});')
        parts.append(f'relation["{tag}"]({bbox});')
    union = "\n  ".join(parts)
    return f"[out:json][timeout:60];\n(\n  {union}\n);\nout body geom;"


def _elements_to_geodataframe(elements: list[dict[str, object]]) -> gpd.GeoDataFrame:
    """Convert Overpass JSON elements to a GeoDataFrame.

    Handles nodes (Point), ways with geometry (LineString or Polygon),
    and skips relations (complex, not needed for P1).
    """
    rows: list[dict[str, object]] = []
    for el in elements:
        el_type = el.get("type")
        tags: dict[str, str] = el.get("tags", {})  # type: ignore[assignment]
        geom = None

        if el_type == "node" and "lat" in el and "lon" in el:
            geom = Point(el["lon"], el["lat"])

        elif el_type == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]  # type: ignore[attr-defined]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geom = Polygon(coords)
            elif len(coords) >= 2:
                geom = LineString(coords)

        if geom is None:
            continue

        row = {"osm_id": el.get("id"), "osm_type": el_type, **tags, "geometry": geom}
        rows.append(row)

    if not rows:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries(dtype="geometry"), crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    return gdf


def fetch_osm_features(
    bounds_4326: tuple[float, float, float, float],
    tags: list[str] | None = None,
    overpass_url: str = _DEFAULT_OVERPASS_URL,
) -> gpd.GeoDataFrame:
    """Query Overpass for OSM features within bounds, return GeoDataFrame in EPSG:26916.

    Raises requests.HTTPError on non-2xx response.
    Warns if zero elements returned.
    """
    if tags is None:
        tags = DEFAULT_TAGS

    query = build_overpass_query(bounds_4326, tags)
    resp = requests.post(overpass_url, data={"data": query}, timeout=90)
    resp.raise_for_status()

    data = resp.json()
    elements: list[dict[str, object]] = data.get("elements", [])
    if not elements:
        log.warning("fetch_osm_features: no elements for bbox=%r tags=%r", bounds_4326, tags)

    gdf = _elements_to_geodataframe(elements)
    if not gdf.empty:
        gdf = gdf.to_crs(26916)

    log.info("fetch_osm_features: %d features", len(gdf))
    return gdf


def save_osm(gdf: gpd.GeoDataFrame, out_path: Path, dry_run: bool = False) -> Path:
    """Write GeoDataFrame to GeoParquet."""
    if dry_run:
        log.info("dry-run: would write %d OSM features to %s", len(gdf), out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path)
    log.info("wrote OSM parquet: %s", out_path)
    return out_path
