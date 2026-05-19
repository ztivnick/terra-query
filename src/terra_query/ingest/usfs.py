"""USFS Ottawa NF administrative boundary ingest: ArcGIS REST -> EPSG:26916 GeoParquet.

NOTE: apps.fs.usda.gov blocks programmatic access from some networks (WAF/IP ACL).
If the default layer_url 403s, substitute a working ArcGIS Online mirror or
supply a local GeoJSON via the CLI --boundary-file flag.
"""

import io
import logging
from pathlib import Path

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

# known starting point; may 403 on restricted networks - see module note above
_DEFAULT_LAYER_URL = (
    "https://apps.fs.usda.gov/arcgis/rest/services/EDW/EDW_ForestSystemBoundaries_01/MapServer/0"
)


def fetch_usfs_boundary(
    forest_name: str = "Ottawa National Forest",
    layer_url: str = _DEFAULT_LAYER_URL,
) -> gpd.GeoDataFrame:
    """Fetch the Ottawa NF admin boundary polygon from ArcGIS REST.

    Returns GeoDataFrame in EPSG:26916.
    Raises requests.HTTPError if the service is unavailable.
    """
    params: dict[str, str] = {
        "where": f"FORESTNAME='{forest_name}'",
        "outFields": "*",
        "f": "geojson",
        "outSR": "4326",
    }
    resp = requests.get(f"{layer_url}/query", params=params, timeout=60)
    resp.raise_for_status()

    gdf = gpd.read_file(io.StringIO(resp.text))
    if gdf.empty:
        log.warning("fetch_usfs_boundary: no features for forest_name=%r", forest_name)
        return gdf

    gdf = gdf.to_crs(26916)

    # basic validity check
    invalid = gdf[~gdf.geometry.is_valid]
    if not invalid.empty:
        log.warning("fetch_usfs_boundary: %d invalid geometries", len(invalid))

    log.info("fetch_usfs_boundary: %d polygon(s) for %s", len(gdf), forest_name)
    return gdf


def save_usfs(gdf: gpd.GeoDataFrame, out_path: Path, dry_run: bool = False) -> Path:
    """Write GeoDataFrame to GeoParquet."""
    if dry_run:
        log.info("dry-run: would write USFS boundary to %s", out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path)
    log.info("wrote USFS boundary parquet: %s", out_path)
    return out_path
