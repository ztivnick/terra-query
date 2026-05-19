"""PAD-US v4.0 ingest: query ArcGIS REST, reproject to EPSG:26916, write GeoParquet."""

import io
import json
import logging
from pathlib import Path

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

PAD_FIELDS = ["Unit_Nm", "GAP_Sts", "Loc_Nm", "Mang_Type", "Pub_Access", "Des_Tp"]

# confirmed working via HEAD check
_DEFAULT_LAYER_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services"
    "/PAD_US4_0Designation/FeatureServer/0"
)


def fetch_padus_features(
    bounds_4326: tuple[float, float, float, float],
    layer_url: str = _DEFAULT_LAYER_URL,
) -> gpd.GeoDataFrame:
    """Query PAD-US ArcGIS REST FeatureServer for features intersecting bounds.

    Only fetches fields: Unit_Nm, GAP_Sts, Loc_Nm, Mang_Type, Pub_Access, Des_Tp.
    Returns GeoDataFrame in EPSG:26916.
    """
    w, s, e, n = bounds_4326
    bbox_geom = json.dumps(
        {"xmin": w, "ymin": s, "xmax": e, "ymax": n, "spatialReference": {"wkid": 4326}}
    )

    params: dict[str, str] = {
        "geometry": bbox_geom,
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": ",".join(PAD_FIELDS),
        "f": "geojson",
        "outSR": "4326",
        "resultRecordCount": "1000",
    }
    resp = requests.get(f"{layer_url}/query", params=params, timeout=60)
    resp.raise_for_status()

    # use io.StringIO - avoids pyogrio warning about driver open option
    gdf = gpd.read_file(io.StringIO(resp.text))
    if gdf.empty:
        log.warning("fetch_padus_features: no features returned for bbox=%r", bounds_4326)
        return gdf

    gdf = gdf.to_crs(26916)
    log.info("fetch_padus_features: %d features", len(gdf))
    return gdf


def save_padus(gdf: gpd.GeoDataFrame, out_path: Path, dry_run: bool = False) -> Path:
    """Write GeoDataFrame to GeoParquet."""
    if dry_run:
        log.info("dry-run: would write %d features to %s", len(gdf), out_path)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out_path)
    log.info("wrote PAD-US parquet: %s", out_path)
    return out_path
