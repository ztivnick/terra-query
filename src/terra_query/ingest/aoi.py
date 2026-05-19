"""AOI helper functions for Terra-query."""

from pathlib import Path

import geopandas as gpd


def load_aoi(path: Path) -> gpd.GeoDataFrame:
    """Load a GeoJSON AOI and reproject to EPSG:26916."""
    gdf = gpd.read_file(path)
    # reproject to working CRS: NAD83 / UTM Zone 16N
    gdf = gdf.to_crs(26916)
    return gdf


def aoi_bounds_4326(path: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for API bbox queries."""
    gdf = gpd.read_file(path)
    b = gdf.total_bounds  # [minx, miny, maxx, maxy]
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def aoi_bounds_26916(path: Path) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in EPSG:26916."""
    gdf = load_aoi(path)
    b = gdf.total_bounds
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
