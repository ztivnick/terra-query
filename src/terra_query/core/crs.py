"""Working CRS for the project.

EPSG:26916 (NAD83 / UTM Zone 16N, meters) is the single projected
frame for every raster and vector. Grid origin convention: pixel
corners snap to integer-meter coordinates in 26916. Per-raster pixel
size is read off each source COG; this module does not pin a pixel
size.

Coordinates here use (x, y) = (lon, lat) for WGS84 and (easting,
northing) for the working CRS. All Transformers are built with
always_xy=True so callers do not have to think about axis order.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from pyproj import CRS, Transformer

WORKING_CRS_EPSG = 26916
WORKING_CRS = f"EPSG:{WORKING_CRS_EPSG}"
WGS84 = "EPSG:4326"


@lru_cache(maxsize=1)
def working_crs() -> CRS:
    return CRS.from_epsg(WORKING_CRS_EPSG)


@lru_cache(maxsize=2)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def to_working() -> Transformer:
    return _transformer(WGS84, WORKING_CRS)


def to_wgs84() -> Transformer:
    return _transformer(WORKING_CRS, WGS84)


def area_of_use_bounds() -> tuple[float, float, float, float]:
    """Return the working CRS's area-of-use bbox in WGS84 (west, south, east, north)."""
    aou = working_crs().area_of_use
    if aou is None:
        raise RuntimeError(f"{WORKING_CRS} has no area_of_use; cannot verify coverage")
    return (aou.west, aou.south, aou.east, aou.north)


def area_of_use_covers(bounds_wgs84: tuple[float, float, float, float]) -> bool:
    """True iff the given WGS84 bbox is strictly inside the working CRS area-of-use."""
    w, s, e, n = bounds_wgs84
    aw, as_, ae, an = area_of_use_bounds()
    return aw < w and as_ < s and ae > e and an > n


def reproject_point(
    coords: Iterable[float], transformer: Transformer | None = None
) -> tuple[float, float]:
    t = transformer or to_working()
    x, y = coords
    return t.transform(x, y)


def reproject_ring(
    ring: list[list[float]], transformer: Transformer | None = None
) -> list[list[float]]:
    t = transformer or to_working()
    return [list(t.transform(x, y)) for x, y in ring]


def reproject_polygon(
    rings: list[list[list[float]]], transformer: Transformer | None = None
) -> list[list[list[float]]]:
    """Reproject a GeoJSON Polygon's coordinates (outer ring + holes)."""
    t = transformer or to_working()
    return [reproject_ring(r, t) for r in rings]
