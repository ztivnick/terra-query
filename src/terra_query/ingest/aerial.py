"""Aerial-imagery ingest for the AOI.

Pulls NAIP (4-band, primary) and Sentinel-2 L2A winter scenes (leaf-off
context) from the Microsoft Planetary Computer STAC. Outputs reprojected,
mosaicked, AOI-clipped COGs in EPSG:26916, plus per-source manifest
sidecars. LiDAR and SAR are out of scope for this module.

This module hosts the reusable pieces. CLI entry points live in
`terra_query.ingest.cli.fetch_naip` and `terra_query.ingest.cli.fetch_sentinel2`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION = "naip"
S2_COLLECTION = "sentinel-2-l2a"

# raster storage grid
# NAIP is stored at native pixel size per cycle (1.0 m for 2012, 0.6 m for
# 2014+ Michigan cycles). One canonical "NAIP_PIXEL_SIZE_M" is not defined
# because mixed-res across cycles is the honest representation.
S2_PIXEL_SIZE_M = 10.0
AOI_BUFFER_M = 250.0

# winter window for the S2 leaf-off pull (inclusive month ints)
S2_WINTER_MONTHS = (12, 1, 2, 3)
S2_CLOUD_COVER_MAX = 20.0
S2_BANDS = ("B02", "B03", "B04", "B08", "B11")


@dataclass(frozen=True)
class StacPick:
    """One item picked from a STAC search."""

    collection: str
    item_id: str
    datetime: str  # ISO8601 capture time
    cloud_cover: float | None  # None for NAIP
    assets: dict[str, str]  # asset key -> signed URL, filled at fetch time


def aoi_bounds_wgs84(aoi_geojson: dict) -> tuple[float, float, float, float]:
    """Return the AOI's WGS84 bbox as (west, south, east, north)."""
    ring = aoi_geojson["features"][0]["geometry"]["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    return (min(lons), min(lats), max(lons), max(lats))


def _open_pc_client():
    from pystac_client import Client

    return Client.open(PC_STAC_URL)


def search_naip(bbox_wgs84: tuple[float, float, float, float]) -> list[StacPick]:
    """List all NAIP STAC items intersecting the bbox across the full catalog."""
    client = _open_pc_client()
    search = client.search(collections=[NAIP_COLLECTION], bbox=list(bbox_wgs84))
    picks: list[StacPick] = []
    for item in search.items():
        if item.datetime is None:
            continue
        picks.append(
            StacPick(
                collection=NAIP_COLLECTION,
                item_id=item.id,
                datetime=item.datetime.isoformat(),
                cloud_cover=None,
                assets={},
            )
        )
    return picks


def _item_covers_bbox(item, bbox_wgs84: tuple[float, float, float, float]) -> bool:
    """True iff the STAC item's geometry fully contains the WGS84 bbox.

    Sentinel-2 L2A items returned by a bbox search can be from MGRS tiles
    whose swath only clips the bbox edge; we want the AOI strictly inside
    the scene footprint or downstream rasters end up with nodata.
    """
    from shapely.geometry import box, shape

    if item.geometry is None:
        return False
    return shape(item.geometry).contains(box(*bbox_wgs84))


def search_sentinel2_winter(
    bbox_wgs84: tuple[float, float, float, float],
) -> list[StacPick]:
    """List Sentinel-2 L2A winter scenes that fully cover the bbox under the cloud cap."""
    client = _open_pc_client()
    search = client.search(
        collections=[S2_COLLECTION],
        bbox=list(bbox_wgs84),
        query={"eo:cloud_cover": {"lt": S2_CLOUD_COVER_MAX}},
    )
    picks: list[StacPick] = []
    for item in search.items():
        if item.datetime is None:
            continue
        if item.datetime.month not in S2_WINTER_MONTHS:
            continue
        if not _item_covers_bbox(item, bbox_wgs84):
            continue
        cc = item.properties.get("eo:cloud_cover")
        picks.append(
            StacPick(
                collection=S2_COLLECTION,
                item_id=item.id,
                datetime=item.datetime.isoformat(),
                cloud_cover=float(cc) if cc is not None else None,
                assets={},
            )
        )
    return picks


def aoi_bounds_buffered_26916(
    aoi_geojson_wgs84: dict,
) -> tuple[float, float, float, float]:
    """Reproject AOI to 26916, expand by AOI_BUFFER_M, snap to integer meters."""
    from terra_query.core.crs import to_working

    w, s, e, n = aoi_bounds_wgs84(aoi_geojson_wgs84)
    fwd = to_working()
    corners = [(w, s), (e, s), (e, n), (w, n)]
    xs, ys = zip(*(fwd.transform(lon, lat) for lon, lat in corners))
    west = math.floor(min(xs) - AOI_BUFFER_M)
    south = math.floor(min(ys) - AOI_BUFFER_M)
    east = math.ceil(max(xs) + AOI_BUFFER_M)
    north = math.ceil(max(ys) + AOI_BUFFER_M)
    return (west, south, east, north)


def is_valid_naip_cog(
    path: Path, expected_bounds: tuple[float, float, float, float]
) -> bool:
    """True if the file is a valid 4-band uint8 26916 COG covering the bounds.

    Allows a 1-pixel slack on each edge because `rasterio.merge.merge`
    quantizes the output bounds to the native pixel size, which is
    typically 0.5-1.0 m and not a divisor of an arbitrary integer-meter
    target.
    """
    if not path.exists():
        return False
    try:
        import rasterio

        with rasterio.open(path) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 26916:
                return False
            if ds.count != 4:
                return False
            if ds.dtypes[0] != "uint8":
                return False
            tol = abs(ds.transform.a)  # one pixel of slack
            w, s, e, n = expected_bounds
            db = ds.bounds
            if db.left > w + tol or db.bottom > s + tol:
                return False
            if db.right < e - tol or db.top < n - tol:
                return False
    except Exception:
        return False
    return True


def _fetch_pc_item(item_id: str, collection: str):
    """Fetch a STAC item from PC and sign its assets. Returns the signed item."""
    import planetary_computer
    from pystac_client import Client

    client = Client.open(PC_STAC_URL)
    coll = client.get_collection(collection)
    item = coll.get_item(item_id)
    return planetary_computer.sign(item)


def fetch_naip_cycle(
    items: list[StacPick],
    out_path: Path,
    aoi_bounds_26916: tuple[float, float, float, float],
) -> tuple[dict[str, str], bool]:
    """Download all NAIP items in one cycle, warp, mosaic, clip, COG-write.

    Returns (item_urls, skipped). `item_urls` maps item_id to the unsigned
    blob URL (always populated, even on skip, so the manifest stays
    complete after `discover` overwrites). `skipped` is True iff the
    output COG already exists and is valid; the heavy download/warp is
    skipped in that case but the URLs are still fetched.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge as rio_merge

    skipped = is_valid_naip_cog(out_path, aoi_bounds_26916)
    item_urls: dict[str, str] = {}

    if skipped:
        # cheap PC lookup per item, no raster I/O
        for pick in items:
            signed = _fetch_pc_item(pick.item_id, NAIP_COLLECTION)
            asset = signed.assets["image"]
            unsigned = asset.extra_fields.get(
                "msft:href", asset.href.split("?")[0]
            )
            item_urls[pick.item_id] = unsigned
        return item_urls, True

    west, south, east, north = aoi_bounds_26916
    src_datasets = []
    try:
        for pick in items:
            signed = _fetch_pc_item(pick.item_id, NAIP_COLLECTION)
            asset = signed.assets["image"]
            signed_url = asset.href
            unsigned = asset.extra_fields.get("msft:href", signed_url.split("?")[0])
            item_urls[pick.item_id] = unsigned
            src_datasets.append(rasterio.open(signed_url))

        for ds in src_datasets:
            if ds.crs.to_epsg() != 26916:
                raise RuntimeError(
                    f"unexpected source CRS {ds.crs} for NAIP item; "
                    f"expected EPSG:26916"
                )

        native_res = abs(src_datasets[0].transform.a)

        mosaic, transform = rio_merge(
            src_datasets,
            bounds=(west, south, east, north),
            res=(native_res, native_res),
            resampling=Resampling.bilinear,
        )
    finally:
        for ds in src_datasets:
            ds.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="COG",
        count=mosaic.shape[0],
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        dtype=mosaic.dtype,
        crs="EPSG:26916",
        transform=transform,
        compress="DEFLATE",
        predictor=2,
        blocksize=512,
        overview_resampling="average",
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(mosaic)

    return item_urls, False


def is_valid_s2_cog(
    path: Path, expected_bounds: tuple[float, float, float, float]
) -> bool:
    """True if the file is a valid 5-band 26916 COG at 10 m covering the bounds."""
    if not path.exists():
        return False
    try:
        import rasterio

        with rasterio.open(path) as ds:
            if ds.crs is None or ds.crs.to_epsg() != 26916:
                return False
            if ds.count != len(S2_BANDS):
                return False
            if abs(ds.transform.a - S2_PIXEL_SIZE_M) > 1e-6:
                return False
            tol = abs(ds.transform.a)
            w, s, e, n = expected_bounds
            db = ds.bounds
            if db.left > w + tol or db.bottom > s + tol:
                return False
            if db.right < e - tol or db.top < n - tol:
                return False
    except Exception:
        return False
    return True


def fetch_sentinel2_scene(
    item: StacPick,
    out_path: Path,
    aoi_bounds_26916: tuple[float, float, float, float],
) -> tuple[dict[str, str], bool]:
    """Fetch one S2 L2A scene, reproject + stack the chosen bands to a 10 m COG in 26916.

    Returns (asset_urls, skipped). Asset URLs are always populated; the
    download/warp is skipped iff a valid COG is already on disk.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    west, south, east, north = aoi_bounds_26916
    # snap to integer multiples of 10 m (the S2 output pixel size)
    west = math.floor(west / S2_PIXEL_SIZE_M) * S2_PIXEL_SIZE_M
    south = math.floor(south / S2_PIXEL_SIZE_M) * S2_PIXEL_SIZE_M
    east = math.ceil(east / S2_PIXEL_SIZE_M) * S2_PIXEL_SIZE_M
    north = math.ceil(north / S2_PIXEL_SIZE_M) * S2_PIXEL_SIZE_M
    width = int((east - west) / S2_PIXEL_SIZE_M)
    height = int((north - south) / S2_PIXEL_SIZE_M)
    target_transform = from_origin(west, north, S2_PIXEL_SIZE_M, S2_PIXEL_SIZE_M)

    signed = _fetch_pc_item(item.item_id, S2_COLLECTION)
    asset_urls: dict[str, str] = {}
    for band_name in S2_BANDS:
        if band_name not in signed.assets:
            raise RuntimeError(f"S2 item {item.item_id} missing band {band_name}")
        asset = signed.assets[band_name]
        unsigned = asset.extra_fields.get(
            "msft:href", asset.href.split("?")[0]
        )
        asset_urls[band_name] = unsigned

    if is_valid_s2_cog(out_path, aoi_bounds_26916):
        return asset_urls, True

    band_arrays: list[np.ndarray] = []

    for band_name in S2_BANDS:
        signed_url = signed.assets[band_name].href
        dst = np.zeros((height, width), dtype=np.uint16)
        with rasterio.open(signed_url) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs="EPSG:26916",
                resampling=Resampling.bilinear,
            )
        band_arrays.append(dst)

    stacked = np.stack(band_arrays, axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="COG",
        count=len(S2_BANDS),
        height=height,
        width=width,
        dtype="uint16",
        crs="EPSG:26916",
        transform=target_transform,
        compress="DEFLATE",
        predictor=2,
        blocksize=256,
        overview_resampling="average",
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(stacked)
        for i, b in enumerate(S2_BANDS, start=1):
            dst.set_band_description(i, b)

    return asset_urls, False
