"""NAIP 0.6m RGBIR ingest via Planetary Computer STAC. Windowed reads only."""

import logging
import tempfile
from pathlib import Path

import planetary_computer
import pystac
import pystac_client
import rasterio
import rasterio.windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

log = logging.getLogger(__name__)

_PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def search_naip_items(
    bounds_4326: tuple[float, float, float, float],
) -> list[pystac.Item]:
    """Search Planetary Computer STAC for the most recent NAIP items intersecting bounds.

    Returns items sorted by date descending (most recent first).
    Logs the vintage year found and warns if zero items returned.
    """
    catalog = pystac_client.Client.open(_PC_STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["naip"],
        bbox=bounds_4326,
        sortby="-datetime",
        max_items=50,
    )
    items: list[pystac.Item] = list(search.items())
    if not items:
        log.warning("search_naip_items: no NAIP items for bbox=%r", bounds_4326)
        return []
    vintage = items[0].datetime.year if items[0].datetime else "unknown"
    log.info("search_naip_items: %d items, vintage %s", len(items), vintage)
    return items


def download_naip_item(
    item: pystac.Item,
    out_path: Path,
    aoi_bounds_4326: tuple[float, float, float, float],
    dst_crs: int = 26916,
    dry_run: bool = False,
) -> Path:
    """Download NAIP (4-band RGBIR) using a windowed read of the AOI region only.

    Reprojects to dst_crs and writes as COG. Never reads the full scene.
    """
    if dry_run:
        log.info("dry-run: would download NAIP item %s -> %s", item.id, out_path)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs_obj = CRS.from_epsg(dst_crs)
    href = item.assets["image"].href

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "naip_raw.tif"
        reprojected_path = Path(tmpdir) / "naip_reprojected.tif"

        # windowed read - only pull the pixels that cover the AOI
        with rasterio.open(href) as src:
            window = rasterio.windows.from_bounds(*aoi_bounds_4326, transform=src.transform)
            # clamp window to actual raster dimensions
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            windowed_transform = src.window_transform(window)
            data = src.read(window=window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=windowed_transform,
                driver="GTiff",
            )
            log.info(
                "windowed read: %d bands, shape (%d, %d) from %s",
                data.shape[0],
                data.shape[1],
                data.shape[2],
                item.id,
            )

        with rasterio.open(raw_path, "w", **profile) as dst:
            dst.write(data)

        # reproject windowed tile to working CRS
        with rasterio.open(raw_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs_obj, src.width, src.height, *src.bounds
            )
            profile2 = src.profile.copy()
            profile2.update(
                crs=dst_crs_obj,
                transform=transform,
                width=width,
                height=height,
                driver="GTiff",
            )
            with rasterio.open(reprojected_path, "w", **profile2) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs_obj,
                        resampling=Resampling.bilinear,
                    )

        cog_profile = cog_profiles.get("deflate")  # type: ignore[no-untyped-call]
        cog_translate(reprojected_path, out_path, cog_profile, quiet=True)

    log.info("wrote NAIP COG: %s", out_path)
    return out_path
