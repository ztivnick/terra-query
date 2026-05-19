"""3DEP 1m DEM ingest: query TNM, download, reproject to EPSG:26916, write COG."""

import logging
import tempfile
from pathlib import Path

import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

log = logging.getLogger(__name__)

_TNM_ENDPOINT = "https://tnmaccess.nationalmap.gov/api/v1/products"


def fetch_3dep_product_urls(
    bounds_4326: tuple[float, float, float, float],
    product_tag: str,
) -> list[str]:
    """Query TNM API and return a list of GeoTIFF download URLs.

    Logs a warning if response contains zero products.
    """
    w, s, e, n = bounds_4326
    resp = requests.get(
        _TNM_ENDPOINT,
        params={
            "bbox": f"{w},{s},{e},{n}",
            "datasets": product_tag,
            "prodFormats": "GeoTIFF",
            "outputFormat": "JSON",
            "max": "50",
        },
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    urls: list[str] = []
    for item in items:
        url = item.get("downloadURL") or (item.get("urls") or {}).get("GeoTIFF")
        if url:
            urls.append(str(url))
    if not urls:
        log.warning(
            "fetch_3dep_product_urls: no products found for tag=%r in bbox=%r",
            product_tag,
            (w, s, e, n),
        )
    return urls


def download_and_reproject(
    url: str,
    out_path: Path,
    dst_crs: int = 26916,
    dry_run: bool = False,
) -> Path:
    """Download a GeoTIFF, reproject to dst_crs, write as COG.

    Uses rasterio.warp.reproject for reprojection and cog_translate for COG output.
    """
    if dry_run:
        log.info("dry-run: would download %s -> %s", url, out_path)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs_obj = CRS.from_epsg(dst_crs)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.tif"
        reprojected_path = Path(tmpdir) / "reprojected.tif"

        # stream download to avoid loading large files into memory
        log.info("downloading %s", url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with raw_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)

        # reproject to working CRS
        with rasterio.open(raw_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs_obj, src.width, src.height, *src.bounds
            )
            profile = src.profile.copy()
            profile.update(
                crs=dst_crs_obj,
                transform=transform,
                width=width,
                height=height,
                driver="GTiff",
            )
            with rasterio.open(reprojected_path, "w", **profile) as dst:
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

        # write as cloud-optimized GeoTIFF
        cog_profile = cog_profiles.get("deflate")  # type: ignore[no-untyped-call]
        cog_translate(reprojected_path, out_path, cog_profile, quiet=True)

    log.info("wrote COG: %s", out_path)
    return out_path
