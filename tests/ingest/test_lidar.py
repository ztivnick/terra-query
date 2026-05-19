"""Tests for 3DEP LiDAR ingest functions."""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import responses as responses_lib
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

from terra_query.ingest.lidar import download_and_reproject, fetch_3dep_product_urls

# Bond Falls AOI bounds (4326)
_BOUNDS = (-89.13, 46.38, -89.099, 46.409)

# Dataset string confirmed in Step 2a recon
_PRODUCT_TAG = "Digital Elevation Model (DEM) 1 meter"


def _make_test_geotiff_bytes(crs_epsg: int = 4326) -> bytes:
    """Build a minimal 10x10 float32 GeoTIFF in memory."""
    if crs_epsg == 4326:
        transform = from_bounds(-89.13, 46.38, -89.099, 46.409, 10, 10)
    else:
        # small UTM patch roughly over the AOI
        transform = from_bounds(332000, 5136000, 334000, 5138000, 10, 10)

    buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype=np.float32,
            crs=CRS.from_epsg(crs_epsg),
            transform=transform,
        ) as ds:
            ds.write(np.ones((1, 10, 10), dtype=np.float32))
        buf.write(memfile.read())
    return buf.getvalue()


@responses_lib.activate
def test_fetch_3dep_product_urls_returns_urls() -> None:
    """fetch_3dep_product_urls returns a non-empty list from mocked TNM response."""
    mock_response = {
        "total": 2,
        "items": [
            {
                "title": "USGS one meter x33y514 MI Ottawa NF 2017",
                "downloadURL": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/foo/x33y514.tif",
                "urls": {
                    "GeoTIFF": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/foo/x33y514.tif"
                },
            },
            {
                "title": "USGS one meter x33y515 MI Ottawa NF 2017",
                "downloadURL": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/foo/x33y515.tif",
                "urls": {
                    "GeoTIFF": "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects/foo/x33y515.tif"
                },
            },
        ],
        "errors": [],
        "messages": [],
    }
    responses_lib.add(
        responses_lib.GET,
        "https://tnmaccess.nationalmap.gov/api/v1/products",
        json=mock_response,
        status=200,
    )

    urls = fetch_3dep_product_urls(_BOUNDS, _PRODUCT_TAG)
    assert len(urls) == 2
    assert all(u.endswith(".tif") for u in urls)


@responses_lib.activate
def test_fetch_3dep_product_urls_warns_on_empty(caplog: pytest.LogCaptureFixture) -> None:
    """fetch_3dep_product_urls returns empty list and logs warning when no products."""
    responses_lib.add(
        responses_lib.GET,
        "https://tnmaccess.nationalmap.gov/api/v1/products",
        json={"total": 0, "items": [], "errors": [], "messages": []},
        status=200,
    )
    import logging

    with caplog.at_level(logging.WARNING, logger="terra_query.ingest.lidar"):
        urls = fetch_3dep_product_urls(_BOUNDS, _PRODUCT_TAG)

    assert urls == []
    assert any(
        "0" in msg
        or "zero" in msg.lower()
        or "no products" in msg.lower()
        or "empty" in msg.lower()
        for msg in caplog.messages
    )


@responses_lib.activate
def test_download_and_reproject_crs_and_cog() -> None:
    """download_and_reproject produces a file in EPSG:26916 that is a valid COG."""
    fake_url = "https://prd-tnm.s3.amazonaws.com/test_dem.tif"
    geotiff_bytes = _make_test_geotiff_bytes(crs_epsg=4326)

    responses_lib.add(
        responses_lib.GET,
        fake_url,
        body=geotiff_bytes,
        status=200,
        content_type="image/tiff",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_dem_26916.tif"
        result = download_and_reproject(fake_url, out_path, dst_crs=26916, dry_run=False)

        assert result == out_path
        assert out_path.exists()

        with rasterio.open(out_path) as ds:
            assert ds.crs is not None
            assert ds.crs.to_epsg() == 26916

        is_valid, errors, warnings = cog_validate(str(out_path))
        assert is_valid, f"COG validation failed: {errors}"


def test_download_and_reproject_dry_run() -> None:
    """download_and_reproject with dry_run=True makes no HTTP call and returns path."""
    out_path = Path("/tmp/dry_run_test.tif")
    result = download_and_reproject(
        "https://example.com/fake.tif", out_path, dst_crs=26916, dry_run=True
    )
    assert result == out_path
    # file should not be created in dry-run mode
    assert not out_path.exists()
