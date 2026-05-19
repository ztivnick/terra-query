"""Tests for NAIP STAC ingest functions."""

import datetime
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

from terra_query.ingest.naip import download_naip_item, search_naip_items

_BOUNDS = (-89.13, 46.38, -89.099, 46.409)


def _make_4band_geotiff(path: Path) -> None:
    """Write a small 4-band uint8 GeoTIFF (RGBIR) in EPSG:4326."""
    transform = from_bounds(*_BOUNDS, width=20, height=20)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=4,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as ds:
        for band in range(1, 5):
            ds.write(np.full((20, 20), band * 50, dtype=np.uint8), band)


def _make_mock_item(href: str) -> MagicMock:
    item = MagicMock()
    item.id = "mi_m_4608955_ne_16_060_20220715"
    item.datetime = datetime.datetime(2022, 7, 15, tzinfo=datetime.UTC)
    asset = MagicMock()
    asset.href = href
    item.assets = {"image": asset}
    return item


def test_search_naip_items_returns_items() -> None:
    """search_naip_items returns a non-empty list when STAC responds with items."""
    mock_item = _make_mock_item("https://example.com/naip.tif")
    mock_search = MagicMock()
    mock_search.items.return_value = [mock_item]
    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    with patch("pystac_client.Client.open", return_value=mock_client):
        items = search_naip_items(_BOUNDS)

    assert len(items) >= 1
    mock_client.search.assert_called_once()


def test_search_naip_items_warns_on_empty(caplog: pytest.LogCaptureFixture) -> None:
    """search_naip_items returns [] and logs warning when STAC returns nothing."""
    mock_search = MagicMock()
    mock_search.items.return_value = []
    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    import logging

    with patch("pystac_client.Client.open", return_value=mock_client):
        with caplog.at_level(logging.WARNING, logger="terra_query.ingest.naip"):
            items = search_naip_items(_BOUNDS)

    assert items == []
    assert any("no NAIP" in msg or "zero" in msg.lower() for msg in caplog.messages)


def test_download_naip_item_4band_crs_cog() -> None:
    """download_naip_item produces 4-band output in EPSG:26916 that is a valid COG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_tif = Path(tmpdir) / "fake_naip.tif"
        out_tif = Path(tmpdir) / "output_naip.tif"

        _make_4band_geotiff(src_tif)
        mock_item = _make_mock_item(str(src_tif))

        result = download_naip_item(mock_item, out_tif, _BOUNDS, dst_crs=26916, dry_run=False)

        assert result == out_tif
        assert out_tif.exists()

        with rasterio.open(out_tif) as ds:
            assert ds.count == 4, f"expected 4 bands, got {ds.count}"
            assert ds.crs is not None
            assert ds.crs.to_epsg() == 26916

        is_valid, errors, _warnings = cog_validate(str(out_tif))
        assert is_valid, f"COG validation failed: {errors}"


def test_download_naip_item_dry_run() -> None:
    """download_naip_item dry_run=True returns path without creating file."""
    out_path = Path("/tmp/naip_dry_run_test.tif")
    mock_item = _make_mock_item("https://example.com/naip.tif")

    result = download_naip_item(mock_item, out_path, _BOUNDS, dst_crs=26916, dry_run=True)

    assert result == out_path
    assert not out_path.exists()
