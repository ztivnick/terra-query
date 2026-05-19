"""Tests for PAD-US ingest functions."""

import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
import responses as responses_lib
from shapely.geometry import box, mapping

from terra_query.ingest.padus import PAD_FIELDS, fetch_padus_features, save_padus

_BOUNDS = (-89.13, 46.38, -89.099, 46.409)

# minimal GeoJSON FeatureCollection matching required PAD_FIELDS
_MOCK_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": mapping(box(*_BOUNDS)),
            "properties": {
                "Unit_Nm": "Ottawa National Forest",
                "GAP_Sts": "2",
                "Loc_Nm": "Ottawa NF",
                "Mang_Type": "FED",
                "Pub_Access": "OA",
                "Des_Tp": "NF",
            },
        }
    ],
}

_MOCK_URL = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services"
    "/PAD_US4_0Designation/FeatureServer/0/query"
)


@responses_lib.activate
def test_fetch_padus_features_columns_and_crs() -> None:
    """fetch_padus_features returns GeoDataFrame with required columns in EPSG:26916."""
    responses_lib.add(
        responses_lib.GET,
        _MOCK_URL,
        json=_MOCK_GEOJSON,
        status=200,
    )

    gdf = fetch_padus_features(_BOUNDS)

    assert not gdf.empty
    assert gdf.crs is not None
    assert gdf.crs.to_epsg() == 26916
    # all required columns present
    for col in PAD_FIELDS:
        assert col in gdf.columns, f"missing column: {col}"


@responses_lib.activate
def test_fetch_padus_features_warns_on_empty(caplog: pytest.LogCaptureFixture) -> None:
    """fetch_padus_features warns when no features returned."""
    import logging

    empty_geojson = {"type": "FeatureCollection", "features": []}
    responses_lib.add(
        responses_lib.GET,
        _MOCK_URL,
        json=empty_geojson,
        status=200,
    )

    with caplog.at_level(logging.WARNING, logger="terra_query.ingest.padus"):
        gdf = fetch_padus_features(_BOUNDS)

    assert gdf.empty
    assert any("no features" in msg.lower() for msg in caplog.messages)


def test_save_padus_writes_parquet() -> None:
    """save_padus writes a readable GeoParquet file."""
    from shapely.geometry import box as shapely_box

    gdf = gpd.GeoDataFrame(
        {col: ["test_val"] for col in PAD_FIELDS},
        geometry=[shapely_box(*_BOUNDS)],
        crs="EPSG:26916",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "padus_aoi.parquet"
        result = save_padus(gdf, out_path)

        assert result == out_path
        assert out_path.exists()

        loaded = gpd.read_parquet(out_path)
        assert len(loaded) == 1
        for col in PAD_FIELDS:
            assert col in loaded.columns


def test_save_padus_dry_run() -> None:
    """save_padus dry_run=True does not create the file."""
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(*_BOUNDS)], crs="EPSG:26916")
    out_path = Path("/tmp/padus_dry_run.parquet")

    save_padus(gdf, out_path, dry_run=True)
    assert not out_path.exists()
