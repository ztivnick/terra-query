"""Gate checks for the aerial ingest pipeline.

Steps that pull network data produce on-disk COGs; these tests read
those COGs. Tests that need a COG that has not been fetched yet skip
cleanly.
"""

from __future__ import annotations

import json

import pytest

from terra_query.core import config
from terra_query.core.paths import (
    aoi_wgs84,
    eval_26916,
    naip_cog,
    naip_manifest,
    s2_cog,
    s2_manifest,
)
from terra_query.ingest import aerial


@pytest.fixture(scope="module")
def cfg() -> dict:
    return config.load_experiment()


@pytest.fixture(scope="module")
def aoi_id(cfg) -> str:
    return config.aoi_id_of(cfg)


@pytest.fixture(scope="module")
def eval_set_id(cfg) -> str:
    return config.eval_set_id_of(cfg)


@pytest.fixture(scope="module")
def aoi_geojson(aoi_id):
    return json.loads(aoi_wgs84(aoi_id).read_text())


@pytest.fixture(scope="module")
def aoi_bounds_unbuffered_26916(aoi_geojson):
    from terra_query.core.crs import to_working

    w, s, e, n = aerial.aoi_bounds_wgs84(aoi_geojson)
    fwd = to_working()
    corners = [(w, s), (e, s), (e, n), (w, n)]
    xs, ys = zip(*(fwd.transform(lon, lat) for lon, lat in corners))
    return (min(xs), min(ys), max(xs), max(ys))


@pytest.fixture(scope="module")
def naip_manifest_doc(aoi_id):
    p = naip_manifest(aoi_id)
    if not p.exists():
        pytest.skip("naip manifest missing; run discover_aerial")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def naip_fetched_years(aoi_id, naip_manifest_doc):
    years = [
        c["year"] for c in naip_manifest_doc["cycles"]
        if naip_cog(aoi_id, c["year"]).exists()
    ]
    if not years:
        pytest.skip("no NAIP COGs fetched yet")
    return years


def test_module_constants_sane():
    assert aerial.NAIP_COLLECTION == "naip"
    assert aerial.S2_COLLECTION == "sentinel-2-l2a"
    assert aerial.S2_PIXEL_SIZE_M == 10.0
    assert aerial.AOI_BUFFER_M == 250.0
    assert set(aerial.S2_WINTER_MONTHS) == {12, 1, 2, 3}
    assert aerial.S2_CLOUD_COVER_MAX == 20.0
    assert aerial.S2_BANDS == ("B02", "B03", "B04", "B08", "B11")
    assert aerial.PC_STAC_URL.startswith("https://")


def test_naip_cogs_are_valid_26916(aoi_id, naip_fetched_years, aoi_bounds_unbuffered_26916):
    """Each fetched NAIP COG opens, declares 26916, has 4 bands uint8, covers AOI."""
    import rasterio

    for year in naip_fetched_years:
        path = naip_cog(aoi_id, year)
        with rasterio.open(path) as ds:
            assert ds.crs.to_epsg() == 26916, f"{year}: crs {ds.crs}"
            assert ds.count == 4, f"{year}: {ds.count} bands"
            assert all(d == "uint8" for d in ds.dtypes), f"{year}: dtypes {ds.dtypes}"
            w, s, e, n = aoi_bounds_unbuffered_26916
            db = ds.bounds
            assert db.left <= w and db.right >= e, f"{year}: x bounds {db} vs AOI"
            assert db.bottom <= s and db.top >= n, f"{year}: y bounds {db} vs AOI"


def test_naip_cogs_pixel_grid_anchored(aoi_id, naip_fetched_years):
    """COG upper-left pixel corner anchored to integer meters in 26916."""
    import rasterio

    for year in naip_fetched_years:
        path = naip_cog(aoi_id, year)
        with rasterio.open(path) as ds:
            t = ds.transform
            assert t.c == int(t.c), f"{year}: ul x {t.c} not integer-meter"
            assert t.f == int(t.f), f"{year}: ul y {t.f} not integer-meter"
            assert t.a > 0 and t.e < 0, f"{year}: unexpected pixel orientation"


def test_naip_eval_points_sample_nonzero(aoi_id, eval_set_id, naip_fetched_years):
    """At each positive_in_scope eval point, the latest NAIP samples are non-zero."""
    import rasterio

    eval_fc = json.loads(eval_26916(eval_set_id).read_text())

    latest_year = max(naip_fetched_years)
    path = naip_cog(aoi_id, latest_year)
    with rasterio.open(path) as ds:
        for feat in eval_fc["features"]:
            if feat["properties"]["category"] != "positive_in_scope":
                continue
            x, y = feat["geometry"]["coordinates"]
            row, col = ds.index(x, y)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                pytest.fail(f"{feat['properties']['id']} outside NAIP raster")
            vals = [ds.read(b + 1)[row, col] for b in range(ds.count)]
            assert any(v > 0 for v in vals), (
                f"{feat['properties']['id']} at ({x}, {y}) sampled all zero"
            )


@pytest.fixture(scope="module")
def s2_manifest_doc(aoi_id):
    p = s2_manifest(aoi_id)
    if not p.exists():
        pytest.skip("sentinel2 manifest missing; run discover_aerial")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def s2_fetched_scenes(aoi_id, s2_manifest_doc):
    scenes = [
        s for s in s2_manifest_doc["scenes"]
        if s2_cog(aoi_id, s["datetime"][:10]).exists()
    ]
    if not scenes:
        pytest.skip("no S2 COGs fetched yet")
    return scenes


def test_s2_cogs_are_valid_26916(aoi_id, s2_fetched_scenes, aoi_bounds_unbuffered_26916):
    """Each fetched S2 COG opens, declares 26916, has 5 bands uint16, 10 m, covers AOI."""
    import rasterio

    for scene in s2_fetched_scenes:
        date_str = scene["datetime"][:10]
        path = s2_cog(aoi_id, date_str)
        with rasterio.open(path) as ds:
            assert ds.crs.to_epsg() == 26916, f"{date_str}: crs {ds.crs}"
            assert ds.count == len(aerial.S2_BANDS), f"{date_str}: {ds.count} bands"
            assert ds.dtypes[0] == "uint16", f"{date_str}: dtype {ds.dtypes[0]}"
            assert abs(ds.transform.a - 10.0) < 1e-6, f"{date_str}: pixel {ds.transform.a}"
            assert tuple(ds.descriptions) == aerial.S2_BANDS, (
                f"{date_str}: band names {ds.descriptions}"
            )
            w, s, e, n = aoi_bounds_unbuffered_26916
            db = ds.bounds
            assert db.left <= w and db.right >= e, f"{date_str}: x bounds {db} vs AOI"
            assert db.bottom <= s and db.top >= n, f"{date_str}: y bounds {db} vs AOI"


def test_s2_capture_in_winter(s2_fetched_scenes):
    """All fetched S2 scenes were captured in the winter window."""
    for scene in s2_fetched_scenes:
        month = int(scene["datetime"][5:7])
        assert month in aerial.S2_WINTER_MONTHS, f"month {month} not in winter set"


def test_s2_cloud_under_cap(s2_fetched_scenes):
    """All fetched S2 scenes are under the cloud cover cap."""
    for scene in s2_fetched_scenes:
        cc = scene["cloud_cover"]
        assert cc is not None, "cloud_cover missing on scene"
        assert cc < aerial.S2_CLOUD_COVER_MAX, f"cloud {cc}% >= cap"


def test_s2_cogs_low_nodata(aoi_id, s2_fetched_scenes):
    """S2 COG must cover the AOI - guards against partial-swath picks (T15TYM R069 etc)."""
    import rasterio

    for scene in s2_fetched_scenes:
        date_str = scene["datetime"][:10]
        path = s2_cog(aoi_id, date_str)
        with rasterio.open(path) as ds:
            b = ds.read(1)
        nodata_frac = float((b == 0).sum()) / b.size
        assert nodata_frac < 0.05, (
            f"{date_str}: {nodata_frac*100:.1f}% nodata on band B02 - "
            f"likely a partial-coverage scene; check the coverage filter"
        )


def test_naip_manifest_items_have_asset_urls_after_fetch(naip_manifest_doc, naip_fetched_years):
    """Every NAIP item belonging to a fetched cycle has an asset URL + fetched_at."""
    fetched = set(naip_fetched_years)
    for cycle in naip_manifest_doc["cycles"]:
        if cycle["year"] not in fetched:
            continue
        for item in cycle["items"]:
            url = item.get("assets", {}).get("image")
            assert url and url.startswith("https://"), (
                f"{item['item_id']}: asset URL missing or not https"
            )
            assert item.get("fetched_at"), f"{item['item_id']}: fetched_at missing"


def test_s2_manifest_scenes_have_asset_urls_after_fetch(s2_manifest_doc, s2_fetched_scenes):
    """Every fetched S2 scene has all chosen-band asset URLs and fetched_at."""
    fetched_ids = {s["item_id"] for s in s2_fetched_scenes}
    for scene in s2_manifest_doc["scenes"]:
        if scene["item_id"] not in fetched_ids:
            continue
        assets = scene.get("assets", {})
        for band in aerial.S2_BANDS:
            url = assets.get(band)
            assert url and url.startswith("https://"), (
                f"{scene['item_id']}: band {band} asset URL missing"
            )
        assert scene.get("fetched_at"), f"{scene['item_id']}: fetched_at missing"
