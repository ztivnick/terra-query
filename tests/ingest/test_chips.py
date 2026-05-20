"""Tests for the S4 chip grid. Populated across BUILD steps 2-5."""

from __future__ import annotations

import json

import pytest

from terra_query.core.paths import (
    AOI_26916,
    CHIP_EVAL_DIR,
    EVAL_26916,
    NAIP_MANIFEST,
    chip_eval,
    naip_cog,
)
from terra_query.ingest.chips import (
    CHIP_SIZE_M,
    CIR_BANDS,
    NAIP_BAND_COUNT,
    RGB_BANDS,
    STRIDE_M,
    ChipBox,
    assemble_chip_index,
    build_chip_grid,
    chip_id,
    containing_chips,
    grid_dims,
    grid_origin,
    primary_containing_chip,
    read_chip,
    render_eval_chip_png,
)

# actual AOI bbox in 26916 (from data/verification/crs_verification.txt).
# the grid uses these (not the buffered AOI) so chips stay inside the COG extent.
AOI_BOUNDS = (333511.040, 5139212.617, 338656.930, 5144347.779)
# buffered AOI / COG extent for sanity checks
BUFFERED_BOUNDS = (333261.0, 5138962.0, 338907.0, 5144598.0)


def test_grid_origin_snaps_below_aoi_west_south():
    ow, os = grid_origin(AOI_BOUNDS, STRIDE_M)
    assert ow <= AOI_BOUNDS[0]
    assert os <= AOI_BOUNDS[1]
    assert ow % STRIDE_M == 0
    assert os % STRIDE_M == 0
    # within one stride of the actual west / south
    assert AOI_BOUNDS[0] - ow < STRIDE_M
    assert AOI_BOUNDS[1] - os < STRIDE_M


def test_grid_dims_for_bond_falls_aoi():
    # AOI is ~5146 m wide x ~5135 m tall; 5146/112 = 45.95 -> 47 cols
    n_rows, n_cols = grid_dims(AOI_BOUNDS)
    assert n_rows == 47
    assert n_cols == 47


def test_build_chip_grid_count_and_shape():
    chips = build_chip_grid(AOI_BOUNDS)
    assert len(chips) == 47 * 47
    for c in chips:
        assert c.east - c.west == CHIP_SIZE_M
        assert c.north - c.south == CHIP_SIZE_M
        # west / south aligned to stride multiples in 26916
        assert c.west % STRIDE_M == 0
        assert c.south % STRIDE_M == 0


def test_build_chip_grid_covers_aoi_and_stays_inside_cog():
    chips = build_chip_grid(AOI_BOUNDS)
    w, s, e, n = AOI_BOUNDS
    bw, bs, be, bn = BUFFERED_BOUNDS
    # last chip extends past the AOI's NE corner (so AOI is fully covered)
    east_max = max(c.east for c in chips)
    north_max = max(c.north for c in chips)
    assert east_max >= e
    assert north_max >= n
    # ... but NOT past the buffered AOI / COG extent (no nodata reads)
    assert east_max <= be
    assert north_max <= bn
    # first chip is at most one stride south/west of the AOI's SW corner
    west_min = min(c.west for c in chips)
    south_min = min(c.south for c in chips)
    assert w - west_min < STRIDE_M
    assert s - south_min < STRIDE_M
    # ... and stays inside the COG west/south edges
    assert west_min >= bw
    assert south_min >= bs


def test_chip_id_format():
    assert chip_id("2022", 0, 0) == "naip_2022_r000_c000"
    assert chip_id("2014", 50, 50) == "naip_2014_r050_c050"
    assert chip_id("2022", 7, 13) == "naip_2022_r007_c013"


def test_containing_chips_returns_four_at_interior_overlap_point():
    chips = build_chip_grid(AOI_BOUNDS)
    center = (
        (AOI_BOUNDS[0] + AOI_BOUNDS[2]) / 2.0,
        (AOI_BOUNDS[1] + AOI_BOUNDS[3]) / 2.0,
    )
    candidates = containing_chips(center, chips)
    # at an interior point with 50% overlap, exactly 4 chips contain the point
    assert len(candidates) == 4


def test_containing_chips_empty_outside_aoi():
    chips = build_chip_grid(AOI_BOUNDS)
    far_away = (100000.0, 4000000.0)
    assert containing_chips(far_away, chips) == []


def test_primary_containing_chip_is_most_centered():
    chips = build_chip_grid(AOI_BOUNDS)
    center = (
        (AOI_BOUNDS[0] + AOI_BOUNDS[2]) / 2.0,
        (AOI_BOUNDS[1] + AOI_BOUNDS[3]) / 2.0,
    )
    primary = primary_containing_chip(center, chips)
    assert primary is not None
    cx, cy = primary.center
    candidates = containing_chips(center, chips)
    dists = [
        ((c.center[0] - center[0]) ** 2 + (c.center[1] - center[1]) ** 2)
        for c in candidates
    ]
    primary_dist = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
    assert primary_dist == min(dists)


def test_primary_containing_chip_none_outside():
    chips = build_chip_grid(AOI_BOUNDS)
    assert primary_containing_chip((0.0, 0.0), chips) is None


def test_chip_box_bbox_and_center():
    chip = ChipBox(row=2, col=3, west=333200.0, south=5138896.0)
    assert chip.bbox == (333200.0, 5138896.0, 333424.0, 5139120.0)
    assert chip.center == (333312.0, 5139008.0)


# ---- step 3: chip index assembly ----------------------------------------


@pytest.fixture(scope="module")
def real_chip_index() -> dict:
    """Assemble the chip index from the real S2/S3 artifacts on disk."""
    from shapely.geometry import shape

    aoi_gj = json.loads(AOI_26916.read_text())
    aoi_poly = shape(aoi_gj["features"][0]["geometry"])

    ev_gj = json.loads(EVAL_26916.read_text())
    ev_pts: list[tuple[str, tuple[float, float]]] = []
    for f in ev_gj["features"]:
        x, y = f["geometry"]["coordinates"]
        ev_pts.append((f["properties"]["id"], (float(x), float(y))))

    manifest = json.loads(NAIP_MANIFEST.read_text())
    cycle_cogs = [(c["year"], naip_cog(c["year"])) for c in manifest["cycles"]]

    return assemble_chip_index(
        aoi_polygon_26916=aoi_poly,
        eval_points_26916=ev_pts,
        naip_cycle_cogs=cycle_cogs,
        aoi_bounds_26916=AOI_BOUNDS,
        aoi_buffered_bounds_26916=BUFFERED_BOUNDS,
    )


def test_index_top_level_parameters(real_chip_index):
    idx = real_chip_index
    assert idx["working_crs"] == "EPSG:26916"
    assert idx["chip_size_m"] == 224
    assert idx["stride_m"] == 112
    assert idx["band_count_available"] == 4
    assert idx["rgb_bands"] == [1, 2, 3]
    assert idx["cir_bands"] == [4, 1, 2]
    assert idx["n_rows"] == 47
    assert idx["n_cols"] == 47


def test_index_six_cycles_with_equal_chip_count(real_chip_index):
    idx = real_chip_index
    assert len(idx["cycles"]) == 6
    counts = {c["year"]: len(c["chips"]) for c in idx["cycles"]}
    assert set(counts.keys()) == {"2012", "2014", "2016", "2018", "2020", "2022"}
    assert all(v == 47 * 47 for v in counts.values())


def test_index_native_chip_array_shape_matches_cycle_resolution(real_chip_index):
    """1.0 m cycles -> 224 px square; 0.6 m cycles -> 373 px square."""
    by_year = {c["year"]: c for c in real_chip_index["cycles"]}
    for y in ("2012", "2014"):
        assert by_year[y]["native_chip_array_shape"] == [224, 224, NAIP_BAND_COUNT]
        assert by_year[y]["native_pixel_size_m"] == pytest.approx(1.0, abs=1e-6)
    for y in ("2016", "2018", "2020", "2022"):
        assert by_year[y]["native_chip_array_shape"] == [373, 373, NAIP_BAND_COUNT]
        assert by_year[y]["native_pixel_size_m"] == pytest.approx(0.6, abs=1e-3)


def test_index_chips_have_correct_bbox_shape(real_chip_index):
    """Every chip has a 224 m square in 26916 with stride-aligned origin."""
    for cycle in real_chip_index["cycles"]:
        for ch in cycle["chips"]:
            w, s, e, n = ch["bbox_26916"]
            assert e - w == pytest.approx(CHIP_SIZE_M, abs=1e-9)
            assert n - s == pytest.approx(CHIP_SIZE_M, abs=1e-9)
            assert w % STRIDE_M == 0
            assert s % STRIDE_M == 0


def test_index_center_wgs84_roundtrips_to_center_26916(real_chip_index):
    """center_wgs84 -> 26916 should match center_26916 to within 1 cm."""
    from terra_query.core.crs import to_working

    fwd = to_working()
    for cycle in real_chip_index["cycles"]:
        for ch in cycle["chips"]:
            lon, lat = ch["center_wgs84"]
            x, y = fwd.transform(lon, lat)
            cx, cy = ch["center_26916"]
            assert abs(x - cx) < 0.01
            assert abs(y - cy) < 0.01


def test_index_inside_aoi_flag(real_chip_index):
    """Chips fully inside the AOI polygon are flagged true; corner-only chips false."""
    interior_count = 0
    boundary_or_outside_count = 0
    for cycle in real_chip_index["cycles"][:1]:  # same grid in 26916, check one cycle
        for ch in cycle["chips"]:
            if ch["inside_aoi"]:
                interior_count += 1
            else:
                boundary_or_outside_count += 1
    # AOI is ~5.1 km square; chips are 224 m. Interior chips (fully inside the
    # polygon) cover ~(5.1 - 0.224)^2 ~24 km^2, vs grid area ~(5.7)^2 ~32 km^2.
    # Expect a large interior count and a non-trivial boundary count.
    assert interior_count > 0
    assert boundary_or_outside_count > 0
    assert interior_count > boundary_or_outside_count


def test_index_eval_lookup_has_six_chips_per_feature(real_chip_index):
    eval_lookup = real_chip_index["eval_lookup"]
    assert len(eval_lookup) == 11
    for fid, entries in eval_lookup.items():
        assert len(entries) == 6
        years = [e["year"] for e in entries]
        assert sorted(years) == ["2012", "2014", "2016", "2018", "2020", "2022"]


def test_index_eval_chips_agree_across_cycles(real_chip_index):
    """Per (feature, cycle) the chip_id varies only by year - row/col are identical."""
    eval_lookup = real_chip_index["eval_lookup"]
    for fid, entries in eval_lookup.items():
        rowcols = {tuple(e["chip_id"].split("_")[-2:]) for e in entries}
        assert len(rowcols) == 1, f"{fid} has different row/col across cycles: {rowcols}"


def test_index_eval_chip_bbox_contains_feature_point(real_chip_index):
    """Each eval feature's 26916 point must lie inside its primary chip's bbox."""
    eval_lookup = real_chip_index["eval_lookup"]
    # build a (cycle, chip_id) -> bbox lookup
    bbox_by_id: dict[str, tuple[float, float, float, float]] = {}
    for cycle in real_chip_index["cycles"]:
        for ch in cycle["chips"]:
            bbox_by_id[ch["chip_id"]] = tuple(ch["bbox_26916"])
    # eval points in 26916
    ev_gj = json.loads(EVAL_26916.read_text())
    pts = {f["properties"]["id"]: tuple(f["geometry"]["coordinates"]) for f in ev_gj["features"]}
    for fid, entries in eval_lookup.items():
        x, y = pts[fid]
        for e in entries:
            w, s, eb, nb = bbox_by_id[e["chip_id"]]
            assert w <= x < eb, f"{fid} x={x} not in [{w},{eb})"
            assert s <= y < nb, f"{fid} y={y} not in [{s},{nb})"


# ---- step 4: read_chip + eval-chip PNG ---------------------------------


def _primary_chip_for(real_chip_index, fid: str, year: str) -> dict:
    chip_id_target = next(
        e["chip_id"] for e in real_chip_index["eval_lookup"][fid] if e["year"] == year
    )
    for cycle in real_chip_index["cycles"]:
        if cycle["year"] == year:
            for ch in cycle["chips"]:
                if ch["chip_id"] == chip_id_target:
                    return ch
    raise AssertionError(f"no chip {chip_id_target} in cycle {year}")


def test_read_chip_rgb_shape_matches_native(real_chip_index):
    """RGB read returns (3, native_h, native_w) for the cycle."""
    ch_2022 = _primary_chip_for(real_chip_index, "bond-falls", "2022")
    cb = ChipBox(row=ch_2022["row"], col=ch_2022["col"], west=ch_2022["bbox_26916"][0], south=ch_2022["bbox_26916"][1])
    arr = read_chip(cb, naip_cog("2022"), bands=RGB_BANDS)
    assert arr.shape == (3, 373, 373)
    assert arr.dtype.name == "uint8"
    # NAIP has no in-AOI nodata - sanity that the read isn't all zero
    assert arr.any()


def test_read_chip_cir_shape(real_chip_index):
    """CIR (NIR-R-G) returns 3 bands."""
    ch = _primary_chip_for(real_chip_index, "bond-falls", "2022")
    cb = ChipBox(row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1])
    arr = read_chip(cb, naip_cog("2022"), bands=CIR_BANDS)
    assert arr.shape == (3, 373, 373)


def test_read_chip_four_band_shape(real_chip_index):
    """All 4 bands return (4, h, w)."""
    ch = _primary_chip_for(real_chip_index, "bond-falls", "2022")
    cb = ChipBox(row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1])
    arr = read_chip(cb, naip_cog("2022"), bands=(1, 2, 3, 4))
    assert arr.shape == (4, 373, 373)


def test_read_chip_1m_cycle_shape(real_chip_index):
    """1.0 m cycle (2012) returns (3, 224, 224)."""
    ch = _primary_chip_for(real_chip_index, "bond-falls", "2012")
    cb = ChipBox(row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1])
    arr = read_chip(cb, naip_cog("2012"), bands=RGB_BANDS)
    assert arr.shape == (3, 224, 224)


def test_read_chip_rejects_invalid_bands():
    cb = ChipBox(row=0, col=0, west=333424.0, south=5139120.0)
    with pytest.raises(ValueError):
        read_chip(cb, naip_cog("2022"), bands=(0,))
    with pytest.raises(ValueError):
        read_chip(cb, naip_cog("2022"), bands=(5,))
    with pytest.raises(ValueError):
        read_chip(cb, naip_cog("2022"), bands=())


def test_grid_has_no_holes_inside_aoi(real_chip_index):
    """Every point inside the AOI is contained by >= 1 chip."""
    chips_recs = real_chip_index["cycles"][0]["chips"]
    bboxes = [tuple(ch["bbox_26916"]) for ch in chips_recs]
    w, s, e, n = AOI_BOUNDS
    # sample a 10x10 grid of points strictly inside the AOI
    holes = 0
    for i in range(1, 10):
        for j in range(1, 10):
            x = w + (e - w) * i / 10.0
            y = s + (n - s) * j / 10.0
            if not any(bw <= x < be and bs_ <= y < bn for bw, bs_, be, bn in bboxes):
                holes += 1
    assert holes == 0


# ---- step 5: CLI artifacts on disk ---------------------------------------


def test_cli_chip_index_json_on_disk():
    """The CLI writes the chip index JSON to disk and it round-trips."""
    from terra_query.core.paths import CHIP_INDEX_JSON

    assert CHIP_INDEX_JSON.exists()
    idx = json.loads(CHIP_INDEX_JSON.read_text())
    assert idx["working_crs"] == "EPSG:26916"
    assert sum(len(c["chips"]) for c in idx["cycles"]) == 13254


def test_cli_eval_chip_pngs_on_disk():
    """One PNG per eval feature, written by the CLI."""
    ev_gj = json.loads(EVAL_26916.read_text())
    fids = [f["properties"]["id"] for f in ev_gj["features"]]
    for fid in fids:
        p = chip_eval(fid)
        assert p.exists(), f"missing eval chip png: {p}"
        # PNG file sanity (header bytes)
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_grid_overview_png_on_disk():
    from terra_query.core.paths import CHIP_GRID_OVERVIEW_PNG

    assert CHIP_GRID_OVERVIEW_PNG.exists()
    assert CHIP_GRID_OVERVIEW_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_eval_chip_png_writes_png_with_crosshair(real_chip_index, tmp_path):
    """PNG opens at native size and the feature pixel is the crosshair color."""
    import numpy as np
    from PIL import Image

    fid = "bond-falls-main-dam"
    ch = _primary_chip_for(real_chip_index, fid, "2022")
    cb = ChipBox(row=ch["row"], col=ch["col"], west=ch["bbox_26916"][0], south=ch["bbox_26916"][1])

    ev_gj = json.loads(EVAL_26916.read_text())
    pt = next(
        tuple(f["geometry"]["coordinates"])
        for f in ev_gj["features"]
        if f["properties"]["id"] == fid
    )

    out = tmp_path / f"{fid}.png"
    render_eval_chip_png(cb, naip_cog("2022"), pt, out)

    assert out.exists()
    img = np.array(Image.open(out))
    assert img.shape == (373, 373, 3)

    # the feature pixel should be inside the chip and within 1 px of expected,
    # but the crosshair has a 2 px gap at center - check the line pixels just
    # outside the gap match the magenta crosshair color.
    pixel_size = (cb.east - cb.west) / img.shape[1]
    col = int(round((pt[0] - cb.west) / pixel_size))
    row = int(round((cb.north - pt[1]) / pixel_size))
    # check pixels 3-5 left/right and up/down (after the 2 px gap) for magenta
    samples = [
        (row, col - 4),
        (row, col + 4),
        (row - 4, col),
        (row + 4, col),
    ]
    matches = sum(1 for (r, c) in samples if tuple(img[r, c]) == (255, 0, 255))
    assert matches >= 3, f"expected magenta on crosshair arms; samples={[tuple(img[r,c]) for r,c in samples]}"
