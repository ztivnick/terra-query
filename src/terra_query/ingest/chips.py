"""Chip grid for the buffered Bond Falls AOI (S4 / N6, NAIP-only).

A "chip" here is a 224 m x 224 m square in EPSG:26916. The grid is the
same in 26916 across every NAIP cycle; only the source pixel array
behind each chip differs (1.0 m native for 2012/2014, 0.6 m native for
2016+). The chip is defined by its bbox: pixel-level reads happen via
windowed reads against the source COG at the cycle's native resolution,
not at chip-cut time. The model's preprocessor at S5 resamples the
native array to its input size.

50% overlap (stride = 112 m) so a small feature near a chip boundary in
one chip is well inside an adjacent chip. Standard RS-retrieval default.

Module boundary: pure-Python chip-grid math sits at the top; rasterio
windowed reads sit at the bottom; PNG rendering sits last. Importing
this module does not import rasterio or PIL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CHIP_SIZE_M = 224
STRIDE_M = 112
NAIP_BAND_COUNT = 4  # R, G, B, NIR
RGB_BANDS = (1, 2, 3)
CIR_BANDS = (4, 1, 2)  # NIR, R, G - standard pseudo-color for vegetation/water


@dataclass(frozen=True)
class ChipBox:
    """One chip slot in the 26916 grid. Cycle-independent."""

    row: int
    col: int
    west: float
    south: float

    @property
    def east(self) -> float:
        return self.west + CHIP_SIZE_M

    @property
    def north(self) -> float:
        return self.south + CHIP_SIZE_M

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.west, self.south, self.east, self.north)

    @property
    def center(self) -> tuple[float, float]:
        return (self.west + CHIP_SIZE_M / 2.0, self.south + CHIP_SIZE_M / 2.0)


def chip_id(year: str, row: int, col: int) -> str:
    """Stable ID; decodes to a bbox given the grid origin."""
    return f"naip_{year}_r{row:03d}_c{col:03d}"


def grid_origin(
    aoi_bounds_26916: tuple[float, float, float, float],
    stride_m: int = STRIDE_M,
) -> tuple[int, int]:
    """Largest multiple of stride <= (west, south) of the AOI bounds.

    Aligned to the AOI itself (not the buffered AOI) so chips at the AOI
    edge extend into the buffer band rather than crossing the COG boundary.
    """
    west, south, _, _ = aoi_bounds_26916
    ow = int(math.floor(west / stride_m) * stride_m)
    os = int(math.floor(south / stride_m) * stride_m)
    return ow, os


def grid_dims(
    aoi_bounds_26916: tuple[float, float, float, float],
    chip_size_m: int = CHIP_SIZE_M,
    stride_m: int = STRIDE_M,
) -> tuple[int, int]:
    """(n_rows, n_cols) such that every chip that overlaps the AOI is included."""
    _, _, east, north = aoi_bounds_26916
    ow, os = grid_origin(aoi_bounds_26916, stride_m)
    n_rows = max(1, int(math.ceil((north - os) / stride_m)))
    n_cols = max(1, int(math.ceil((east - ow) / stride_m)))
    return n_rows, n_cols


def build_chip_grid(
    aoi_bounds_26916: tuple[float, float, float, float],
    chip_size_m: int = CHIP_SIZE_M,
    stride_m: int = STRIDE_M,
) -> list[ChipBox]:
    """Return every chip that overlaps the AOI bbox.

    50% overlap (stride = chip_size_m / 2): adjacent chips share half their area,
    so any interior point sits in up to 4 chips. Grid origin snaps to the largest
    multiple of stride <= (AOI_west, AOI_south), and dims size up to cover the
    AOI east/north edges. Chips at the AOI boundary extend into the S3 buffer
    band; the buffer's 250 m gives plenty of room before the chip would cross
    the COG edge.
    """
    ow, os = grid_origin(aoi_bounds_26916, stride_m)
    n_rows, n_cols = grid_dims(aoi_bounds_26916, chip_size_m, stride_m)
    chips: list[ChipBox] = []
    for r in range(n_rows):
        for c in range(n_cols):
            chips.append(
                ChipBox(
                    row=r,
                    col=c,
                    west=float(ow + c * stride_m),
                    south=float(os + r * stride_m),
                )
            )
    return chips


def containing_chips(
    point_26916: tuple[float, float],
    grid: Iterable[ChipBox],
) -> list[ChipBox]:
    """All chips whose bbox contains the point. Under 50% overlap, up to 4."""
    x, y = point_26916
    return [c for c in grid if c.west <= x < c.east and c.south <= y < c.north]


def primary_containing_chip(
    point_26916: tuple[float, float],
    grid: Iterable[ChipBox],
) -> ChipBox | None:
    """The chip whose center is closest to the point. Ties broken by (row, col)."""
    x, y = point_26916
    candidates = containing_chips((x, y), grid)
    if not candidates:
        return None

    def keyfn(c: ChipBox):
        cx, cy = c.center
        return ((cx - x) ** 2 + (cy - y) ** 2, c.row, c.col)

    return min(candidates, key=keyfn)


def _native_pixel_size_from_cog(cog_path: Path) -> float:
    """Read pixel size off the COG's transform. Trust the file, not the plan."""
    import rasterio

    with rasterio.open(cog_path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 26916:
            raise RuntimeError(f"{cog_path} not in EPSG:26916")
        return float(abs(ds.transform.a))


def _native_chip_array_shape(
    pixel_size_m: float, chip_size_m: int, band_count: int
) -> tuple[int, int, int]:
    """Expected (h, w, bands) for a chip read at the cycle's native res."""
    side_px = int(round(chip_size_m / pixel_size_m))
    return (side_px, side_px, band_count)


def assemble_chip_index(
    aoi_polygon_26916,
    eval_points_26916: list[tuple[str, tuple[float, float]]],
    naip_cycle_cogs: list[tuple[str, Path]],
    aoi_bounds_26916: tuple[float, float, float, float],
    aoi_buffered_bounds_26916: tuple[float, float, float, float],
    chip_size_m: int = CHIP_SIZE_M,
    stride_m: int = STRIDE_M,
) -> dict:
    """Build the full chip index.

    `aoi_polygon_26916` is a shapely Polygon in 26916. `eval_points_26916` is
    a list of (feature_id, (easting, northing)). `naip_cycle_cogs` is a list
    of (year, cog_path); pixel size per cycle is read from the COG transform
    so the index reflects what is actually on disk, not what the S3 plan
    claimed.

    Returns the index dict; caller writes it to disk.
    """
    from datetime import datetime, timezone

    from shapely.geometry import box, mapping  # noqa: F401  (mapping kept for debug)

    from terra_query.core.crs import to_wgs84

    chips = build_chip_grid(aoi_bounds_26916, chip_size_m, stride_m)
    ow, os_ = grid_origin(aoi_bounds_26916, stride_m)
    n_rows, n_cols = grid_dims(aoi_bounds_26916, chip_size_m, stride_m)
    bwd = to_wgs84()

    # cache inside_aoi per ChipBox so we don't reproject and re-check per cycle
    inside_flags: dict[tuple[int, int], bool] = {}
    centers_wgs84: dict[tuple[int, int], tuple[float, float]] = {}
    for c in chips:
        inside_flags[(c.row, c.col)] = aoi_polygon_26916.contains(box(*c.bbox))
        lon, lat = bwd.transform(*c.center)
        centers_wgs84[(c.row, c.col)] = (float(lon), float(lat))

    cycles_block: list[dict] = []
    for year, cog_path in sorted(naip_cycle_cogs):
        pixel_size = _native_pixel_size_from_cog(cog_path)
        h, w, bcount = _native_chip_array_shape(pixel_size, chip_size_m, NAIP_BAND_COUNT)
        cycle_chips: list[dict] = []
        for c in chips:
            cycle_chips.append(
                {
                    "chip_id": chip_id(year, c.row, c.col),
                    "row": c.row,
                    "col": c.col,
                    "bbox_26916": [c.west, c.south, c.east, c.north],
                    "center_26916": [c.center[0], c.center[1]],
                    "center_wgs84": list(centers_wgs84[(c.row, c.col)]),
                    "inside_aoi": bool(inside_flags[(c.row, c.col)]),
                }
            )
        cycles_block.append(
            {
                "year": year,
                "source_cog": str(cog_path),
                "native_pixel_size_m": pixel_size,
                "native_chip_array_shape": [h, w, bcount],
                "chips": cycle_chips,
            }
        )

    # eval_lookup: per feature, the primary chip per cycle (one entry per cycle)
    eval_lookup: dict[str, list[dict]] = {}
    for fid, pt in eval_points_26916:
        primary = primary_containing_chip(pt, chips)
        if primary is None:
            raise RuntimeError(f"eval feature {fid} at {pt} is outside the chip grid")
        eval_lookup[fid] = [
            {"year": year, "chip_id": chip_id(year, primary.row, primary.col)}
            for year, _ in sorted(naip_cycle_cogs)
        ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "working_crs": "EPSG:26916",
        "chip_size_m": chip_size_m,
        "stride_m": stride_m,
        "grid_origin_26916": [ow, os_],
        "n_rows": n_rows,
        "n_cols": n_cols,
        "aoi_bounds_26916": list(aoi_bounds_26916),
        "aoi_buffered_bounds_26916": list(aoi_buffered_bounds_26916),
        "band_count_available": NAIP_BAND_COUNT,
        "rgb_bands": list(RGB_BANDS),
        "cir_bands": list(CIR_BANDS),
        "cycles": cycles_block,
        "eval_lookup": eval_lookup,
    }


def read_chip(
    chip: ChipBox,
    source_cog_path: Path,
    bands: tuple[int, ...] = RGB_BANDS,
):
    """Windowed read of the chip's bbox from the source COG at native res.

    Returns a numpy array of shape (n_bands, native_h, native_w), uint8.
    No resampling: the array is the source cycle's native pixels clipped
    to the chip bbox. S5's model preprocessor handles the resize to the
    model's input size.

    `bands` is 1-indexed (rasterio convention): (1,2,3)=RGB, (4,1,2)=CIR
    (NIR-R-G), (1,2,3,4)=4-band. Any subset of {1,2,3,4} works.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    if not bands or any(b < 1 or b > NAIP_BAND_COUNT for b in bands):
        raise ValueError(f"bands must be 1-indexed subset of 1..{NAIP_BAND_COUNT}, got {bands}")

    with rasterio.open(source_cog_path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 26916:
            raise RuntimeError(f"{source_cog_path} not in EPSG:26916")
        win = from_bounds(*chip.bbox, transform=ds.transform)
        # snap to integer pixel offsets - the bbox is on a 112-m stride, native
        # pixel size is 0.6 or 1.0 m so the window aligns to integers within
        # rasterio's float-rounding tolerance.
        rounded = win.round_offsets(op="floor").round_lengths(op="ceil")
        arr = ds.read(
            indexes=list(bands),
            window=rounded,
            out_dtype=np.uint8,
            resampling=Resampling.nearest,  # no resampling at chip-cut; read native pixels
        )
    return arr


def _feature_pixel_offset(
    chip: ChipBox, feature_point_26916: tuple[float, float], pixel_size_m: float
) -> tuple[int, int]:
    """Pixel offset (col, row) of the feature point inside the chip.

    Rasterio convention: row 0 = north edge, increasing south. Col 0 = west.
    """
    fx, fy = feature_point_26916
    col = int(round((fx - chip.west) / pixel_size_m))
    row = int(round((chip.north - fy) / pixel_size_m))
    return col, row


def render_eval_chip_png(
    chip: ChipBox,
    source_cog_path: Path,
    feature_point_26916: tuple[float, float],
    out_path: Path,
    crosshair_radius_px: int = 12,
    crosshair_color: tuple[int, int, int] = (255, 0, 255),
) -> None:
    """Render the chip's RGB at source-native res with a crosshair on the feature.

    PNG output. Magenta crosshair survives on every NAIP background
    (forest green, road tan, water dark) without coloring like NAIP itself.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    rgb = read_chip(chip, source_cog_path, bands=RGB_BANDS)  # (3, h, w)
    hwc = np.transpose(rgb, (1, 2, 0))  # (h, w, 3)
    img = Image.fromarray(hwc, mode="RGB")

    pixel_size = (chip.east - chip.west) / hwc.shape[1]
    col, row = _feature_pixel_offset(chip, feature_point_26916, pixel_size)

    draw = ImageDraw.Draw(img)
    r = crosshair_radius_px
    # crosshair: horizontal + vertical 1-px lines, with a small gap at center
    gap = 2
    draw.line([(col - r, row), (col - gap, row)], fill=crosshair_color, width=2)
    draw.line([(col + gap, row), (col + r, row)], fill=crosshair_color, width=2)
    draw.line([(col, row - r), (col, row - gap)], fill=crosshair_color, width=2)
    draw.line([(col, row + gap), (col, row + r)], fill=crosshair_color, width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def render_chip_grid_overview(
    grid: Iterable[ChipBox],
    source_cog_path: Path,
    aoi_polygon_26916_ring: list[tuple[float, float]],
    eval_points_26916: list[tuple[str, tuple[float, float]]],
    out_path: Path,
    overview_width_px: int = 1500,
    stride_line_every: int = 4,
) -> None:
    """Render a screen-sized RGB overview with the chip grid, AOI, and eval points.

    Reads the source COG downsampled to `overview_width_px` wide (height
    proportional), draws stride lines at every `stride_line_every` strides
    (default every 4 strides = every 448 m, so a 47x47 grid renders as
    ~12x12 boxes), draws the AOI polygon outline, and labels every eval
    point.
    """
    import numpy as np
    import rasterio
    from PIL import Image, ImageDraw, ImageFont
    from rasterio.enums import Resampling

    with rasterio.open(source_cog_path) as ds:
        w_m = ds.bounds.right - ds.bounds.left
        h_m = ds.bounds.top - ds.bounds.bottom
        h_out = int(round(overview_width_px * h_m / w_m))
        arr = ds.read(
            indexes=[1, 2, 3],
            out_shape=(3, h_out, overview_width_px),
            resampling=Resampling.average,
        )
        cog_bounds = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

    img = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    px_per_m_x = overview_width_px / w_m
    px_per_m_y = h_out / h_m

    def world_to_px(x: float, y: float) -> tuple[int, int]:
        col = int(round((x - cog_bounds[0]) * px_per_m_x))
        row = int(round((cog_bounds[3] - y) * px_per_m_y))
        return col, row

    # stride lines (every Nth stride)
    chips_list = list(grid)
    if chips_list:
        westmost = min(c.west for c in chips_list)
        eastmost = max(c.east for c in chips_list)
        southmost = min(c.south for c in chips_list)
        northmost = max(c.north for c in chips_list)
        x = westmost
        i = 0
        while x <= eastmost:
            if i % stride_line_every == 0:
                col1, row1 = world_to_px(x, southmost)
                col2, row2 = world_to_px(x, northmost)
                draw.line([(col1, row1), (col2, row2)], fill=(255, 255, 255, 120), width=1)
            x += STRIDE_M
            i += 1
        y = southmost
        j = 0
        while y <= northmost:
            if j % stride_line_every == 0:
                col1, row1 = world_to_px(westmost, y)
                col2, row2 = world_to_px(eastmost, y)
                draw.line([(col1, row1), (col2, row2)], fill=(255, 255, 255, 120), width=1)
            y += STRIDE_M
            j += 1

    # AOI polygon outline (cyan)
    aoi_px = [world_to_px(x, y) for x, y in aoi_polygon_26916_ring]
    for i in range(len(aoi_px) - 1):
        draw.line([aoi_px[i], aoi_px[i + 1]], fill=(0, 255, 255, 255), width=3)

    # eval points + labels (yellow dot, white text)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except OSError:
        font = ImageFont.load_default()
    for fid, (x, y) in eval_points_26916:
        col, row = world_to_px(x, y)
        r = 5
        draw.ellipse([col - r, row - r, col + r, row + r], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255))
        draw.text((col + 8, row - 8), fid, fill=(255, 255, 255, 255), font=font, stroke_width=1, stroke_fill=(0, 0, 0, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
