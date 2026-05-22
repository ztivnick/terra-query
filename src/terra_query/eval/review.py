"""Re-verify `findable_aerial` per eval feature against the real NAIP.

Renders one 200 m x 200 m NAIP RGB chip per eval feature centered on the
feature with a crosshair marker, plus an AOI-wide overlay PNG with the
AOI polygon and all eval points. Updating the `findable_aerial` flag
itself is done out-of-band after visual inspection of the chips.

    uv run python -m terra_query.eval.review
    uv run python -m terra_query.eval.review --experiment /path/to/cfg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window

from terra_query.core import config
from terra_query.core.paths import (
    REPO_ROOT,
    aoi_26916,
    eval_26916,
    eval_chip,
    eval_chips_dir,
    naip_cog,
    naip_manifest,
    overlay_png,
)

CHIP_HALF_M = 100.0  # 200 m x 200 m chip per feature
CHIP_DISPLAY_PX = 400  # output PNG width, nearest-neighbor scaled


def latest_naip_cycle(aoi_id: str) -> tuple[str, Path]:
    """Return (year, cog_path) for the latest NAIP cycle in this AOI's manifest."""
    manifest = json.loads(naip_manifest(aoi_id).read_text())
    years = sorted(c["year"] for c in manifest["cycles"])
    if not years:
        raise RuntimeError(f"no NAIP cycles in manifest for aoi {aoi_id}")
    latest = years[-1]
    cog = naip_cog(aoi_id, latest)
    if not cog.exists():
        raise RuntimeError(f"NAIP COG missing for {aoi_id}/{latest}: {cog}")
    return latest, cog


def _stretch_uint8(arr: np.ndarray) -> np.ndarray:
    """Linear 2-98% percentile stretch per channel for visual contrast.

    NAIP RGB is naturally dark over closed-canopy forest; raw uint8 values
    cluster in the 40-100 range. Stretching makes feature boundaries
    visible without distorting interpretation.
    """
    out = np.empty_like(arr)
    for c in range(arr.shape[-1]):
        ch = arr[..., c]
        lo, hi = np.percentile(ch, [2, 98])
        if hi <= lo:
            out[..., c] = ch
            continue
        scaled = (ch.astype(np.float32) - lo) / (hi - lo) * 255
        out[..., c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def render_chip(ds, x: float, y: float, out_path: Path, label: str) -> None:
    res = abs(ds.transform.a)
    half_px = int(round(CHIP_HALF_M / res))
    row, col = ds.index(x, y)
    row_off = max(0, row - half_px)
    col_off = max(0, col - half_px)
    height = min(2 * half_px, ds.height - row_off)
    width = min(2 * half_px, ds.width - col_off)
    if height <= 0 or width <= 0:
        raise RuntimeError(f"chip out of raster for ({x}, {y})")
    window = Window(col_off, row_off, width, height)
    rgb = ds.read([1, 2, 3], window=window)
    rgb = np.transpose(rgb, (1, 2, 0))
    stretched = _stretch_uint8(rgb)
    img = Image.fromarray(stretched)
    scale = CHIP_DISPLAY_PX / img.width
    new_h = int(round(img.height * scale))
    img = img.resize((CHIP_DISPLAY_PX, new_h), Image.NEAREST)

    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.line([(cx - 22, cy), (cx + 22, cy)], fill="red", width=2)
    draw.line([(cx, cy - 22), (cx, cy + 22)], fill="red", width=2)
    draw.ellipse([(cx - 9, cy - 9), (cx + 9, cy + 9)], outline="red", width=2)

    font = ImageFont.load_default()
    bar_h = 20
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=(0, 0, 0))
    draw.text((4, 3), label, fill="white", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def render_overlay(
    naip_path: Path, aoi_26916_fc: dict, eval_26916_fc: dict, out_path: Path
) -> None:
    with rasterio.open(naip_path) as ds:
        scale = 6
        out_h = ds.height // scale
        out_w = ds.width // scale
        rgb = ds.read([1, 2, 3], out_shape=(3, out_h, out_w))
        rgb = np.transpose(rgb, (1, 2, 0))
        stretched = _stretch_uint8(rgb)
        img = Image.fromarray(stretched)
        bounds = ds.bounds
        px_x = abs(ds.transform.a) * scale
        px_y = abs(ds.transform.e) * scale

        def xy_to_px(x: float, y: float) -> tuple[float, float]:
            return (x - bounds.left) / px_x, (bounds.top - y) / px_y

    draw = ImageDraw.Draw(img)
    aoi_ring = aoi_26916_fc["features"][0]["geometry"]["coordinates"][0]
    px_ring = [xy_to_px(x, y) for x, y in aoi_ring]
    draw.line(px_ring, fill="yellow", width=3)

    font = ImageFont.load_default()
    for feat in eval_26916_fc["features"]:
        x, y = feat["geometry"]["coordinates"]
        cx, cy = xy_to_px(x, y)
        cat = feat["properties"]["category"]
        color = (
            "lime"
            if cat == "positive_in_scope"
            else ("red" if cat == "negative_mapped" else "white")
        )
        r = 6
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=color, width=2)
        draw.text((cx + r + 2, cy - 6), feat["properties"]["id"], fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def sample_at_point(ds, x: float, y: float) -> list[int] | None:
    row, col = ds.index(x, y)
    if not (0 <= row < ds.height and 0 <= col < ds.width):
        return None
    return [int(ds.read(b + 1)[row, col]) for b in range(ds.count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--experiment", type=Path, default=None,
        help="Path to experiment YAML; defaults to config resolution order.",
    )
    args = parser.parse_args()
    cfg = config.load_experiment(args.experiment)
    experiment_id = config.experiment_id_of(cfg)
    aoi_id = config.aoi_id_of(cfg)
    eval_set_id = config.eval_set_id_of(cfg)

    latest_year, naip_path = latest_naip_cycle(aoi_id)
    print(f"Using latest NAIP cycle {latest_year}: {naip_path.relative_to(REPO_ROOT)}")

    aoi_26916_fc = json.loads(aoi_26916(aoi_id).read_text())
    eval_26916_fc = json.loads(eval_26916(eval_set_id).read_text())

    eval_chips_dir(eval_set_id).mkdir(parents=True, exist_ok=True)
    with rasterio.open(naip_path) as ds:
        for feat in eval_26916_fc["features"]:
            fid = feat["properties"]["id"]
            x, y = feat["geometry"]["coordinates"]
            chip_path = eval_chip(eval_set_id, fid)
            render_chip(ds, x, y, chip_path, fid)
            sample = sample_at_point(ds, x, y)
            print(f"  {fid}: chip={chip_path.relative_to(REPO_ROOT)} sample={sample}")

    overlay_out = overlay_png(experiment_id)
    render_overlay(naip_path, aoi_26916_fc, eval_26916_fc, overlay_out)
    print(f"Overlay: {overlay_out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
