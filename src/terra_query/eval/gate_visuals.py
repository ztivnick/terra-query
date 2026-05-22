"""Aerial-ingest gate verification visuals.

Renders artifacts that prove the gate's manual-check items without QGIS:
  1. Side-by-side cross-cycle NAIP comparison (oldest vs latest) at the
     falls/dam area.
  2. Sentinel-2 winter RGB (B04/B03/B02) at the full AOI extent, one
     PNG per scene.
  3. NAIP RGB at the Bond Falls dam area with the dam eval point
     marked - proof of the dam-on-dam coordinate check on the actual
     mosaicked raster (not just a chip crop).

    uv run python -m terra_query.eval.gate_visuals
    uv run python -m terra_query.eval.gate_visuals --experiment /path/to/cfg.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import from_bounds

from terra_query.core import config
from terra_query.core.paths import (
    cross_cycle_png,
    dam_on_dam_png,
    eval_26916,
    gate_dir,
    naip_cog,
    naip_manifest,
    s2_manifest,
    s2_cog,
    s2_rgb_png,
)


def _stretch_uint8(arr: np.ndarray) -> np.ndarray:
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


def _stretch_s2_to_uint8(arr: np.ndarray) -> np.ndarray:
    """S2 L2A is uint16 surface reflectance scaled by 10000.
    Apply per-band 2-98 stretch like NAIP."""
    arr32 = arr.astype(np.float32)
    out = np.empty_like(arr32)
    for c in range(arr.shape[-1]):
        ch = arr32[..., c]
        lo, hi = np.percentile(ch[ch > 0], [2, 98]) if (ch > 0).any() else (0, 1)
        if hi <= lo:
            out[..., c] = 0
            continue
        scaled = (ch - lo) / (hi - lo) * 255
        out[..., c] = np.clip(scaled, 0, 255)
    return out.astype(np.uint8)


def _bond_falls_xy_26916(eval_fc: dict) -> tuple[float, float]:
    for feat in eval_fc["features"]:
        if feat["properties"]["id"] == "bond-falls":
            x, y = feat["geometry"]["coordinates"]
            return float(x), float(y)
    raise RuntimeError("bond-falls not in eval set")


def _dam_xy_26916(eval_fc: dict) -> tuple[float, float]:
    for feat in eval_fc["features"]:
        if feat["properties"]["id"] == "bond-falls-main-dam":
            x, y = feat["geometry"]["coordinates"]
            return float(x), float(y)
    raise RuntimeError("bond-falls-main-dam not in eval set")


def _read_window_rgb(naip_path: Path, x: float, y: float, half_m: float) -> Image.Image:
    """Read NAIP RGB around (x, y) and return a stretched PIL image."""
    with rasterio.open(naip_path) as ds:
        window = from_bounds(
            x - half_m, y - half_m, x + half_m, y + half_m, ds.transform
        )
        rgb = ds.read([1, 2, 3], window=window)
    rgb = np.transpose(rgb, (1, 2, 0))
    stretched = _stretch_uint8(rgb)
    return Image.fromarray(stretched)


def _naip_years(aoi_id: str) -> list[str]:
    manifest = json.loads(naip_manifest(aoi_id).read_text())
    return sorted(c["year"] for c in manifest["cycles"])


def render_cross_cycle(eval_set_id: str, aoi_id: str, out_path: Path) -> None:
    """Oldest-vs-latest NAIP side-by-side, 500 m around Bond Falls."""
    eval_fc = json.loads(eval_26916(eval_set_id).read_text())
    fx, fy = _bond_falls_xy_26916(eval_fc)

    years = _naip_years(aoi_id)
    earliest, latest = years[0], years[-1]
    img_a = _read_window_rgb(naip_cog(aoi_id, earliest), fx, fy, 500)
    img_b = _read_window_rgb(naip_cog(aoi_id, latest), fx, fy, 500)

    # equalise output sizes
    target_h = 500
    img_a = img_a.resize(
        (int(img_a.width * target_h / img_a.height), target_h), Image.LANCZOS
    )
    img_b = img_b.resize(
        (int(img_b.width * target_h / img_b.height), target_h), Image.LANCZOS
    )

    pad = 12
    label_h = 24
    total_w = img_a.width + img_b.width + 3 * pad
    total_h = target_h + label_h + 2 * pad
    canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
    canvas.paste(img_a, (pad, pad + label_h))
    canvas.paste(img_b, (img_a.width + 2 * pad, pad + label_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((pad, 6), f"NAIP {earliest}", fill="white", font=font)
    draw.text(
        (img_a.width + 2 * pad, 6),
        f"NAIP {latest}",
        fill="white",
        font=font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_s2_rgb(aoi_id: str, date_str: str, out_path: Path) -> None:
    """Render full-AOI S2 RGB (B04/B03/B02) for one scene."""
    scene_path = s2_cog(aoi_id, date_str)
    with rasterio.open(scene_path) as ds:
        # bands stored as B02 B03 B04 B08 B11 (positions 1..5)
        # RGB = R(B04 -> 3), G(B03 -> 2), B(B02 -> 1)
        rgb = ds.read([3, 2, 1])
    rgb = np.transpose(rgb, (1, 2, 0))
    stretched = _stretch_s2_to_uint8(rgb)
    img = Image.fromarray(stretched)

    # upscale 4x for viewing (S2 AOI is ~565 px wide)
    img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0))
    draw.text(
        (6, 4),
        f"Sentinel-2 RGB (B04/B03/B02), {date_str}, full AOI",
        fill="white",
        font=font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def render_dam_on_dam(eval_set_id: str, aoi_id: str, out_path: Path) -> None:
    """Render latest NAIP around the dam point with a marker on the dam coord."""
    eval_fc = json.loads(eval_26916(eval_set_id).read_text())
    dx, dy = _dam_xy_26916(eval_fc)
    latest = _naip_years(aoi_id)[-1]
    img = _read_window_rgb(naip_cog(aoi_id, latest), dx, dy, 200)
    img = img.resize((600, 600), Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.line([(cx - 30, cy), (cx + 30, cy)], fill="red", width=3)
    draw.line([(cx, cy - 30), (cx, cy + 30)], fill="red", width=3)
    draw.ellipse([(cx - 12, cy - 12), (cx + 12, cy + 12)], outline="red", width=3)
    draw.rectangle([(0, 0), (img.width, 22)], fill=(0, 0, 0))
    font = ImageFont.load_default()
    draw.text(
        (6, 4),
        f"NAIP {latest} around bond-falls-main-dam point",
        fill="white",
        font=font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


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

    gate_dir(experiment_id).mkdir(parents=True, exist_ok=True)

    cc_path = cross_cycle_png(experiment_id)
    render_cross_cycle(eval_set_id, aoi_id, cc_path)
    print(f"wrote {cc_path}")

    manifest = json.loads(s2_manifest(aoi_id).read_text())
    for scene in manifest["scenes"]:
        date_str = scene["datetime"][:10]
        out = s2_rgb_png(experiment_id, date_str)
        render_s2_rgb(aoi_id, date_str, out)
        print(f"wrote {out}")

    dd_path = dam_on_dam_png(experiment_id)
    render_dam_on_dam(eval_set_id, aoi_id, dd_path)
    print(f"wrote {dd_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
