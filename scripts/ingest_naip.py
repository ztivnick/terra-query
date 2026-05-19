#!/usr/bin/env python3
"""Download NAIP RGBIR tiles for an AOI via Planetary Computer STAC."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from terra_query.ingest.aoi import aoi_bounds_4326
from terra_query.ingest.naip import download_naip_item, search_naip_items

app = typer.Typer(add_completion=False)


@app.command()
def main(
    aoi: Path = typer.Option(Path("config/aoi_test.geojson"), "--aoi", help="GeoJSON AOI file"),
    out_dir: Path = typer.Option(
        Path("data/raw/naip"), "--out-dir", help="Output directory for NAIP tiles"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log actions, no downloads"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Download most recent NAIP tiles for the AOI (windowed reads, 4-band RGBIR)."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(handlers=[RichHandler()], level=level, format="%(message)s")
    log = logging.getLogger(__name__)

    bounds = aoi_bounds_4326(aoi)
    log.info("AOI bounds (4326): %s", bounds)

    items = search_naip_items(bounds)
    log.info("found %d NAIP item(s)", len(items))

    for item in items:
        tile_id = item.id
        out_path = out_dir / f"{tile_id}_naip.tif"
        download_naip_item(item, out_path, bounds, dst_crs=26916, dry_run=dry_run)

    log.info("done")


if __name__ == "__main__":
    app()
