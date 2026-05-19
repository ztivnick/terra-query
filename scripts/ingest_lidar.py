#!/usr/bin/env python3
"""Download 3DEP 1m DEM tiles for an AOI and save as COGs in EPSG:26916."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from terra_query.ingest.aoi import aoi_bounds_4326
from terra_query.ingest.lidar import download_and_reproject, fetch_3dep_product_urls

app = typer.Typer(add_completion=False)

# confirmed in Step 2a recon
_DATASET_TAG = "Digital Elevation Model (DEM) 1 meter"


@app.command()
def main(
    aoi: Path = typer.Option(Path("config/aoi_test.geojson"), "--aoi", help="GeoJSON AOI file"),
    out_dir: Path = typer.Option(
        Path("data/raw/lidar/dem"), "--out-dir", help="Output directory for DEM tiles"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log actions, no downloads"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Download 3DEP 1m DEM tiles for the AOI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(handlers=[RichHandler()], level=level, format="%(message)s")
    log = logging.getLogger(__name__)

    bounds = aoi_bounds_4326(aoi)
    log.info("AOI bounds (4326): %s", bounds)

    urls = fetch_3dep_product_urls(bounds, _DATASET_TAG)
    log.info("found %d tile(s)", len(urls))

    for url in urls:
        tile_name = url.rsplit("/", 1)[-1]
        out_path = out_dir / tile_name
        download_and_reproject(url, out_path, dst_crs=26916, dry_run=dry_run)

    log.info("done")


if __name__ == "__main__":
    app()
