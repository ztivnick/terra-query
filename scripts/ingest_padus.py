#!/usr/bin/env python3
"""Download PAD-US v4.0 features for an AOI and save as GeoParquet."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from terra_query.ingest.aoi import aoi_bounds_4326
from terra_query.ingest.padus import fetch_padus_features, save_padus

app = typer.Typer(add_completion=False)


@app.command()
def main(
    aoi: Path = typer.Option(Path("config/aoi_test.geojson"), "--aoi", help="GeoJSON AOI file"),
    out_dir: Path = typer.Option(Path("data/raw/padus"), "--out-dir", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log actions, no downloads"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Download PAD-US v4.0 designation features intersecting the AOI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(handlers=[RichHandler()], level=level, format="%(message)s")
    log = logging.getLogger(__name__)

    bounds = aoi_bounds_4326(aoi)
    log.info("AOI bounds (4326): %s", bounds)

    gdf = fetch_padus_features(bounds)
    log.info("fetched %d features", len(gdf))

    out_path = out_dir / "padus_aoi.parquet"
    save_padus(gdf, out_path, dry_run=dry_run)
    log.info("done")


if __name__ == "__main__":
    app()
