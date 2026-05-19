#!/usr/bin/env python3
"""Download OSM features for an AOI via Overpass and save as GeoParquet."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from terra_query.ingest.aoi import aoi_bounds_4326
from terra_query.ingest.osm import DEFAULT_TAGS, fetch_osm_features, save_osm

app = typer.Typer(add_completion=False)


@app.command()
def main(
    aoi: Path = typer.Option(Path("config/aoi_test.geojson"), "--aoi", help="GeoJSON AOI file"),
    out_dir: Path = typer.Option(Path("data/raw/osm"), "--out-dir", help="Output directory"),
    overpass_url: str = typer.Option(
        "https://overpass-api.de/api/interpreter",
        "--overpass-url",
        help="Overpass API endpoint (swap for mirror if rate-limited)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log actions, no downloads"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Download OSM features (roads, buildings, natural, waterways) for the AOI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(handlers=[RichHandler()], level=level, format="%(message)s")
    log = logging.getLogger(__name__)

    bounds = aoi_bounds_4326(aoi)
    log.info("AOI bounds (4326): %s", bounds)

    gdf = fetch_osm_features(bounds, tags=DEFAULT_TAGS, overpass_url=overpass_url)
    log.info("fetched %d features", len(gdf))

    out_path = out_dir / "osm_aoi.parquet"
    save_osm(gdf, out_path, dry_run=dry_run)
    log.info("done")


if __name__ == "__main__":
    app()
