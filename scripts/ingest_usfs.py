#!/usr/bin/env python3
"""Download USFS Ottawa NF administrative boundary and save as GeoParquet."""

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from terra_query.ingest.usfs import fetch_usfs_boundary, save_usfs

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Path = typer.Option(Path("data/raw/usfs"), "--out-dir", help="Output directory"),
    layer_url: str = typer.Option(
        "https://apps.fs.usda.gov/arcgis/rest/services/EDW"
        "/EDW_ForestSystemBoundaries_01/MapServer/0",
        "--layer-url",
        help="ArcGIS REST layer URL (override if default 403s)",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log actions, no downloads"),
    verbose: bool = typer.Option(False, "--verbose", help="Debug logging"),
) -> None:
    """Download Ottawa NF administrative boundary polygon."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(handlers=[RichHandler()], level=level, format="%(message)s")
    log = logging.getLogger(__name__)

    gdf = fetch_usfs_boundary(forest_name="Ottawa National Forest", layer_url=layer_url)
    log.info("fetched %d polygon(s)", len(gdf))

    out_path = out_dir / "ottawa_nf_boundary.parquet"
    save_usfs(gdf, out_path, dry_run=dry_run)
    log.info("done")


if __name__ == "__main__":
    app()
