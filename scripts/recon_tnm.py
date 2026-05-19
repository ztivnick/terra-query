#!/usr/bin/env python3
"""TNM reconnaissance - probe available 3DEP products for the Ottawa NF test AOI."""

import json
from pathlib import Path

import requests

from terra_query.ingest.aoi import aoi_bounds_4326


def query_tnm(
    bbox: str,
    datasets: str | None = None,
    prod_formats: str | None = None,
    label: str = "",
) -> dict:  # type: ignore[type-arg]
    """Hit the TNM /products endpoint and return the parsed JSON."""
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"
    params: dict[str, str] = {"bbox": bbox, "outputFormat": "JSON", "max": "50"}
    if datasets:
        params["datasets"] = datasets
    if prod_formats:
        params["prodFormats"] = prod_formats

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    count = data.get("total", len(data.get("items", [])))
    titles = [it.get("title", "") for it in data.get("items", [])]
    print(f"\n=== {label} === ({count} items) ===")
    for t in titles:
        print("  ", t)
    return data  # type: ignore[return-value]


def main() -> None:
    aoi_path = Path("config/aoi_test.geojson")
    w, s, e, n = aoi_bounds_4326(aoi_path)
    bbox = f"{w},{s},{e},{n}"
    print(f"AOI bbox (W,S,E,N): {bbox}")

    # 1. baseline - all GeoTIFF products (what the original recon ran)
    query_tnm(bbox, prod_formats="GeoTIFF", label="All GeoTIFF")

    # 2. explicit 1m DEM dataset string
    query_tnm(bbox, datasets="Digital Elevation Model (DEM) 1 meter", label="DEM 1m (GeoTIFF)")

    # 3. 1/3 arc-second DEM - what actually came back originally
    query_tnm(
        bbox,
        datasets="National Elevation Dataset (NED) 1/3 arc-second",
        label="NED 1/3 arc-sec",
    )

    # 4. LPC - raw point cloud tiles (LAZ); these exist even when 1m raster doesn't
    query_tnm(bbox, datasets="Lidar Point Cloud (LPC)", label="LPC (LAZ)")

    # 5. no filter at all - see every product type for this bbox
    all_data = query_tnm(bbox, label="ALL products (no filter)")
    print("\n=== FULL JSON (last query, all products) ===")
    print(json.dumps(all_data, indent=2))


if __name__ == "__main__":
    main()
