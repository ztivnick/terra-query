#!/usr/bin/env python3
"""TNM reconnaissance script for Ottawa NF AOI."""

import json
from pathlib import Path

import requests

from terra_query.ingest.aoi import aoi_bounds_4326


def main():
    """Query TNM API for available 3DEP products for the AOI."""
    # Get AOI bounds
    aoi_path = Path("config/aoi_test.geojson")
    west, south, east, north = aoi_bounds_4326(aoi_path)

    # TNM API endpoint for products
    url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    # Query parameters
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "prodFormats": "GeoTIFF",
        "outputFormat": "JSON",
    }

    print(f"Querying TNM API for AOI bounds: {west}, {south}, {east}, {north}")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        print("TNM API Response:")
        print(json.dumps(data, indent=2))

    except requests.RequestException as e:
        print(f"Error querying TNM API: {e}")
        raise


if __name__ == "__main__":
    main()
