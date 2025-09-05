"""
Download scenes from the Microsoft Planetary Computer STAC API.
This module searches for the best scene(s) covering a bbox and date range and
downloads the necessary spectral bands and QA/SCL assets.

Notes:
- We use the Planetary Computer STAC API because it exposes both Sentinel-2 L2A
  and Landsat Collection 2 L2 datasets as cloud-optimized GeoTIFFs (COGs) and
  provides signed URLs via the planetary_computer helper. This avoids scraping
  and uses documented APIs.
- For Copernicus's official portal (Copernicus Open Access Hub) or USGS M2M you
  can follow the docs linked in the README; this example focuses on a single
  reproducible path using Planetary Computer / public COGs.
"""

import os
import requests
from pathlib import Path
from shapely.geometry import box, mapping
from pystac_client import Client
import planetary_computer as pc


def pc_client():
    """Open the Planetary Computer STAC and return a client that signs results.
    This adds SAS tokens to the returned item.assets hrefs automatically.
    """
    return Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/", modifier=pc.sign_inplace)


def find_best_scene(collection_id, bbox, start_date, end_date, max_cloud=30, limit=50):
    """Search for scenes in the given collection that intersect the bbox and date range.

    Parameters
    - collection_id: e.g. 'sentinel-2-l2a' or 'landsat-c2-l2'
    - bbox: (minx, miny, maxx, maxy)
    - start_date/end_date: 'YYYY-MM-DD'
    - max_cloud: maximum acceptable cloud cover percentage

    Returns: first best-matching pystac.Item or None
    """
    client = pc_client()
    geom = mapping(box(*bbox))

    search = client.search(
        collections=[collection_id],
        intersects=geom,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )

    items = list(search.get_items())
    if not items:
        return None

    # sort by cloud cover then by datetime
    items.sort(key=lambda it: (it.properties.get('eo:cloud_cover', 100), it.properties.get('datetime')))
    return items[0]


def download_asset(href, out_path, chunk_size=1024 * 64):
    """Stream-download an HTTP(S) asset to disk with a simple progress courtesy of requests."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(href, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    return out_path


def download_scene_assets(item, band_keys, out_dir):
    """Given a STAC item, download the asset keys listed in `band_keys`.

    - item: pystac.Item (already signed via sign_inplace)
    - band_keys: list like ['B02','B03','B04','B08','SCL'] for Sentinel-2
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    for key in band_keys:
        if key in item.assets:
            href = item.assets[key].get('href', item.assets[key].href)
            filename = out_dir / f"{item.id}_{key}.tif"
            print(f"Downloading {key} -> {filename.name}")
            download_asset(href, filename)
            downloaded[key] = str(filename)
        else:
            print(f"Warning: asset {key} not found in item {item.id}")
    return downloaded


if __name__ == '__main__':
    # tiny CLI example
    import argparse

    parser = argparse.ArgumentParser(description='Find and download a best scene (Planetary Computer STAC)')
    parser.add_argument('--collection', default='sentinel-2-l2a')
    parser.add_argument('--bbox', required=True, help='bbox as minx,miny,maxx,maxy')
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--out', default='downloads')
    args = parser.parse_args()

    bbox = tuple(map(float, args.bbox.split(',')))
    item = find_best_scene(args.collection, bbox, args.start, args.end)
    if item is None:
        print('No items found')
    else:
        print('Selected item:', item.id)
        # recommended band_keys per collection
        if 'sentinel-2' in args.collection:
            band_keys = ['B02','B03','B04','B08','SCL']
        else:
            # Landsat Collection 2 Level-2 names typically use SR_B* asset keys
            band_keys = ['SR_B2','SR_B3','SR_B4','SR_B5','QA_PIXEL']
        downloaded = download_scene_assets(item, band_keys, args.out)
        print('Downloaded files:', downloaded)
