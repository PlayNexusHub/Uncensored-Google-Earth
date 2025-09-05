"""
High-level example that ties downloader, preprocess and indices together for
one or two dates (single-date outputs + change detection between two dates).

Usage example:
python run_workflow.py --bbox "-122.6,37.6,-122.3,37.9" --date1 2023-07-01 --date2 2023-07-25 --outdir outputs

Notes:
- This script is intentionally a clear, readable educational example, not a
  production-grade batch system.
"""

import argparse
from pathlib import Path
from downloader import find_best_scene, download_scene_assets
from preprocess import read_band, apply_scale_and_nodata, mask_clouds_sentinel, mask_clouds_landsat, resample_to_resolution
from indices import compute_ndvi, compute_ndwi, save_geotiff, simple_change_detection
import numpy as np
import rasterio
from rasterio.enums import Resampling


SENTINEL_BANDS = {'blue': 'B02', 'green': 'B03', 'red': 'B04', 'nir': 'B08', 'scl': 'SCL'}
LANDSAT_BANDS = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'qa': 'QA_PIXEL'}


def process_item(item, collection_hint, out_dir, target_resolution=10.0):
    # decide band keys and load
    if 'sentinel-2' in collection_hint:
        band_keys = list(SENTINEL_BANDS.values())
    else:
        band_keys = list(LANDSAT_BANDS.values())

    downloads = download_scene_assets(item, band_keys, out_dir)

    # read bands (with fallback if some are missing)
    band_arrays = {}
    profiles = {}
    for key, path in downloads.items():
        arr, prof = read_band(path)
        band_arrays[key] = arr
        profiles[key] = prof

    # apply scaling
    sensor_type = 'sentinel' if 'sentinel-2' in collection_hint else 'landsat'
    # map to variable names
    if sensor_type == 'sentinel':
        blue = apply_scale_and_nodata(band_arrays.get('B02'), profiles.get('B02'), 'sentinel')
        green = apply_scale_and_nodata(band_arrays.get('B03'), profiles.get('B03'), 'sentinel')
        red = apply_scale_and_nodata(band_arrays.get('B04'), profiles.get('B04'), 'sentinel')
        nir = apply_scale_and_nodata(band_arrays.get('B08'), profiles.get('B08'), 'sentinel')
        scl = band_arrays.get('SCL')
        mask = mask_clouds_sentinel(scl) if scl is not None else None
        profile = profiles.get('B04')
    else:
        blue = apply_scale_and_nodata(band_arrays.get('SR_B2'), profiles.get('SR_B2'), 'landsat')
        green = apply_scale_and_nodata(band_arrays.get('SR_B3'), profiles.get('SR_B3'), 'landsat')
        red = apply_scale_and_nodata(band_arrays.get('SR_B4'), profiles.get('SR_B4'), 'landsat')
        nir = apply_scale_and_nodata(band_arrays.get('SR_B5'), profiles.get('SR_B5'), 'landsat')
        qa = band_arrays.get('QA_PIXEL')
        mask = mask_clouds_landsat(qa) if qa is not None else None
        profile = profiles.get('SR_B4')

    # resample all to target_resolution
    red_r, prof_r = resample_to_resolution(red, profile, target_resolution)
    nir_r, _ = resample_to_resolution(nir, profile, target_resolution)
    green_r, _ = resample_to_resolution(green, profile, target_resolution)

    if mask is not None:
        mask_r, _ = resample_to_resolution(mask.astype('uint8'), profile, target_resolution, Resampling.nearest)
        mask_r = mask_r.astype(bool)
    else:
        mask_r = None

    ndvi = compute_ndvi(nir_r, red_r, mask=mask_r)
    ndwi = compute_ndwi(green_r, nir_r, mask=mask_r)

    # save products
    out_ndvi = Path(out_dir) / 'NDVI.tif'
    out_ndwi = Path(out_dir) / 'NDWI.tif'
    save_geotiff(out_ndvi, ndvi, prof_r, nodata=np.nan)
    save_geotiff(out_ndwi, ndwi, prof_r, nodata=np.nan)

    # also save an RGB quicklook (simple stretch)
    rgb = np.dstack([red_r, green_r, nir_r])
    # simple 2% linear stretch for visualization
    def stretch(img):
        p2 = np.nanpercentile(img, 2)
        p98 = np.nanpercentile(img, 98)
        return np.clip((img - p2) / (p98 - p2 + 1e-9), 0, 1)

    rgb_vis = stretch(rgb)

    # write RGB as 3-band GeoTIFF
    with rasterio.open(str(Path(out_dir) / 'TRUE_COLOR.tif'), 'w', driver='GTiff',
                       height=rgb_vis.shape[0], width=rgb_vis.shape[1], count=3,
                       dtype='float32', crs=prof_r['crs'], transform=prof_r['transform']) as dst:
        dst.write(rgb_vis[:, :, 0].astype('float32'), 1)
        dst.write(rgb_vis[:, :, 1].astype('float32'), 2)
        dst.write(rgb_vis[:, :, 2].astype('float32'), 3)

    return str(out_ndvi), str(out_ndwi), str(Path(out_dir) / 'TRUE_COLOR.tif')


def orchestrate(bbox, date1, date2=None, out_root='outputs', sensor_priority=('sentinel-2-l2a','landsat-c2-l2')):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # select scenes for date1 and optional date2
    docs = {}
    for idx, date in enumerate([date1, date2] if date2 else [date1]):
        if date is None:
            continue
        out_dir = out_root / f'date_{idx+1}'
        out_dir.mkdir(exist_ok=True)
        
        # try each sensor in priority order
        item = None
        collection_used = None
        for collection in sensor_priority:
            print(f"Searching {collection} for {date}...")
            item = find_best_scene(collection, bbox, date, date)
            if item is not None:
                collection_used = collection
                break
        
        if item is None:
            print(f"No suitable scene found for {date}")
            continue
            
        print(f"Processing {item.id} from {collection_used}")
        docs[f'date_{idx+1}'] = {
            'item': item,
            'collection': collection_used,
            'out_dir': out_dir
        }
    
    # process each found scene
    results = {}
    for date_key, doc in docs.items():
        print(f"\nProcessing {date_key}...")
        ndvi_path, ndwi_path, rgb_path = process_item(
            doc['item'], doc['collection'], doc['out_dir']
        )
        results[date_key] = {
            'ndvi': ndvi_path,
            'ndwi': ndwi_path,
            'rgb': rgb_path
        }
    
    # if we have two dates, compute change detection
    if len(results) == 2 and date2 is not None:
        print("\nComputing change detection...")
        change_dir = out_root / 'change_detection'
        change_dir.mkdir(exist_ok=True)
        
        # load NDVI from both dates
        ndvi1 = rasterio.open(results['date_1']['ndvi']).read(1)
        ndvi2 = rasterio.open(results['date_2']['ndvi']).read(1)
        
        # compute change
        diff, abs_change = simple_change_detection(ndvi1, ndvi2)
        
        # save change products
        change_profile = rasterio.open(results['date_1']['ndvi']).profile.copy()
        save_geotiff(change_dir / 'NDVI_DIFFERENCE.tif', diff, change_profile)
        save_geotiff(change_dir / 'NDVI_ABS_CHANGE.tif', abs_change, change_profile)
        
        print(f"Change detection saved to {change_dir}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Satellite imagery processing workflow')
    parser.add_argument('--bbox', required=True, help='bbox as minx,miny,maxx,maxy')
    parser.add_argument('--date1', required=True, help='First date (YYYY-MM-DD)')
    parser.add_argument('--date2', help='Second date for change detection (YYYY-MM-DD)')
    parser.add_argument('--outdir', default='outputs', help='Output directory')
    parser.add_argument('--sensor-priority', nargs='+', 
                       default=['sentinel-2-l2a', 'landsat-c2-l2'],
                       help='Sensor priority order')
    
    args = parser.parse_args()
    
    bbox = tuple(map(float, args.bbox.split(',')))
    results = orchestrate(
        bbox=bbox,
        date1=args.date1,
        date2=args.date2,
        out_root=args.outdir,
        sensor_priority=args.sensor_priority
    )
    
    print(f"\nWorkflow complete! Results saved to {args.outdir}")
    for date_key, paths in results.items():
        print(f"{date_key}: {paths}")
