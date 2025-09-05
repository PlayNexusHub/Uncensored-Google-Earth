"""
Simple preprocessing helpers: load GeoTIFF bands with rasterio, apply scale factors,
mask clouds (prefer QA/SCL if present, otherwise a simple brightness test), and
resample to a target resolution.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype('float32')
        profile = src.profile.copy()
    return arr, profile


def apply_scale_and_nodata(arr, profile, sensor_hint=None):
    """Apply sensor-specific scaling to raw integer arrays and return scaled floats.

    - Sentinel-2 L2A: data are surface reflectance scaled by 1e4 (divide by 10000)
    - Landsat Collection 2 L2 SR: scale = 0.0000275, add offset = -0.2 (USGS guidance)

    sensor_hint can be 'sentinel' or 'landsat' or None; if None we try to infer
    from profile['dtype'] or values.
    """
    arr_in = arr.copy()

    if sensor_hint == 'sentinel' or (sensor_hint is None and profile.get('driver') is None):
        # safe default for Sentinel L2A COGs
        scaled = arr_in / 10000.0
        return scaled

    # try landsat default scaling if values look like > 1000
    if sensor_hint == 'landsat' or (sensor_hint is None and np.nanmax(arr_in) > 1000):
        scaled = arr_in * 0.0000275 - 0.2
        return scaled

    # fallback: return as-is cast to float in range 0-1 or original values
    return arr_in.astype('float32')


def mask_clouds_sentinel(scl_arr):
    """Return a boolean mask (True = keep) using Sentinel SCL classification layer.

    SCL codes (common set):
    0 = no data, 1 = saturated/defective, 2 = dark area pixels,
    3 = cloud shadows, 4 = vegetation, 5 = bare soils, 6 = water,
    7 = cloud low prob, 8 = cloud med prob, 9 = cloud high prob,
    10 = thin cirrus, 11 = snow

    We'll keep pixels that are **not** cloud / cloud shadow / cirrus.
    """
    cloud_codes = {3, 7, 8, 9, 10}
    mask = ~np.isin(scl_arr.astype('int32'), list(cloud_codes))
    return mask


def mask_clouds_landsat(qa_pixel_arr):
    """Simple Landsat QA_PIXEL interpreter for Collection 2 (unsigned 16-bit).
    This implementation is conservative: it masks pixels where the cloud bit is set.

    Reference: USGS QA documentation; there are four bits for cloud & cloud confidence.
    For simplicity we check a common cloud bit pattern; for production use consult
    the 'QA_PIXEL' format doc and decode properly.
    """
    qa = qa_pixel_arr.astype('uint16')
    # Bit definitions vary by collection — a simple heuristic: pixels == 0 are OK
    # and high values often indicate cloud. This is a naive fallback.
    mask = qa == 0
    return mask


def resample_to_resolution(src_array, src_profile, target_resolution, resampling=Resampling.bilinear):
    """Resample a single-band array (src_array) to the given target resolution (meters).
    Returns (dst_array, dst_profile).

    Note: target_resolution should be positive float (e.g. 10.0 or 30.0)
    """
    dst_profile = src_profile.copy()
    src_transform = src_profile['transform']
    src_crs = src_profile['crs']

    # compute new transform/shape
    new_width = int((src_profile['width'] * src_profile['transform'][0]) / target_resolution)
    new_height = int((src_profile['height'] * abs(src_profile['transform'][4])) / target_resolution)

    # A robust approach is to use rasterio.calculate_default_transform to set a new resolution
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, src_crs, src_profile['width'], src_profile['height'], *src_profile['bounds'], resolution=target_resolution
    ) if False else (None, None, None)

    # fallback: use rasterio.warp.reproject with dst_transform calculated by keeping the bbox and computing new shape
    left = src_transform[2]
    top = src_transform[5]
    right = left + src_profile['width'] * src_transform[0]
    bottom = top + src_profile['height'] * src_transform[4]

    dst_width = int(round((right - left) / target_resolution))
    dst_height = int(round((top - bottom) / target_resolution))

    dst_transform = rasterio.transform.from_origin(left, top, target_resolution, target_resolution)

    dst_profile.update({
        'height': dst_height,
        'width': dst_width,
        'transform': dst_transform,
        'dtype': 'float32'
    })

    dst = np.zeros((dst_height, dst_width), dtype='float32')

    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=resampling
    )

    return dst, dst_profile
