"""
Compute NDVI, NDWI, simple change detection, and write GeoTIFFs.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin


def compute_ndvi(nir, red, mask=None):
    nir = nir.astype('float32')
    red = red.astype('float32')
    denom = (nir + red)
    ndvi = np.where(denom == 0, 0.0, (nir - red) / (denom + 1e-10))
    if mask is not None:
        ndvi = np.where(mask, ndvi, np.nan)
    return ndvi


def compute_ndwi(green, nir, mask=None):
    green = green.astype('float32')
    nir = nir.astype('float32')
    denom = (green + nir)
    ndwi = np.where(denom == 0, 0.0, (green - nir) / (denom + 1e-10))
    if mask is not None:
        ndwi = np.where(mask, ndwi, np.nan)
    return ndwi


def save_geotiff(path, array, profile, nodata=np.nan):
    p = profile.copy()
    p.update({
        'count': 1,
        'dtype': 'float32',
        'compress': 'deflate',
        'nodata': nodata
    })
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(array.astype('float32'), 1)


def simple_change_detection(ndvi_a, ndvi_b):
    """Return the difference (B - A) and an absolute change map."""
    diff = ndvi_b - ndvi_a
    abs_change = np.abs(diff)
    return diff, abs_change


if __name__ == '__main__':
    # demo example (assumes arrays loaded elsewhere)
    pass
