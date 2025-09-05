"""
PlayNexus Satellite Toolkit - Sample Data Generator
Creates synthetic satellite imagery for testing and demonstration.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import os


def create_sample_sentinel2_data(output_dir: str = "sample_data", size: tuple = (512, 512)):
    """Create sample Sentinel-2 like data for testing."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create synthetic data
    height, width = size
    
    # B2 (Blue) - 490nm
    blue_band = np.random.normal(0.3, 0.1, (height, width))
    blue_band = np.clip(blue_band, 0, 1)
    
    # B3 (Green) - 560nm  
    green_band = np.random.normal(0.4, 0.15, (height, width))
    green_band = np.clip(green_band, 0, 1)
    
    # B4 (Red) - 665nm
    red_band = np.random.normal(0.35, 0.12, (height, width))
    red_band = np.clip(red_band, 0, 1)
    
    # B8 (NIR) - 842nm
    nir_band = np.random.normal(0.5, 0.2, (height, width))
    nir_band = np.clip(nir_band, 0, 1)
    
    # B11 (SWIR1) - 1610nm
    swir1_band = np.random.normal(0.25, 0.1, (height, width))
    swir1_band = np.clip(swir1_band, 0, 1)
    
    # B12 (SWIR2) - 2190nm
    swir2_band = np.random.normal(0.2, 0.08, (height, width))
    swir2_band = np.clip(swir2_band, 0, 1)
    
    # Add some realistic features
    # Add water bodies
    water_mask = np.random.random((height, width)) < 0.1
    blue_band[water_mask] *= 0.3
    green_band[water_mask] *= 0.4
    red_band[water_mask] *= 0.2
    nir_band[water_mask] *= 0.1
    swir1_band[water_mask] *= 0.05
    swir2_band[water_mask] *= 0.05
    
    # Add vegetation
    veg_mask = np.random.random((height, width)) < 0.3
    nir_band[veg_mask] *= 1.5
    nir_band[veg_mask] = np.clip(nir_band[veg_mask], 0, 1)
    
    # Add urban areas
    urban_mask = np.random.random((height, width)) < 0.15
    red_band[urban_mask] *= 1.3
    red_band[urban_mask] = np.clip(red_band[urban_mask], 0, 1)
    
    # Create GeoTIFF files
    bands = {
        'B02_Blue': blue_band,
        'B03_Green': green_band,
        'B04_Red': red_band,
        'B08_NIR': nir_band,
        'B11_SWIR1': swir1_band,
        'B12_SWIR2': swir2_band
    }
    
    # Define geospatial bounds (example: New York City area)
    bounds = (-74.259, 40.477, -73.700, 40.917)
    transform = from_bounds(*bounds, width, height)
    
    for band_name, band_data in bands.items():
        output_file = output_path / f"sample_sentinel2_{band_name}.tif"
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=band_data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(band_data, 1)
    
    # Create RGB composite
    rgb_data = np.stack([red_band, green_band, blue_band], axis=0)
    rgb_file = output_path / "sample_sentinel2_RGB.tif"
    
    with rasterio.open(
        rgb_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=rgb_data.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(rgb_data)
    
    print(f"✅ Created sample Sentinel-2 data in {output_path}")
    print(f"   - 6 individual bands (B02, B03, B04, B08, B11, B12)")
    print(f"   - RGB composite")
    print(f"   - Size: {size}")
    print(f"   - Location: New York City area")
    
    return str(output_path)


def create_sample_landsat_data(output_dir: str = "sample_data", size: tuple = (512, 512)):
    """Create sample Landsat-like data for testing."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create synthetic data
    height, width = size
    
    # Landsat bands
    blue_band = np.random.normal(0.25, 0.08, (height, width))
    blue_band = np.clip(blue_band, 0, 1)
    
    green_band = np.random.normal(0.35, 0.12, (height, width))
    green_band = np.clip(green_band, 0, 1)
    
    red_band = np.random.normal(0.3, 0.1, (height, width))
    red_band = np.clip(red_band, 0, 1)
    
    nir_band = np.random.normal(0.45, 0.18, (height, width))
    nir_band = np.clip(nir_band, 0, 1)
    
    swir1_band = np.random.normal(0.2, 0.08, (height, width))
    swir1_band = np.clip(swir1_band, 0, 1)
    
    swir2_band = np.random.normal(0.15, 0.06, (height, width))
    swir2_band = np.clip(swir2_band, 0, 1)
    
    # Add realistic features
    # Water bodies
    water_mask = np.random.random((height, width)) < 0.08
    blue_band[water_mask] *= 0.4
    green_band[water_mask] *= 0.3
    red_band[water_mask] *= 0.2
    nir_band[water_mask] *= 0.1
    swir1_band[water_mask] *= 0.05
    swir2_band[water_mask] *= 0.05
    
    # Vegetation
    veg_mask = np.random.random((height, width)) < 0.25
    nir_band[veg_mask] *= 1.4
    nir_band[veg_mask] = np.clip(nir_band[veg_mask], 0, 1)
    
    # Create GeoTIFF files
    bands = {
        'B2_Blue': blue_band,
        'B3_Green': green_band,
        'B4_Red': red_band,
        'B5_NIR': nir_band,
        'B6_SWIR1': swir1_band,
        'B7_SWIR2': swir2_band
    }
    
    # Define geospatial bounds (example: London area)
    bounds = (-0.510, 51.286, 0.334, 51.691)
    transform = from_bounds(*bounds, width, height)
    
    for band_name, band_data in bands.items():
        output_file = output_path / f"sample_landsat_{band_name}.tif"
        
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=band_data.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(band_data, 1)
    
    # Create RGB composite
    rgb_data = np.stack([red_band, green_band, blue_band], axis=0)
    rgb_file = output_path / "sample_landsat_RGB.tif"
    
    with rasterio.open(
        rgb_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=rgb_data.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(rgb_data)
    
    print(f"✅ Created sample Landsat data in {output_path}")
    print(f"   - 6 individual bands (B2, B3, B4, B5, B6, B7)")
    print(f"   - RGB composite")
    print(f"   - Size: {size}")
    print(f"   - Location: London area")
    
    return str(output_path)


def create_sample_anomaly_data(output_dir: str = "sample_data", size: tuple = (512, 512)):
    """Create sample data with anomalies for testing detection algorithms."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create synthetic data
    height, width = size
    
    # Base image
    base_image = np.random.normal(0.4, 0.1, (height, width))
    base_image = np.clip(base_image, 0, 1)
    
    # Add anomalies
    # Statistical anomalies (outliers)
    stat_anomalies = np.random.random((height, width)) < 0.02  # 2% of pixels
    base_image[stat_anomalies] = np.random.uniform(0.8, 1.0, np.sum(stat_anomalies))
    
    # Spatial anomalies (clusters)
    spatial_anomalies = np.random.random((height, width)) < 0.01  # 1% of pixels
    # Expand spatial anomalies to create clusters
    from scipy.ndimage import binary_dilation
    spatial_anomalies = binary_dilation(spatial_anomalies, iterations=2)
    base_image[spatial_anomalies] = np.random.uniform(0.7, 0.9, np.sum(spatial_anomalies))
    
    # Spectral anomalies (different spectral signature)
    spectral_anomalies = np.random.random((height, width)) < 0.015  # 1.5% of pixels
    base_image[spectral_anomalies] = np.random.uniform(0.1, 0.3, np.sum(spectral_anomalies))
    
    # Create GeoTIFF file
    output_file = output_path / "sample_anomaly_data.tif"
    
    # Define geospatial bounds (example: Tokyo area)
    bounds = (139.691, 35.633, 139.767, 35.707)
    transform = from_bounds(*bounds, width, height)
    
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=base_image.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(base_image, 1)
    
    print(f"✅ Created sample anomaly data in {output_path}")
    print(f"   - Single band with embedded anomalies")
    print(f"   - Statistical anomalies: ~2% of pixels")
    print(f"   - Spatial anomalies: ~1% of pixels (clustered)")
    print(f"   - Spectral anomalies: ~1.5% of pixels")
    print(f"   - Size: {size}")
    print(f"   - Location: Tokyo area")
    
    return str(output_path)


def create_all_sample_data(output_dir: str = "sample_data"):
    """Create all sample data types."""
    
    print("🌟 Creating sample satellite data for testing...")
    print("=" * 60)
    
    # Create different types of sample data
    sentinel2_path = create_sample_sentinel2_data(output_dir)
    print()
    
    landsat_path = create_sample_landsat_data(output_dir)
    print()
    
    anomaly_path = create_sample_anomaly_data(output_dir)
    print()
    
    print("=" * 60)
    print("🎉 All sample data created successfully!")
    print(f"📁 Output directory: {output_dir}")
    print()
    print("📊 Data Summary:")
    print("   • Sentinel-2: 6 bands + RGB composite")
    print("   • Landsat: 6 bands + RGB composite") 
    print("   • Anomaly data: Single band with embedded anomalies")
    print()
    print("💡 Use these files to test:")
    print("   • Image enhancement features")
    print("   • Anomaly detection algorithms")
    print("   • NDVI/NDWI calculations")
    print("   • Multi-temporal analysis")
    
    return {
        'sentinel2': sentinel2_path,
        'landsat': landsat_path,
        'anomaly': anomaly_path
    }


if __name__ == "__main__":
    create_all_sample_data()
