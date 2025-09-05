#!/usr/bin/env python3
"""
Example usage of the Satellite Imagery Educational Toolkit.
This script demonstrates how to use the toolkit programmatically.
"""

import sys
from pathlib import Path

# Add the scripts directory to the Python path
sys.path.append(str(Path(__file__).parent / "scripts"))

def example_single_date_analysis():
    """Example of single date analysis."""
    print("🛰️ Example: Single Date Analysis")
    print("=" * 40)
    
    try:
        from run_workflow import orchestrate
        
        # Define parameters for San Francisco Bay Area
        bbox = (-122.6, 37.6, -122.3, 37.9)  # San Francisco Bay Area
        date1 = "2023-07-01"
        out_root = "example_outputs"
        
        print(f"Processing area: {bbox}")
        print(f"Date: {date1}")
        print(f"Output directory: {out_root}")
        
        # Run the analysis
        results = orchestrate(
            bbox=bbox,
            date1=date1,
            out_root=out_root,
            sensor_priority=['sentinel-2-l2a', 'landsat-c2-l2']
        )
        
        print(f"\n✅ Analysis complete! Results saved to {out_root}")
        for date_key, paths in results.items():
            print(f"  {date_key}:")
            for product, path in paths.items():
                print(f"    {product}: {path}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def example_change_detection():
    """Example of change detection analysis."""
    print("\n🔄 Example: Change Detection Analysis")
    print("=" * 40)
    
    try:
        from run_workflow import orchestrate
        
        # Define parameters for change detection
        bbox = (-122.6, 37.6, -122.3, 37.9)  # San Francisco Bay Area
        date1 = "2023-07-01"
        date2 = "2023-07-25"
        out_root = "example_change_detection"
        
        print(f"Processing area: {bbox}")
        print(f"Date 1: {date1}")
        print(f"Date 2: {date2}")
        print(f"Output directory: {out_root}")
        
        # Run the change detection analysis
        results = orchestrate(
            bbox=bbox,
            date1=date1,
            date2=date2,
            out_root=out_root,
            sensor_priority=['sentinel-2-l2a', 'landsat-c2-l2']
        )
        
        print(f"\n✅ Change detection complete! Results saved to {out_root}")
        for date_key, paths in results.items():
            print(f"  {date_key}:")
            for product, path in paths.items():
                print(f"    {product}: {path}")
        
        # Check for change detection outputs
        change_dir = Path(out_root) / "change_detection"
        if change_dir.exists():
            print(f"\n📊 Change detection outputs:")
            for file in change_dir.glob("*.tif"):
                print(f"  {file.name}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during change detection: {e}")

def example_individual_components():
    """Example of using individual components."""
    print("\n🔧 Example: Using Individual Components")
    print("=" * 40)
    
    try:
        from indices import compute_ndvi, compute_ndwi
        import numpy as np
        
        # Create sample data (simulating satellite bands)
        print("Creating sample satellite data...")
        
        # Simulate a 100x100 pixel area
        height, width = 100, 100
        
        # Simulate NIR band (near-infrared)
        nir = np.random.uniform(0.1, 0.8, (height, width)).astype('float32')
        
        # Simulate Red band
        red = np.random.uniform(0.1, 0.6, (height, width)).astype('float32')
        
        # Simulate Green band
        green = np.random.uniform(0.1, 0.7, (height, width)).astype('float32')
        
        # Create a simple cloud mask (most pixels are clear)
        mask = np.random.choice([True, False], (height, width), p=[0.9, 0.1])
        
        print(f"Sample data created: {height}x{width} pixels")
        print(f"Cloud mask: {np.sum(mask)} clear pixels, {np.sum(~mask)} cloudy pixels")
        
        # Compute indices
        print("\nComputing vegetation indices...")
        ndvi = compute_ndvi(nir, red, mask=mask)
        ndwi = compute_ndwi(green, nir, mask=mask)
        
        print(f"NDVI range: {np.nanmin(ndvi):.3f} to {np.nanmax(ndvi):.3f}")
        print(f"NDWI range: {np.nanmin(ndwi):.3f} to {np.nanmax(ndwi):.3f}")
        
        # Analyze results
        print("\n📊 Analysis Results:")
        print(f"  High vegetation (NDVI > 0.6): {np.sum(ndvi > 0.6)} pixels")
        print(f"  Medium vegetation (0.2 < NDVI ≤ 0.6): {np.sum((ndvi > 0.2) & (ndvi <= 0.6))} pixels")
        print(f"  Low vegetation (NDVI ≤ 0.2): {np.sum(ndvi <= 0.2)} pixels")
        
        print(f"  Water bodies (NDWI > 0.3): {np.sum(ndwi > 0.3)} pixels")
        print(f"  Wet areas (0.0 < NDWI ≤ 0.3): {np.sum((ndwi > 0.0) & (ndwi <= 0.3))} pixels")
        print(f"  Dry areas (NDWI ≤ 0.0): {np.sum(ndwi <= 0.0)} pixels")
        
        print("\n✅ Individual component test completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during component test: {e}")

def main():
    """Run all examples."""
    print("🚀 Satellite Imagery Educational Toolkit - Examples")
    print("=" * 60)
    
    print("This script demonstrates various ways to use the toolkit.")
    print("Note: Some examples require internet connection and may take time to run.")
    print()
    
    # Example 1: Single date analysis
    example_single_date_analysis()
    
    # Example 2: Change detection
    example_change_detection()
    
    # Example 3: Individual components
    example_individual_components()
    
    print("\n" + "=" * 60)
    print("🎉 Examples completed!")
    print("\n📖 For more information:")
    print("  - Read README.md for detailed documentation")
    print("  - Run 'python scripts/run_workflow.py --help' for command-line options")
    print("  - Open viewer/index.html for interactive visualization")
    print("  - Check gee/gee_ndvi_compare.js for Google Earth Engine alternative")

if __name__ == "__main__":
    main()
