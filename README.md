# 🛰️ PlayNexus Satellite Imagery Educational Toolkit

**A comprehensive, production-ready toolkit for satellite imagery analysis, powered by PlayNexus.**

[![PlayNexus](https://img.shields.io/badge/Powered%20by-PlayNexus-blue.svg)](https://playnexus.com)
[![Version](https://img.shields.io/badge/Version-1.1.0-green.svg)](https://github.com/playnexus/satellite-toolkit)
[![License](https://img.shields.io/badge/License-Educational%20Use%20Only-orange.svg)](docs/EULA.md)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/playnexus/satellite-toolkit)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checked: mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

---

## 🏷️ **Branding & Ownership**

- **Brand**: PlayNexus
- **Subsystems**: ClanForge (clan/esports tools), BotForge (AI/Discord bots)
- **Owner**: Nortaq
- **Contact**: playnexushq@gmail.com
- **License**: Educational use only - see [EULA](docs/EULA.md) for details

---

## 🏗️ **New Architecture (v1.1.0)**

### **Modular MVC Architecture**
- **Base Components**: Reusable base classes for views, controllers, and models
- **Separation of Concerns**: Clear separation between UI, business logic, and data
- **Event-Driven Design**: Decoupled components communicating through events
- **Type Hints**: Full type annotations for better code quality and IDE support

### **Enhanced UI Framework**
- **Modern Widgets**: Custom widgets with consistent styling and behavior
- **Animation System**: Smooth animations with configurable easing functions
- **Responsive Layout**: Adapts to different screen sizes and DPI settings
- **Theming Support**: Light and dark theme support with easy customization

### **Developer Experience**
- **Code Quality**: Pre-commit hooks, linting, and type checking
- **Testing**: Comprehensive test suite with pytest
- **Documentation**: API docs, architecture guides, and examples
- **Development Tools**: VS Code configurations and debugging profiles

---

## 🌟 **What's New in v1.0.0**

### ✨ **Major Features**
- **Complete Satellite Imagery Toolkit**: Full-featured educational toolkit for satellite data analysis
- **Multi-Sensor Support**: Sentinel-2 and Landsat data processing capabilities  
- **Advanced Image Enhancement**: Multiple enhancement techniques with configurable parameters
- **Comprehensive Anomaly Detection**: Statistical, spatial, and machine learning-based detection
- **Professional GUI**: Modern desktop interface with tabbed workflow
- **Command-Line Interface**: Full CLI for batch processing and automation

### 🔧 **Core Components**
- **Image Enhancement Module**: Noise reduction, contrast stretching, histogram equalization
- **Anomaly Detection Engine**: Multiple detection algorithms with configurable thresholds
- **Data Download System**: Automated satellite data retrieval from Microsoft Planetary Computer
- **Progress Tracking**: Real-time progress indicators with ETA calculations
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Security Validation**: Input validation and file security checks

---

## 🚀 **Quick Start**

### **System Requirements**
- **Python**: 3.8 or higher (3.10+ recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for installation
- **OS**: Windows 10+, macOS 11+, Ubuntu 20.04+
- **Dependencies**: Git, make (for development)

### **Installation**

#### **For End Users**
1. Download the latest release from the [releases page](https://github.com/playnexus/satellite-toolkit/releases)
2. Extract the package
3. Run `install.bat` (Windows) or `./install.sh` (macOS/Linux)
4. Launch using `run_toolkit.bat` or `./run_toolkit.sh`

#### **For Developers**
```bash
# Clone the repository
git clone https://github.com/playnexus/satellite-toolkit.git
cd satellite-toolkit

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in development mode
pip install -e .

# Run the application
python -m playnexus_satellite_toolkit
```

### **Running Tests**
```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=playnexus_satellite_toolkit

# Run type checking
mypy .

# Run linting
flake8 .
```

### **Quick Start**
1. **Launch GUI**: Start the application with `--gui` flag
2. **Select Input**: Choose a GeoTIFF file for processing
3. **Configure Options**: Select enhancement methods and parameters
4. **Process**: Click "Start Enhancement" to begin processing
5. **View Results**: Check the output directory for processed files

## 🌟 **Core Features**

- **Multi-sensor support**: Works with both Sentinel-2 L2A and Landsat Collection 2 L2 data
- **Automated data acquisition**: Downloads imagery via Microsoft Planetary Computer STAC API
- **Cloud masking**: Automatic cloud detection and masking using QA/SCL bands
- **Vegetation indices**: Computes NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index)
- **Change detection**: Analyzes vegetation changes between two dates
- **Interactive visualization**: Web-based Leaflet viewer for exploring results
- **Google Earth Engine alternative**: JavaScript script for cloud-based processing

---

## 🚀 **Enhanced Features (v1.0.0)**

### **Advanced Image Enhancement**
- **Noise Reduction**: Gaussian, median, Wiener, and bilateral filtering
- **Contrast Enhancement**: Adaptive histogram equalization and contrast stretching
- **Edge Enhancement**: Sobel, Canny, and Laplacian edge detection
- **Multi-Scale Processing**: Pyramid-based enhancement for different detail levels
- **Anomaly Highlighting**: Specialized enhancement for anomaly detection

### **Comprehensive Anomaly Detection**
- **Statistical Methods**: Z-score, IQR, and percentile-based detection
- **Spatial Analysis**: Neighborhood-based anomaly identification
- **Machine Learning**: Isolation Forest for spectral anomaly detection
- **Specialized Detectors**: NDVI and water-specific anomaly detection
- **Multi-Method Analysis**: Combines enhancement and detection for thorough investigation

### **Professional User Experience**
- **Modern GUI**: Tabbed interface with progress tracking and configuration management
- **Progress Monitoring**: Real-time progress bars with ETA calculations
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Security Validation**: Input validation and file security checks
- **Configuration Management**: Centralized settings with platform-specific defaults

## 🛠️ **Development Setup**

### **Prerequisites**
- Python 3.8+ (3.10+ recommended)
- Git
- make (optional, for development commands)
- Virtual environment (recommended)

### **Project Structure**
```
playnexus_satellite_toolkit/
├── gui/                     # GUI components
│   ├── components/          # Reusable UI components
│   │   ├── controllers/     # Business logic
│   │   ├── models/          # Data models
│   │   ├── views/           # UI views
│   │   └── widgets/         # Custom widgets
│   └── utils/               # UI utilities
│       ├── __init__.py
│       ├── animation_utils.py  # Animation system
│       └── ui_utils.py         # Common UI helpers
├── scripts/                 # Core functionality
├── tests/                   # Test suite
└── demo_new_architecture.py # Demo of new architecture
```

### **Code Style**
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Type hints for all function signatures
- 100 character line length
- Use Black for code formatting

### **Development Workflow**
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest`
4. Check types: `mypy .`
5. Format code: `black .`
6. Submit a pull request
- Basic understanding of remote sensing concepts
- Internet connection for downloading satellite data

## 🚀 Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd satellite-imagery-toolkit
   ```

2. **Test the enhanced functionality**
   ```bash
   # Run the demo to verify everything works
   python demo_enhanced_analysis.py
   
   # This will test all enhancement and anomaly detection capabilities
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ **User Interfaces**

### **Graphical User Interface (GUI)**
The toolkit includes a professional desktop application with:
- **Tabbed Interface**: Welcome, Image Enhancement, Anomaly Detection, Comprehensive Analysis, Download Data
- **Progress Tracking**: Real-time progress bars with ETA calculations
- **Configuration Management**: Centralized settings with validation
- **File Browsing**: Integrated file selection and output directory management
- **Help System**: Built-in documentation and support information

### **Command-Line Interface (CLI)**
Full command-line support for automation and batch processing:
- **Image Enhancement**: `python playnexus_satellite_toolkit.py enhance input.tif output_dir`
- **Anomaly Detection**: `python playnexus_satellite_toolkit.py detect input.tif output_dir`
- **Comprehensive Analysis**: `python playnexus_satellite_toolkit.py analyze input.tif output_dir`
- **Data Download**: `python playnexus_satellite_toolkit.py download --bbox -122.6,37.6,-122.3,37.9 --date 2023-07-01`

---

## 📖 **Quick Start Guide**

### **1. GUI Mode (Recommended for Beginners)**
```bash
# Start the graphical interface
python playnexus_satellite_toolkit.py --gui
```

### **2. Command-Line Mode (Advanced Users)**
```bash
# Show all available commands
python playnexus_satellite_toolkit.py --help

# Check dependencies
python playnexus_satellite_toolkit.py --check-deps

# Show version information
python playnexus_satellite_toolkit.py --version
```

### **3. Basic Usage - Single Date Analysis**

```bash
python scripts/run_workflow.py \
  --bbox "-122.6,37.6,-122.3,37.9" \
  --date1 "2023-07-01" \
  --outdir "my_analysis"
```

This will:
- Search for the best Sentinel-2 or Landsat scene covering San Francisco Bay Area
- Download the necessary spectral bands
- Apply cloud masking and atmospheric corrections
- Compute NDVI and NDWI
- Save results as GeoTIFF files

### 2. Change Detection Analysis

```bash
python scripts/run_workflow.py \
  --bbox "-122.6,37.6,-122.3,37.9" \
  --date1 "2023-07-01" \
  --date2 "2023-07-25" \
  --outdir "change_analysis"
```

This will perform the same processing for two dates and additionally:
- Compute NDVI differences between the two dates
- Generate change detection maps
- Save change analysis results

### 3. Custom Sensor Priority

```bash
python scripts/run_workflow.py \
  --bbox "-122.6,37.6,-122.3,37.9" \
  --date1 "2023-07-01" \
  --sensor-priority "landsat-c2-l2" "sentinel-2-l2a" \
  --outdir "landsat_priority"
```

This prioritizes Landsat data over Sentinel-2.

### 4. Enhanced Image Analysis and Anomaly Detection

```bash
# Run comprehensive analysis with image enhancement and anomaly detection
python scripts/comprehensive_anomaly_analysis.py \
  --bbox "-122.6,37.6,-122.3,37.9" \
  --date1 "2023-07-01" \
  --outdir "enhanced_analysis" \
  --enhancement-methods noise_reduction contrast_stretching adaptive_histogram \
  --detection-methods statistical spatial spectral
```

This performs:
- Advanced image enhancement (noise reduction, contrast stretching, histogram equalization)
- Multi-method anomaly detection (statistical, spatial, machine learning)
- Comprehensive comparison and visualization
- Detailed analysis reports

## 🔧 **New Modules & Components**

### **Error Handling & Validation** (`scripts/error_handling.py`)
- **PlayNexusError**: Base exception class for the toolkit
- **InputValidator**: Validates GeoTIFF paths, bounding boxes, dates, and arrays
- **PlayNexusLogger**: Centralized logging with file and console output
- **ErrorBoundary**: Context manager for graceful error handling

### **Progress Tracking** (`scripts/progress_tracker.py`)
- **ProgressTracker**: Multi-step progress tracking with weights
- **ConsoleProgressBar**: Visual progress bar for command-line use
- **FileProgressLogger**: Persistent progress logging to files
- **Progress Decorators**: Easy progress tracking for functions

### **Configuration Management** (`scripts/config.py`)
- **PlayNexusConfig**: Centralized configuration with platform-specific defaults
- **ConfigManager**: Configuration loading, validation, and persistence
- **ProcessingConfig**: Image processing and enhancement settings
- **SecurityConfig**: Security and privacy settings

### **Security & Validation** (`scripts/security.py`)
- **SecurityValidator**: File path, size, and type validation
- **SafeFileHandler**: Secure file operations with validation
- **DataSanitizer**: Input sanitization and array validation
- **Security Decorators**: Automatic security validation for functions

---

## 🔧 **Individual Script Usage**

### **Downloader Script**

```bash
python scripts/downloader.py \
  --collection "sentinel-2-l2a" \
  --bbox "-122.6,37.6,-122.3,37.9" \
  --start "2023-07-01" \
  --end "2023-07-01" \
  --out "downloads"
```

### Preprocessing Script

```python
from scripts.preprocess import read_band, apply_scale_and_nodata, mask_clouds_sentinel

# Load a band
arr, profile = read_band("path/to/band.tif")

# Apply scaling
scaled = apply_scale_and_nodata(arr, profile, 'sentinel')

# Mask clouds
mask = mask_clouds_sentinel(scl_array)
```

### Indices Script

```python
from scripts.indices import compute_ndvi, compute_ndwi, save_geotiff

# Compute NDVI
ndvi = compute_ndvi(nir_band, red_band, mask=cloud_mask)

# Save as GeoTIFF
save_geotiff("ndvi.tif", ndvi, profile)
```

### Enhanced Image Analysis Scripts

#### Image Enhancement
```python
from scripts.image_enhancement import enhance_satellite_image

# Apply multiple enhancement methods
enhanced = enhance_satellite_image(
    geotiff_path='path/to/image.tif',
    output_path='path/to/enhanced.tif',
    enhancement_methods=['noise_reduction', 'contrast_stretching', 'adaptive_histogram']
)
```

#### Anomaly Detection
```python
from scripts.anomaly_detection import detect_anomalies_in_geotiff

# Detect anomalies using multiple methods
anomalies, info = detect_anomalies_in_geotiff(
    geotiff_path='path/to/image.tif',
    output_dir='path/to/results',
    detection_methods=['statistical', 'spatial', 'spectral']
)
```

#### Comprehensive Analysis
```python
from scripts.comprehensive_anomaly_analysis import ComprehensiveAnomalyAnalyzer

# Run complete analysis pipeline
analyzer = ComprehensiveAnomalyAnalyzer()
results = analyzer.analyze_satellite_image(
    input_path='path/to/image.tif',
    output_dir='path/to/results',
    enhancement_methods=['noise_reduction', 'adaptive_histogram'],
    detection_methods=['statistical', 'spatial']
)
```

## 🌐 Web Viewer

The toolkit includes an interactive web viewer for exploring processed results:

1. **Open the viewer**: Navigate to `viewer/index.html` in your web browser
2. **Load GeoTIFF files**: Use the file inputs to load your processed imagery
3. **Layer controls**: Toggle layers on/off and adjust opacity
4. **Interactive map**: Pan, zoom, and explore your satellite data

### Supported File Types
- **True Color**: 3-band RGB GeoTIFFs
- **NDVI**: Single-band GeoTIFFs with vegetation index values
- **NDWI**: Single-band GeoTIFFs with water index values

## ☁️ Google Earth Engine Alternative

For users who prefer cloud-based processing, the `gee/gee_ndvi_compare.js` script provides:

- **No local installation required**
- **Access to Google's computing infrastructure**
- **Real-time visualization**
- **Export capabilities**

### Usage
1. Go to [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Copy and paste the contents of `gee/gee_ndvi_compare.js`
3. Modify the region of interest and dates as needed
4. Click "Run" to execute the analysis

## 📚 Educational Concepts

### What is NDVI?
**Normalized Difference Vegetation Index** measures vegetation health and density:
- **High values (0.6-1.0)**: Dense, healthy vegetation
- **Medium values (0.2-0.6)**: Sparse vegetation or stressed plants
- **Low values (-1.0-0.2)**: No vegetation, water, or urban areas

**Formula**: NDVI = (NIR - Red) / (NIR + Red)

### What is NDWI?
**Normalized Difference Water Index** identifies water bodies and moisture:
- **High values (0.3-1.0)**: Open water
- **Medium values (0.0-0.3)**: Wet soil or vegetation
- **Low values (-1.0-0.0)**: Dry soil or dense vegetation

**Formula**: NDWI = (Green - NIR) / (Green + NIR)

### Cloud Masking
The toolkit automatically detects and masks clouds using:
- **Sentinel-2**: SCL (Scene Classification Layer) with 12 classification codes
- **Landsat**: QA_PIXEL band with cloud confidence indicators

## 🗺️ Understanding Bounding Boxes

Bounding boxes are specified as: `minx,miny,maxx,maxy`

**Example**: `-122.6,37.6,-122.3,37.9`
- **minx**: -122.6 (western boundary)
- **miny**: 37.6 (southern boundary)  
- **maxx**: -122.3 (eastern boundary)
- **maxy**: 37.9 (northern boundary)

**Coordinate System**: WGS84 (latitude/longitude)

## 📊 Output Files

The toolkit generates several output files:

### Per-Date Outputs
- `NDVI.tif`: Vegetation index values
- `NDWI.tif`: Water index values  
- `TRUE_COLOR.tif`: RGB composite for visual interpretation

### Change Detection Outputs
- `NDVI_DIFFERENCE.tif`: NDVI change between dates (Date2 - Date1)
- `NDVI_ABS_CHANGE.tif`: Absolute magnitude of NDVI changes

## 🔍 Troubleshooting

### Common Issues

1. **"No items found" error**
   - Check your bounding box coordinates
   - Verify the date format (YYYY-MM-DD)
   - Try increasing the `--max-cloud` parameter

2. **Download failures**
   - Check your internet connection
   - Verify the STAC API is accessible
   - Some regions may have limited data coverage

3. **Memory errors**
   - Reduce the bounding box size
   - Process smaller areas at a time
   - Close other applications to free memory

### Performance Tips

- **Smaller areas**: Process smaller bounding boxes for faster results
- **Recent dates**: Newer imagery is more likely to be available
- **Cloud thresholds**: Lower cloud cover requirements may reduce available scenes

## 📖 Learning Resources

### Remote Sensing Fundamentals
- [NASA Earth Observatory](https://earthobservatory.nasa.gov/)
- [USGS Remote Sensing](https://www.usgs.gov/programs/national-geospatial-program/remote-sensing)
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/)

### Python Remote Sensing
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [GeoPandas Tutorials](https://geopandas.org/getting_started.html)
- [STAC Specification](https://stacspec.org/)

### Google Earth Engine
- [GEE Documentation](https://developers.google.com/earth-engine)
- [GEE Code Editor](https://code.earthengine.google.com/)
- [GEE Community](https://earthengine.google.com/community/)

## 🧪 **Testing & Verification**

### **Installation Verification**
Run the comprehensive verification script to test all components:
```bash
python verify_installation.py
```

This script tests:
- ✅ Module imports and dependencies
- ✅ PlayNexus-specific modules
- ✅ GUI availability and functionality
- ✅ Configuration system
- ✅ Error handling and logging
- ✅ Progress tracking
- ✅ Security validation
- ✅ Image processing capabilities
- ✅ Anomaly detection
- ✅ Main application entry point

### **Smoke Tests**
The verification script includes smoke tests to ensure basic functionality:
- Configuration loading and validation
- Error handling and logging
- Progress tracking initialization
- Security validation
- Module initialization

### **Quality Assurance**
- **Test Coverage**: Comprehensive testing of all features
- **Error Handling**: Graceful degradation and user-friendly messages
- **Security**: Input validation and file security checks
- **Performance**: Memory management and processing optimization
- **Documentation**: Complete API reference and usage examples

---

## 🤝 **Contributing**

This toolkit is designed for educational use. Contributions that improve clarity, add examples, or fix bugs are welcome!

## 🚀 **Production Release & Deployment**

### **v1.0.0 Production Release**
- **Release Date**: January 21, 2025
- **Build Date**: January 21, 2025
- **Platform Support**: Windows, macOS, Linux
- **Status**: Production Ready ✅

### **Build & Packaging**
```bash
# Create distributable package
python build.py

# This generates:
# - playnexus_satellite_toolkit_v1.0.0_[platform].zip
# - Installation scripts (install.bat/install.sh)
# - Launcher scripts (run_toolkit.bat/run_toolkit.sh)
# - Comprehensive documentation
# - Checksums for verification
```

### **Deployment Artifacts**
- **Main Application**: `playnexus_satellite_toolkit.py`
- **Core Modules**: All scripts in `scripts/` directory
- **GUI Interface**: Professional desktop application
- **Documentation**: Complete user guides and API reference
- **Legal Documents**: EULA and Privacy Policy
- **Configuration**: Platform-specific settings and defaults

### **Installation & Distribution**
1. **Extract Package**: Unzip the downloaded package
2. **Run Installer**: Execute platform-specific installer
3. **Launch Application**: Use launcher script or main application
4. **Verify Installation**: Run verification script to confirm functionality

---

## 📄 **License & Legal**

### **Educational Use License**
This project is provided for educational purposes under the PlayNexus brand. Please respect the terms of service for the data sources used:
- **Sentinel-2**: [Copernicus Open Data License](https://www.copernicus.eu/en/access-data/copyright-and-licences)
- **Landsat**: [USGS Public Domain](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)

### **PlayNexus Branding**
- **Brand**: PlayNexus
- **Subsystems**: ClanForge (clan/esports tools), BotForge (AI/Discord bots)
- **Owner**: Nortaq
- **Contact**: playnexushq@gmail.com
- **EULA**: See [docs/EULA.md](docs/EULA.md) for complete terms
- **Privacy Policy**: See [docs/PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) for data handling

## 🙏 **Acknowledgments**

### **Development Team**
- **Principal Engineer**: AI Assistant (Claude) - Comprehensive system design and implementation
- **QA Lead**: Comprehensive testing and validation of all components
- **Release Manager**: Production deployment and distribution management

### **Data Providers**
- **Microsoft Planetary Computer** for providing STAC API access
- **Copernicus Programme** for Sentinel-2 data
- **USGS** for Landsat data

### **Open Source Community**
- **Scientific Python**: NumPy, SciPy, scikit-image, scikit-learn
- **Geospatial Tools**: Rasterio, rioxarray, shapely, geopandas
- **GUI Framework**: Tkinter and modern theming
- **All contributors** to the Python libraries used

### **Educational Partners**
- **Academic Institutions**: Feedback and testing support
- **Research Organizations**: Use case validation and requirements
- **Student Users**: Beta testing and feature feedback

---

## 🎯 **Mission Statement**

**PlayNexus Satellite Toolkit** is dedicated to making satellite imagery analysis accessible to students, researchers, and educators worldwide. By providing professional-grade tools with an educational focus, we empower the next generation of remote sensing professionals.

---

**Happy satellite imagery analysis! 🛰️🌍**

"Powered by PlayNexus — Subsystems: ClanForge, BotForge. Owned by Nortaq. Contact: playnexushq@gmail.com"
