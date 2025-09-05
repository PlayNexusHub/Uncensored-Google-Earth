"""
Advanced Data Sources Module for PlayNexus Satellite Toolkit
Provides access to multiple satellite platforms and enhanced data acquisition capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import requests
from pystac_client import Client
from pystac import Item, Collection
import planetary_computer as pc
from shapely.geometry import box, Polygon
import geopandas as gpd

from .error_handling import PlayNexusLogger, ValidationError, DataError
from .config import ConfigManager
from .security import SecurityValidator

logger = PlayNexusLogger(__name__)

@dataclass
class SatellitePlatform:
    """Represents a satellite platform with its capabilities."""
    name: str
    provider: str
    spatial_resolution: float  # meters
    temporal_resolution: int   # days
    spectral_bands: List[str]
    data_access: str  # 'free', 'subscription', 'commercial'
    max_coverage: float  # km² per scene
    data_quality: str  # 'research', 'operational', 'commercial'

@dataclass
class DataSourceConfig:
    """Configuration for data source access."""
    api_keys: Dict[str, str]
    rate_limits: Dict[str, int]
    max_concurrent_downloads: int
    cache_directory: Path
    enable_parallel: bool
    timeout_seconds: int

class AdvancedDataSources:
    """Enhanced data source management with multiple platforms and real-time access."""
    
    # Supported satellite platforms
    SUPPORTED_PLATFORMS = {
        'sentinel-2': SatellitePlatform(
            name='Sentinel-2',
            provider='ESA/Copernicus',
            spatial_resolution=10.0,
            temporal_resolution=5,
            spectral_bands=['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            data_access='free',
            max_coverage=10000.0,
            data_quality='operational'
        ),
        'landsat-9': SatellitePlatform(
            name='Landsat 9',
            provider='NASA/USGS',
            spatial_resolution=15.0,
            temporal_resolution=16,
            spectral_bands=['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
            data_access='free',
            max_coverage=185000.0,
            data_quality='operational'
        ),
        'planet-scope': SatellitePlatform(
            name='PlanetScope',
            provider='Planet Labs',
            spatial_resolution=3.0,
            temporal_resolution=1,
            spectral_bands=['B', 'G', 'R', 'NIR'],
            data_access='subscription',
            max_coverage=25000.0,
            data_quality='commercial'
        ),
        'modis': SatellitePlatform(
            name='MODIS',
            provider='NASA',
            spatial_resolution=250.0,
            temporal_resolution=1,
            spectral_bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
            data_access='free',
            max_coverage=1000000.0,
            data_quality='research'
        ),
        'viirs': SatellitePlatform(
            name='VIIRS',
            provider='NASA/NOAA',
            spatial_resolution=375.0,
            temporal_resolution=1,
            spectral_bands=['I1', 'I2', 'I3', 'I4', 'I5'],
            data_access='free',
            max_coverage=2000000.0,
            data_quality='operational'
        )
    }
    
    def __init__(self, config: Optional[DataSourceConfig] = None):
        """Initialize the advanced data sources manager."""
        self.config = config or self._get_default_config()
        self.security_validator = SecurityValidator()
        self._setup_logging()
        self._validate_config()
        
    def _get_default_config(self) -> DataSourceConfig:
        """Get default configuration for data sources."""
        return DataSourceConfig(
            api_keys={},
            rate_limits={'default': 100, 'planetary_computer': 1000},
            max_concurrent_downloads=5,
            cache_directory=Path.home() / '.playnexus' / 'cache',
            enable_parallel=True,
            timeout_seconds=300
        )
    
    def _setup_logging(self):
        """Setup logging for the data sources module."""
        self.logger = PlayNexusLogger(__name__)
        
    def _validate_config(self):
        """Validate the configuration."""
        if not self.config.cache_directory.exists():
            self.config.cache_directory.mkdir(parents=True, exist_ok=True)
            
        # Validate API keys if provided
        for platform, key in self.config.api_keys.items():
            if not self.security_validator.is_valid_api_key(key):
                raise ValidationError(f"Invalid API key format for {platform}")
    
    def get_available_platforms(self) -> Dict[str, SatellitePlatform]:
        """Get all available satellite platforms."""
        return self.SUPPORTED_PLATFORMS.copy()
    
    def get_platform_info(self, platform_name: str) -> Optional[SatellitePlatform]:
        """Get information about a specific platform."""
        return self.SUPPORTED_PLATFORMS.get(platform_name.lower())
    
    async def search_multi_platform(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        platforms: List[str] = None,
        max_cloud_cover: float = 30.0,
        min_resolution: float = None
    ) -> Dict[str, List[Item]]:
        """Search for data across multiple platforms simultaneously."""
        if platforms is None:
            platforms = list(self.SUPPORTED_PLATFORMS.keys())
            
        # Validate inputs
        self._validate_search_params(bbox, start_date, end_date, platforms)
        
        # Filter platforms by resolution if specified
        if min_resolution:
            platforms = [
                p for p in platforms 
                if self.SUPPORTED_PLATFORMS[p].spatial_resolution <= min_resolution
            ]
        
        # Search each platform
        tasks = []
        for platform in platforms:
            task = self._search_platform(
                platform, bbox, start_date, end_date, max_cloud_cover
            )
            tasks.append(task)
        
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        platform_results = {}
        for platform, result in zip(platforms, results):
            if isinstance(result, Exception):
                self.logger.error(f"Error searching {platform}: {result}")
                platform_results[platform] = []
            else:
                platform_results[platform] = result
                
        return platform_results
    
    async def _search_platform(
        self,
        platform: str,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float
    ) -> List[Item]:
        """Search for data from a specific platform."""
        try:
            if platform == 'sentinel-2':
                return await self._search_sentinel2(bbox, start_date, end_date, max_cloud_cover)
            elif platform == 'landsat-9':
                return await self._search_landsat9(bbox, start_date, end_date, max_cloud_cover)
            elif platform == 'modis':
                return await self._search_modis(bbox, start_date, end_date)
            elif platform == 'viirs':
                return await self._search_viirs(bbox, start_date, end_date)
            else:
                self.logger.warning(f"Platform {platform} not yet implemented")
                return []
        except Exception as e:
            self.logger.error(f"Error searching {platform}: {e}")
            return []
    
    async def _search_sentinel2(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float
    ) -> List[Item]:
        """Search Sentinel-2 data using Planetary Computer."""
        try:
            # Use Planetary Computer STAC API
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace
            )
            
            # Create search parameters
            search_params = {
                "collections": ["sentinel-2-l2a"],
                "bbox": bbox,
                "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
                "query": {
                    "eo:cloud_cover": {"lte": max_cloud_cover}
                }
            }
            
            # Perform search
            search = catalog.search(**search_params)
            items = list(search.items())
            
            self.logger.info(f"Found {len(items)} Sentinel-2 items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error searching Sentinel-2: {e}")
            return []
    
    async def _search_landsat9(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float
    ) -> List[Item]:
        """Search Landsat 9 data using Planetary Computer."""
        try:
            catalog = Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=pc.sign_inplace
            )
            
            search_params = {
                "collections": ["landsat-c2-l2"],
                "bbox": bbox,
                "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}",
                "query": {
                    "landsat:cloud_cover_land": {"lte": max_cloud_cover}
                }
            }
            
            search = catalog.search(**search_params)
            items = list(search.items())
            
            self.logger.info(f"Found {len(items)} Landsat 9 items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error searching Landsat 9: {e}")
            return []
    
    async def _search_modis(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> List[Item]:
        """Search MODIS data using NASA's CMR API."""
        try:
            # NASA CMR API endpoint
            cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
            
            # Search parameters for MODIS Terra/Aqua
            params = {
                "collection_concept_id": "C1940468264-NOAA_NCEI",  # MODIS Terra
                "bounding_box": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",
                "temporal": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z,{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
                "page_size": 100
            }
            
            response = requests.get(cmr_url, params=params, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            
            # Parse results (simplified - would need full UMM parsing in production)
            data = response.json()
            items = []
            
            # Convert CMR results to STAC-like items
            for feature in data.get('items', []):
                # Create minimal STAC item structure
                item = {
                    'id': feature.get('id', ''),
                    'properties': {
                        'datetime': feature.get('meta', {}).get('native-id', ''),
                        'platform': 'MODIS',
                        'instrument': 'MODIS'
                    },
                    'geometry': None,  # Would need to parse from CMR
                    'links': []
                }
                items.append(item)
            
            self.logger.info(f"Found {len(items)} MODIS items")
            return items
            
        except Exception as e:
            self.logger.error(f"Error searching MODIS: {e}")
            return []
    
    async def _search_viirs(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime
    ) -> List[Item]:
        """Search VIIRS data using NOAA's CLASS API."""
        try:
            # NOAA CLASS API endpoint (simplified)
            class_url = "https://www.avl.class.noaa.gov/saa/products/search"
            
            # Search parameters for VIIRS
            params = {
                "dataType": "VIIRS",
                "startDate": start_date.strftime('%Y%m%d'),
                "endDate": end_date.strftime('%Y%m%d'),
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            }
            
            response = requests.get(class_url, params=params, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            
            # Parse results (simplified - would need full CLASS parsing in production)
            items = []
            
            # Convert CLASS results to STAC-like items
            # This is a placeholder - actual implementation would parse CLASS XML/JSON
            self.logger.info("VIIRS search completed (placeholder implementation)")
            return items
            
        except Exception as e:
            self.logger.error(f"Error searching VIIRS: {e}")
            return []
    
    def _validate_search_params(
        self,
        bbox: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        platforms: List[str]
    ):
        """Validate search parameters."""
        # Validate bounding box
        if len(bbox) != 4:
            raise ValidationError("Bounding box must have 4 coordinates (minx, miny, maxx, maxy)")
        
        minx, miny, maxx, maxy = bbox
        if minx >= maxx or miny >= maxy:
            raise ValidationError("Invalid bounding box coordinates")
        
        # Validate dates
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Validate platforms
        invalid_platforms = [p for p in platforms if p not in self.SUPPORTED_PLATFORMS]
        if invalid_platforms:
            raise ValidationError(f"Unsupported platforms: {invalid_platforms}")
    
    def get_data_quality_metrics(self, items: List[Item]) -> Dict[str, float]:
        """Calculate data quality metrics for a list of items."""
        if not items:
            return {}
        
        metrics = {
            'total_items': len(items),
            'avg_cloud_cover': 0.0,
            'coverage_area': 0.0,
            'temporal_coverage': 0.0
        }
        
        cloud_covers = []
        areas = []
        dates = []
        
        for item in items:
            # Extract cloud cover
            cloud_cover = item.properties.get('eo:cloud_cover', 0)
            cloud_covers.append(cloud_cover)
            
            # Extract area (simplified)
            if item.geometry:
                # Calculate approximate area from geometry
                try:
                    poly = Polygon(item.geometry['coordinates'][0])
                    area = poly.area
                    areas.append(area)
                except:
                    pass
            
            # Extract date
            if 'datetime' in item.properties:
                try:
                    date = datetime.fromisoformat(item.properties['datetime'].replace('Z', '+00:00'))
                    dates.append(date)
                except:
                    pass
        
        if cloud_covers:
            metrics['avg_cloud_cover'] = np.mean(cloud_covers)
        if areas:
            metrics['coverage_area'] = np.sum(areas)
        if dates:
            date_range = max(dates) - min(dates)
            metrics['temporal_coverage'] = date_range.days
        
        return metrics
    
    def export_search_results(
        self,
        results: Dict[str, List[Item]],
        output_path: Path,
        format: str = 'json'
    ) -> Path:
        """Export search results to various formats."""
        if format.lower() == 'json':
            return self._export_json(results, output_path)
        elif format.lower() == 'csv':
            return self._export_csv(results, output_path)
        elif format.lower() == 'geojson':
            return self._export_geojson(results, output_path)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def _export_json(self, results: Dict[str, List[Item]], output_path: Path) -> Path:
        """Export results as JSON."""
        output_file = output_path.with_suffix('.json')
        
        # Convert STAC items to serializable format
        export_data = {}
        for platform, items in results.items():
            export_data[platform] = []
            for item in items:
                export_data[platform].append({
                    'id': item.id,
                    'properties': item.properties,
                    'geometry': item.geometry,
                    'links': [link.href for link in item.links]
                })
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return output_file
    
    def _export_csv(self, results: Dict[str, List[Item]], output_path: Path) -> Path:
        """Export results as CSV."""
        output_file = output_path.with_suffix('.csv')
        
        # Flatten results for CSV export
        rows = []
        for platform, items in results.items():
            for item in items:
                row = {
                    'platform': platform,
                    'id': item.id,
                    'datetime': item.properties.get('datetime', ''),
                    'cloud_cover': item.properties.get('eo:cloud_cover', ''),
                    'geometry_type': item.geometry.get('type', '') if item.geometry else ''
                }
                rows.append(row)
        
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
        
        return output_file
    
    def _export_geojson(self, results: Dict[str, List[Item]], output_path: Path) -> Path:
        """Export results as GeoJSON."""
        output_file = output_path.with_suffix('.geojson')
        
        # Convert to GeoJSON format
        features = []
        for platform, items in results.items():
            for item in items:
                if item.geometry:
                    feature = {
                        'type': 'Feature',
                        'geometry': item.geometry,
                        'properties': {
                            'platform': platform,
                            'id': item.id,
                            **item.properties
                        }
                    }
                    features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2, default=str)
        
        return output_file

# Convenience function for easy usage
async def search_satellite_data(
    bbox: Tuple[float, float, float, float],
    start_date: datetime,
    end_date: datetime,
    platforms: List[str] = None,
    max_cloud_cover: float = 30.0,
    min_resolution: float = None
) -> Dict[str, List[Item]]:
    """Convenience function to search satellite data across multiple platforms."""
    data_sources = AdvancedDataSources()
    return await data_sources.search_multi_platform(
        bbox, start_date, end_date, platforms, max_cloud_cover, min_resolution
    )
