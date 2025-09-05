"""
Advanced Processing Pipeline Module for PlayNexus Satellite Toolkit
Provides enhanced preprocessing, multi-temporal analysis, and time series processing capabilities.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import rasterize
import rioxarray as rio
import xarray as xr
from scipy import ndimage, signal
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings
import logging

from .error_handling import PlayNexusLogger, ProcessingError, ValidationError
from .progress_tracker import ProgressTracker, track_progress
from .config import ConfigManager

logger = PlayNexusLogger(__name__)

class AdvancedProcessor:
    """Advanced satellite image processing with multi-temporal and time series capabilities."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the advanced processor."""
        self.config = config or ConfigManager()
        self.logger = PlayNexusLogger(__name__)
        self._setup_processing_config()
    
    def _setup_processing_config(self):
        """Setup processing configuration."""
        self.resampling_methods = {
            'nearest': Resampling.nearest,
            'bilinear': Resampling.bilinear,
            'cubic': Resampling.cubic,
            'cubic_spline': Resampling.cubic_spline,
            'lanczos': Resampling.lanczos,
            'average': Resampling.average,
            'mode': Resampling.mode,
            'gauss': Resampling.gauss,
            'max': Resampling.max,
            'min': Resampling.min,
            'med': Resampling.med,
            'q1': Resampling.q1,
            'q3': Resampling.q3
        }
        
        self.filter_types = {
            'gaussian': ndimage.gaussian_filter,
            'median': ndimage.median_filter,
            'uniform': ndimage.uniform_filter,
            'wiener': signal.wiener,
            'bilateral': self._bilateral_filter
        }
    
    @track_progress("Multi-temporal analysis")
    def multi_temporal_analysis(
        self,
        image_paths: List[Path],
        dates: List[datetime],
        analysis_type: str = 'change_detection',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Perform multi-temporal analysis on a series of images."""
        if len(image_paths) != len(dates):
            raise ValidationError("Number of image paths must match number of dates")
        
        if len(image_paths) < 2:
            raise ValidationError("At least 2 images required for multi-temporal analysis")
        
        # Sort by date
        sorted_data = sorted(zip(dates, image_paths), key=lambda x: x[0])
        dates, image_paths = zip(*sorted_data)
        
        if analysis_type == 'change_detection':
            return self._change_detection_analysis(image_paths, dates, **kwargs)
        elif analysis_type == 'time_series':
            return self._time_series_analysis(image_paths, dates, **kwargs)
        elif analysis_type == 'trend_analysis':
            return self._trend_analysis(image_paths, dates, **kwargs)
        elif analysis_type == 'anomaly_detection':
            return self._temporal_anomaly_detection(image_paths, dates, **kwargs)
        else:
            raise ValidationError(f"Unsupported analysis type: {analysis_type}")
    
    def _change_detection_analysis(
        self,
        image_paths: List[Path],
        dates: List[datetime],
        method: str = 'difference',
        normalize: bool = True,
        threshold: float = None
    ) -> Dict[str, np.ndarray]:
        """Perform change detection analysis between images."""
        self.logger.info(f"Performing change detection analysis with {len(image_paths)} images")
        
        # Load first image to get reference
        with rasterio.open(image_paths[0]) as src:
            reference_profile = src.profile
            reference_shape = src.shape
        
        # Initialize results
        results = {
            'change_maps': [],
            'change_magnitudes': [],
            'change_directions': [],
            'dates': dates,
            'reference_date': dates[0]
        }
        
        # Process each subsequent image
        for i, (date, image_path) in enumerate(zip(dates[1:], image_paths[1:])):
            self.logger.info(f"Processing change from {dates[0]} to {date}")
            
            # Load images
            img1 = self._load_and_preprocess_image(image_paths[0], normalize)
            img2 = self._load_and_preprocess_image(image_path, normalize)
            
            # Ensure same dimensions
            if img1.shape != img2.shape:
                img2 = self._resample_to_match(img2, img1.shape)
            
            # Calculate change
            if method == 'difference':
                change_map = img2 - img1
            elif method == 'ratio':
                change_map = np.divide(img2, img1, out=np.zeros_like(img2), where=img1 != 0)
            elif method == 'normalized_difference':
                change_map = (img2 - img1) / (img2 + img1 + 1e-8)
            else:
                raise ValidationError(f"Unsupported change detection method: {method}")
            
            # Calculate magnitude and direction
            change_magnitude = np.abs(change_map)
            change_direction = np.sign(change_map)
            
            # Apply threshold if specified
            if threshold is not None:
                change_map = np.where(change_magnitude > threshold, change_map, 0)
                change_magnitude = np.where(change_magnitude > threshold, change_magnitude, 0)
            
            results['change_maps'].append(change_map)
            results['change_magnitudes'].append(change_magnitude)
            results['change_directions'].append(change_direction)
        
        return results
    
    def _time_series_analysis(
        self,
        image_paths: List[Path],
        dates: List[datetime],
        method: str = 'pixel_trajectory',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Perform time series analysis on image stack."""
        self.logger.info(f"Performing time series analysis with {len(image_paths)} images")
        
        # Load all images
        images = []
        for image_path in image_paths:
            img = self._load_and_preprocess_image(image_path, normalize=True)
            images.append(img)
        
        # Stack images
        image_stack = np.stack(images, axis=0)
        
        if method == 'pixel_trajectory':
            return self._analyze_pixel_trajectories(image_stack, dates, **kwargs)
        elif method == 'temporal_statistics':
            return self._calculate_temporal_statistics(image_stack, dates, **kwargs)
        elif method == 'seasonal_decomposition':
            return self._seasonal_decomposition(image_stack, dates, **kwargs)
        else:
            raise ValidationError(f"Unsupported time series method: {method}")
    
    def _analyze_pixel_trajectories(
        self,
        image_stack: np.ndarray,
        dates: List[datetime],
        min_valid_pixels: int = 3
    ) -> Dict[str, np.ndarray]:
        """Analyze pixel-level trajectories over time."""
        # Convert dates to numeric for analysis
        date_nums = [(d - dates[0]).days for d in dates]
        
        # Initialize results
        height, width = image_stack.shape[1:]
        results = {
            'slopes': np.full((height, width), np.nan),
            'intercepts': np.full((height, width), np.nan),
            'r_squared': np.full((height, width), np.nan),
            'p_values': np.full((height, width), np.nan),
            'trend_significance': np.full((height, width), False)
        }
        
        # Analyze each pixel
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                
                # Check for valid data
                valid_mask = ~np.isnan(pixel_series) & ~np.isinf(pixel_series)
                if np.sum(valid_mask) < min_valid_pixels:
                    continue
                
                valid_series = pixel_series[valid_mask]
                valid_dates = [date_nums[k] for k, valid in enumerate(valid_mask) if valid]
                
                if len(valid_series) < min_valid_pixels:
                    continue
                
                try:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(valid_dates, valid_series)
                    
                    results['slopes'][i, j] = slope
                    results['intercepts'][i, j] = intercept
                    results['r_squared'][i, j] = r_value ** 2
                    results['p_values'][i, j] = p_value
                    results['trend_significance'][i, j] = p_value < 0.05
                    
                except Exception as e:
                    self.logger.debug(f"Error analyzing pixel ({i}, {j}): {e}")
                    continue
        
        return results
    
    def _calculate_temporal_statistics(
        self,
        image_stack: np.ndarray,
        dates: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Calculate temporal statistics for each pixel."""
        results = {
            'mean': np.nanmean(image_stack, axis=0),
            'std': np.nanstd(image_stack, axis=0),
            'min': np.nanmin(image_stack, axis=0),
            'max': np.nanmax(image_stack, axis=0),
            'median': np.nanmedian(image_stack, axis=0),
            'range': np.nanmax(image_stack, axis=0) - np.nanmin(image_stack, axis=0),
            'coefficient_of_variation': np.nanstd(image_stack, axis=0) / (np.nanmean(image_stack, axis=0) + 1e-8)
        }
        
        return results
    
    def _seasonal_decomposition(
        self,
        image_stack: np.ndarray,
        dates: List[datetime],
        period: int = 365
    ) -> Dict[str, np.ndarray]:
        """Perform seasonal decomposition of time series."""
        # This is a simplified seasonal decomposition
        # In production, you might use more sophisticated methods like STL or X-13ARIMA-SEATS
        
        height, width = image_stack.shape[1:]
        
        # Calculate seasonal component (simplified)
        seasonal = np.zeros_like(image_stack)
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                if not np.all(np.isnan(pixel_series)):
                    # Simple moving average for seasonal pattern
                    seasonal[:, i, j] = self._moving_average(pixel_series, period)
        
        # Calculate trend component
        trend = np.zeros_like(image_stack)
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                if not np.all(np.isnan(pixel_series)):
                    # Linear trend
                    x = np.arange(len(pixel_series))
                    valid_mask = ~np.isnan(pixel_series)
                    if np.sum(valid_mask) > 1:
                        try:
                            slope, intercept = np.polyfit(x[valid_mask], pixel_series[valid_mask], 1)
                            trend[:, i, j] = slope * x + intercept
                        except:
                            pass
        
        # Calculate residual component
        residual = image_stack - seasonal - trend
        
        return {
            'original': image_stack,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _trend_analysis(
        self,
        image_paths: List[Path],
        dates: List[datetime],
        method: str = 'mann_kendall',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Perform trend analysis on time series."""
        if method == 'mann_kendall':
            return self._mann_kendall_trend(image_paths, dates, **kwargs)
        elif method == 'sen_slope':
            return self._sen_slope_trend(image_paths, dates, **kwargs)
        else:
            raise ValidationError(f"Unsupported trend method: {method}")
    
    def _mann_kendall_trend(
        self,
        image_paths: List[Path],
        dates: List[datetime]
    ) -> Dict[str, np.ndarray]:
        """Calculate Mann-Kendall trend test statistics."""
        # Load images
        images = []
        for image_path in image_paths:
            img = self._load_and_preprocess_image(image_path, normalize=True)
            images.append(img)
        
        image_stack = np.stack(images, axis=0)
        height, width = image_stack.shape[1:]
        
        results = {
            'mk_statistic': np.full((height, width), np.nan),
            'mk_p_value': np.full((height, width), np.nan),
            'mk_trend': np.full((height, width), np.nan),  # -1: decreasing, 0: no trend, 1: increasing
            'mk_slope': np.full((height, width), np.nan)
        }
        
        # Calculate for each pixel
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                
                if np.all(np.isnan(pixel_series)) or len(pixel_series) < 3:
                    continue
                
                try:
                    mk_stat, mk_p, mk_trend, mk_slope = self._mann_kendall_test(pixel_series)
                    
                    results['mk_statistic'][i, j] = mk_stat
                    results['mk_p_value'][i, j] = mk_p
                    results['mk_trend'][i, j] = mk_trend
                    results['mk_slope'][i, j] = mk_slope
                    
                except Exception as e:
                    self.logger.debug(f"Error in Mann-Kendall for pixel ({i}, {j}): {e}")
                    continue
        
        return results
    
    def _mann_kendall_test(self, data: np.ndarray) -> Tuple[float, float, int, float]:
        """Perform Mann-Kendall trend test on a single time series."""
        n = len(data)
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1
        
        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # Calculate Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - self._normal_cdf(abs(z)))
        
        # Determine trend
        if p_value < 0.05:
            if s > 0:
                trend = 1  # increasing
            else:
                trend = -1  # decreasing
        else:
            trend = 0  # no trend
        
        # Calculate Sen's slope
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] != data[i]:
                    slope = (data[j] - data[i]) / (j - i)
                    slopes.append(slope)
        
        sen_slope = np.median(slopes) if slopes else 0
        
        return s, p_value, trend, sen_slope
    
    def _normal_cdf(self, x: float) -> float:
        """Calculate cumulative distribution function of standard normal distribution."""
        return 0.5 * (1 + self._erf(x / np.sqrt(2)))
    
    def _erf(self, x: float) -> float:
        """Calculate error function approximation."""
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return sign * y
    
    def _temporal_anomaly_detection(
        self,
        image_paths: List[Path],
        dates: List[datetime],
        method: str = 'z_score',
        window_size: int = 5,
        threshold: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """Detect temporal anomalies in time series."""
        # Load images
        images = []
        for image_path in image_paths:
            img = self._load_and_preprocess_image(image_path, normalize=True)
            images.append(img)
        
        image_stack = np.stack(images, axis=0)
        
        if method == 'z_score':
            return self._z_score_anomaly_detection(image_stack, dates, window_size, threshold)
        elif method == 'moving_average':
            return self._moving_average_anomaly_detection(image_stack, dates, window_size, threshold)
        else:
            raise ValidationError(f"Unsupported anomaly detection method: {method}")
    
    def _z_score_anomaly_detection(
        self,
        image_stack: np.ndarray,
        dates: List[datetime],
        window_size: int,
        threshold: float
    ) -> Dict[str, np.ndarray]:
        """Detect anomalies using Z-score method with rolling window."""
        height, width = image_stack.shape[1:]
        n_timesteps = len(dates)
        
        anomalies = np.zeros_like(image_stack, dtype=bool)
        z_scores = np.full_like(image_stack, np.nan)
        
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                
                if np.all(np.isnan(pixel_series)):
                    continue
                
                # Calculate rolling statistics
                for t in range(n_timesteps):
                    start_idx = max(0, t - window_size // 2)
                    end_idx = min(n_timesteps, t + window_size // 2 + 1)
                    
                    window_data = pixel_series[start_idx:end_idx]
                    valid_data = window_data[~np.isnan(window_data)]
                    
                    if len(valid_data) > 1:
                        window_mean = np.mean(valid_data)
                        window_std = np.std(valid_data)
                        
                        if window_std > 0:
                            z_score = (pixel_series[t] - window_mean) / window_std
                            z_scores[t, i, j] = z_score
                            
                            if abs(z_score) > threshold:
                                anomalies[t, i, j] = True
        
        return {
            'anomalies': anomalies,
            'z_scores': z_scores,
            'threshold': threshold,
            'window_size': window_size
        }
    
    def _moving_average_anomaly_detection(
        self,
        image_stack: np.ndarray,
        dates: List[datetime],
        window_size: int,
        threshold: float
    ) -> Dict[str, np.ndarray]:
        """Detect anomalies using moving average method."""
        height, width = image_stack.shape[1:]
        
        anomalies = np.zeros_like(image_stack, dtype=bool)
        moving_avg = np.full_like(image_stack, np.nan)
        deviations = np.full_like(image_stack, np.nan)
        
        for i in range(height):
            for j in range(width):
                pixel_series = image_stack[:, i, j]
                
                if np.all(np.isnan(pixel_series)):
                    continue
                
                # Calculate moving average
                ma = self._moving_average(pixel_series, window_size)
                moving_avg[:, i, j] = ma
                
                # Calculate deviations
                dev = pixel_series - ma
                deviations[:, i, j] = dev
                
                # Detect anomalies
                anomaly_threshold = threshold * np.nanstd(dev)
                anomalies[:, i, j] = np.abs(dev) > anomaly_threshold
        
        return {
            'anomalies': anomalies,
            'moving_average': moving_avg,
            'deviations': deviations,
            'threshold': threshold,
            'window_size': window_size
        }
    
    def _load_and_preprocess_image(self, image_path: Path, normalize: bool = True) -> np.ndarray:
        """Load and preprocess an image."""
        try:
            with rasterio.open(image_path) as src:
                # Read first band for simplicity (could be extended to multi-band)
                data = src.read(1)
                
                if normalize:
                    # Normalize to 0-1 range
                    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-8)
                
                return data
                
        except Exception as e:
            raise ProcessingError(f"Error loading image {image_path}: {e}")
    
    def _resample_to_match(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resample image to match target shape."""
        from scipy.ndimage import zoom
        
        zoom_factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
        return zoom(image, zoom_factors, order=1)
    
    def _moving_average(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """Calculate moving average with specified window size."""
        if window_size <= 1:
            return data
        
        # Pad data for edge handling
        padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
        
        # Calculate moving average
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            window = padded[i:i + window_size]
            valid_data = window[~np.isnan(window)]
            if len(valid_data) > 0:
                result[i] = np.mean(valid_data)
        
        return result
    
    def _bilateral_filter(self, image: np.ndarray, sigma_color: float = 0.1, sigma_space: float = 1.0) -> np.ndarray:
        """Apply bilateral filter to image."""
        from scipy.ndimage import gaussian_filter
        
        # Simplified bilateral filter implementation
        # In production, you might use scikit-image's bilateral filter
        
        # Spatial filtering
        spatial_filtered = gaussian_filter(image, sigma=sigma_space)
        
        # Color filtering (simplified)
        color_filtered = gaussian_filter(image, sigma=sigma_color)
        
        # Combine (simplified approach)
        result = 0.5 * spatial_filtered + 0.5 * color_filtered
        
        return result
    
    def save_results(
        self,
        results: Dict[str, np.ndarray],
        output_path: Path,
        format: str = 'geotiff',
        reference_image: Optional[Path] = None
    ) -> Path:
        """Save processing results to file."""
        if format.lower() == 'geotiff':
            return self._save_geotiff(results, output_path, reference_image)
        elif format.lower() == 'numpy':
            return self._save_numpy(results, output_path)
        else:
            raise ValidationError(f"Unsupported output format: {format}")
    
    def _save_geotiff(
        self,
        results: Dict[str, np.ndarray],
        reference_image: Optional[Path],
        output_path: Path
    ) -> Path:
        """Save results as GeoTIFF files."""
        if reference_image is None:
            raise ValidationError("Reference image required for GeoTIFF output")
        
        # Get reference profile
        with rasterio.open(reference_image) as src:
            profile = src.profile.copy()
        
        # Save each result
        saved_files = []
        for key, data in results.items():
            if isinstance(data, np.ndarray):
                output_file = output_path.parent / f"{output_path.stem}_{key}.tif"
                
                # Update profile for single band
                profile.update(
                    count=1,
                    dtype=data.dtype,
                    nodata=np.nan
                )
                
                with rasterio.open(output_file, 'w', **profile) as dst:
                    dst.write(data, 1)
                
                saved_files.append(output_file)
        
        return output_path.parent
    
    def _save_numpy(self, results: Dict[str, np.ndarray], output_path: Path) -> Path:
        """Save results as numpy files."""
        output_file = output_path.with_suffix('.npz')
        
        # Prepare data for saving
        save_data = {}
        for key, data in results.items():
            if isinstance(data, np.ndarray):
                save_data[key] = data
        
        np.savez_compressed(output_file, **save_data)
        return output_file

# Convenience functions
def analyze_multi_temporal(
    image_paths: List[Path],
    dates: List[datetime],
    analysis_type: str = 'change_detection',
    **kwargs
) -> Dict[str, np.ndarray]:
    """Convenience function for multi-temporal analysis."""
    processor = AdvancedProcessor()
    return processor.multi_temporal_analysis(image_paths, dates, analysis_type, **kwargs)

def detect_temporal_anomalies(
    image_paths: List[Path],
    dates: List[datetime],
    method: str = 'z_score',
    **kwargs
) -> Dict[str, np.ndarray]:
    """Convenience function for temporal anomaly detection."""
    processor = AdvancedProcessor()
    return processor._temporal_anomaly_detection(image_paths, dates, method, **kwargs)
