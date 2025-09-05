"""
Advanced Anomaly Detection for Satellite Imagery
This module provides multiple algorithms to identify unusual patterns, changes, and anomalies
in satellite data including NDVI, NDWI, and spectral bands.
"""

import numpy as np
import rasterio
from scipy import stats
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path


class SatelliteAnomalyDetector:
    """Advanced anomaly detection for satellite imagery using multiple algorithms."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
    def statistical_anomaly_detection(self, data, method='zscore', threshold=3.0):
        """
        Detect anomalies using statistical methods.
        
        Parameters:
        - data: numpy array of satellite data
        - method: 'zscore', 'iqr', or 'percentile'
        - threshold: threshold for anomaly detection
        
        Returns:
        - anomaly_mask: boolean mask where True indicates anomalies
        - anomaly_scores: numerical scores for each pixel
        """
        if method == 'zscore':
            # Z-score method: detect values beyond threshold standard deviations
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            anomaly_mask = z_scores > threshold
            anomaly_scores = z_scores
            
        elif method == 'iqr':
            # Interquartile range method: detect values beyond 1.5 * IQR
            q1 = np.nanpercentile(data, 25)
            q3 = np.nanpercentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomaly_mask = (data < lower_bound) | (data > upper_bound)
            anomaly_scores = np.where(anomaly_mask, 
                                    np.abs(data - np.nanmedian(data)) / iqr, 0)
            
        elif method == 'percentile':
            # Percentile method: detect values beyond specified percentiles
            lower_percentile = np.nanpercentile(data, threshold)
            upper_percentile = np.nanpercentile(data, 100 - threshold)
            anomaly_mask = (data < lower_percentile) | (data > upper_percentile)
            anomaly_scores = np.where(anomaly_mask, 
                                    np.abs(data - np.nanmedian(data)), 0)
        
        return anomaly_mask, anomaly_scores
    
    def temporal_change_anomaly(self, time_series_data, window_size=5, threshold=2.0):
        """
        Detect anomalies based on temporal changes in time series data.
        
        Parameters:
        - time_series_data: 3D array (time, height, width)
        - window_size: size of moving window for trend analysis
        - threshold: threshold for change detection
        
        Returns:
        - anomaly_mask: boolean mask of anomalous changes
        - change_magnitude: magnitude of changes
        """
        if len(time_series_data) < window_size:
            raise ValueError("Time series too short for window analysis")
        
        # Calculate temporal statistics
        mean_trend = np.nanmean(time_series_data, axis=0)
        std_trend = np.nanstd(time_series_data, axis=0)
        
        # Detect sudden changes
        change_magnitude = np.zeros_like(mean_trend)
        anomaly_mask = np.zeros_like(mean_trend, dtype=bool)
        
        for i in range(window_size, len(time_series_data)):
            # Calculate change from previous window
            current_window = time_series_data[i-window_size:i]
            previous_window = time_series_data[i-window_size-1:i-1]
            
            # Change magnitude
            change = np.nanmean(current_window, axis=0) - np.nanmean(previous_window, axis=0)
            change_magnitude += np.abs(change)
            
            # Detect if change is anomalous
            change_zscore = np.abs(change) / (std_trend + 1e-8)
            anomaly_mask |= change_zscore > threshold
        
        return anomaly_mask, change_magnitude
    
    def spatial_anomaly_detection(self, data, neighborhood_size=5, threshold=2.0):
        """
        Detect spatial anomalies by comparing pixels to their local neighborhood.
        
        Parameters:
        - data: 2D array of satellite data
        - neighborhood_size: size of neighborhood window
        - threshold: threshold for anomaly detection
        
        Returns:
        - anomaly_mask: boolean mask of spatial anomalies
        - local_deviation: local deviation scores
        """
        # Apply uniform filter to get local mean
        local_mean = uniform_filter(data, size=neighborhood_size)
        
        # Calculate local standard deviation
        local_var = uniform_filter(data**2, size=neighborhood_size) - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Calculate deviation from local mean
        local_deviation = np.abs(data - local_mean) / (local_std + 1e-8)
        
        # Detect anomalies
        anomaly_mask = local_deviation > threshold
        
        return anomaly_mask, local_deviation
    
    def spectral_anomaly_detection(self, spectral_data, contamination=0.1):
        """
        Detect spectral anomalies using machine learning approaches.
        
        Parameters:
        - spectral_data: 3D array (bands, height, width)
        - contamination: expected fraction of anomalies
        
        Returns:
        - anomaly_mask: boolean mask of spectral anomalies
        - anomaly_scores: anomaly scores from isolation forest
        """
        # Reshape data for machine learning
        original_shape = spectral_data.shape
        spectral_data_2d = spectral_data.reshape(original_shape[0], -1).T
        
        # Remove NaN values
        valid_mask = ~np.any(np.isnan(spectral_data_2d), axis=1)
        valid_data = spectral_data_2d[valid_mask]
        
        if len(valid_data) == 0:
            return np.zeros(original_shape[1:], dtype=bool), np.zeros(original_shape[1:])
        
        # Standardize data
        scaled_data = self.scaler.fit_transform(valid_data)
        
        # Fit isolation forest
        self.isolation_forest.set_params(contamination=contamination)
        anomaly_labels = self.isolation_forest.fit_predict(scaled_data)
        anomaly_scores = self.isolation_forest.decision_function(scaled_data)
        
        # Create output arrays
        full_anomaly_mask = np.zeros(original_shape[1:], dtype=bool)
        full_anomaly_scores = np.zeros(original_shape[1:])
        
        # Map results back to original positions
        valid_indices = np.where(valid_mask)
        full_anomaly_mask.flat[valid_indices] = (anomaly_labels == -1)
        full_anomaly_scores.flat[valid_indices] = -anomaly_scores  # Negative for anomalies
        
        return full_anomaly_mask, full_anomaly_scores
    
    def ndvi_anomaly_detection(self, ndvi_data, method='combined', **kwargs):
        """
        Specialized anomaly detection for NDVI data.
        
        Parameters:
        - ndvi_data: NDVI array or time series
        - method: detection method ('statistical', 'temporal', 'spatial', 'combined')
        - **kwargs: additional parameters for specific methods
        
        Returns:
        - anomaly_mask: boolean mask of NDVI anomalies
        - anomaly_info: dictionary with detailed anomaly information
        """
        anomaly_info = {}
        
        if method == 'statistical' or method == 'combined':
            stat_mask, stat_scores = self.statistical_anomaly_detection(
                ndvi_data, method='iqr', threshold=kwargs.get('stat_threshold', 1.5)
            )
            anomaly_info['statistical'] = {'mask': stat_mask, 'scores': stat_scores}
        
        if method == 'spatial' or method == 'combined':
            spatial_mask, spatial_scores = self.spatial_anomaly_detection(
                ndvi_data, 
                neighborhood_size=kwargs.get('neighborhood_size', 5),
                threshold=kwargs.get('spatial_threshold', 2.0)
            )
            anomaly_info['spatial'] = {'mask': spatial_mask, 'scores': spatial_scores}
        
        if method == 'combined':
            # Combine different detection methods
            combined_mask = np.zeros_like(ndvi_data, dtype=bool)
            for method_name, method_data in anomaly_info.items():
                combined_mask |= method_data['mask']
            
            # Calculate confidence scores
            confidence_scores = np.zeros_like(ndvi_data)
            for method_name, method_data in anomaly_info.items():
                confidence_scores += method_data['scores']
            
            anomaly_info['combined'] = {
                'mask': combined_mask,
                'scores': confidence_scores,
                'confidence': confidence_scores / len(anomaly_info)
            }
            
            return anomaly_info['combined']['mask'], anomaly_info
        
        # Return the first available method
        method_name = list(anomaly_info.keys())[0]
        return anomaly_info[method_name]['mask'], anomaly_info
    
    def water_anomaly_detection(self, ndwi_data, water_threshold=0.3, **kwargs):
        """
        Specialized anomaly detection for water bodies using NDWI.
        
        Parameters:
        - ndwi_data: NDWI array
        - water_threshold: threshold for water classification
        - **kwargs: additional parameters
        
        Returns:
        - anomaly_mask: boolean mask of water anomalies
        - anomaly_info: dictionary with water anomaly details
        """
        # Identify water bodies
        water_mask = ndwi_data > water_threshold
        
        # Detect anomalies in water areas
        water_anomalies = np.zeros_like(ndwi_data, dtype=bool)
        
        if np.any(water_mask):
            # Analyze water areas for unusual patterns
            water_data = ndwi_data[water_mask]
            
            # Detect statistical anomalies in water
            water_anomaly_mask, water_scores = self.statistical_anomaly_detection(
                water_data, method='zscore', threshold=kwargs.get('water_threshold', 2.5)
            )
            
            # Map back to full array
            water_anomalies[water_mask] = water_anomaly_mask
        
        # Detect spatial anomalies (unusual water patterns)
        spatial_anomalies, spatial_scores = self.spatial_anomaly_detection(
            ndwi_data, 
            neighborhood_size=kwargs.get('neighborhood_size', 7),
            threshold=kwargs.get('spatial_threshold', 2.0)
        )
        
        # Combine water-specific and spatial anomalies
        combined_anomalies = water_anomalies | spatial_anomalies
        
        anomaly_info = {
            'water_specific': {'mask': water_anomalies, 'scores': water_scores if 'water_scores' in locals() else np.zeros_like(ndwi_data)},
            'spatial': {'mask': spatial_anomalies, 'scores': spatial_scores},
            'combined': {'mask': combined_anomalies, 'scores': spatial_scores}
        }
        
        return combined_anomalies, anomaly_info
    
    def save_anomaly_results(self, output_path, anomaly_mask, anomaly_scores, 
                           profile, description="Anomaly Detection Results"):
        """
        Save anomaly detection results as GeoTIFF files.
        
        Parameters:
        - output_path: path to save results
        - anomaly_mask: boolean anomaly mask
        - anomaly_scores: numerical anomaly scores
        - profile: rasterio profile for georeference
        - description: description of the results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save anomaly mask
        mask_profile = profile.copy()
        mask_profile.update({
            'count': 1,
            'dtype': 'uint8',
            'compress': 'deflate',
            'nodata': 255
        })
        
        with rasterio.open(str(output_path / 'anomaly_mask.tif'), 'w', **mask_profile) as dst:
            dst.write(anomaly_mask.astype('uint8'), 1)
        
        # Save anomaly scores
        scores_profile = profile.copy()
        scores_profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'deflate',
            'nodata': np.nan
        })
        
        with rasterio.open(str(output_path / 'anomaly_scores.tif'), 'w', **scores_profile) as dst:
            dst.write(anomaly_scores.astype('float32'), 1)
        
        # Save metadata
        metadata = {
            'description': description,
            'anomaly_count': int(np.sum(anomaly_mask)),
            'total_pixels': int(anomaly_mask.size),
            'anomaly_percentage': float(np.sum(anomaly_mask) / anomaly_mask.size * 100),
            'score_range': [float(np.nanmin(anomaly_scores)), float(np.nanmax(anomaly_scores))]
        }
        
        metadata_path = output_path / 'anomaly_metadata.txt'
        with open(metadata_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Anomaly detection results saved to {output_path}")
        print(f"Found {metadata['anomaly_count']} anomalous pixels ({metadata['anomaly_percentage']:.2f}%)")


def detect_anomalies_in_geotiff(geotiff_path, output_dir, detection_methods=None, **kwargs):
    """
    Convenience function to detect anomalies in a GeoTIFF file.
    
    Parameters:
    - geotiff_path: path to input GeoTIFF
    - output_dir: directory to save results
    - detection_methods: list of detection methods to use
    - **kwargs: additional parameters for detection methods
    
    Returns:
    - anomaly_mask: detected anomalies
    - anomaly_info: detailed anomaly information
    """
    if detection_methods is None:
        detection_methods = ['statistical', 'spatial']
    
    # Load GeoTIFF
    with rasterio.open(geotiff_path) as src:
        data = src.read(1)
        profile = src.profile
    
    # Initialize detector
    detector = SatelliteAnomalyDetector()
    
    # Detect anomalies
    anomaly_mask = np.zeros_like(data, dtype=bool)
    anomaly_scores = np.zeros_like(data)
    anomaly_info = {}
    
    for method in detection_methods:
        if method == 'statistical':
            mask, scores = detector.statistical_anomaly_detection(data, **kwargs)
            anomaly_mask |= mask
            anomaly_scores += scores
            anomaly_info[method] = {'mask': mask, 'scores': scores}
            
        elif method == 'spatial':
            mask, scores = detector.spatial_anomaly_detection(data, **kwargs)
            anomaly_mask |= mask
            anomaly_scores += scores
            anomaly_info[method] = {'mask': mask, 'scores': scores}
            
        elif method == 'spectral' and len(data.shape) > 2:
            mask, scores = detector.spectral_anomaly_detection(data, **kwargs)
            anomaly_mask |= mask
            anomaly_scores += scores
            anomaly_info[method] = {'mask': mask, 'scores': scores}
    
    # Save results
    output_path = Path(output_dir)
    detector.save_anomaly_results(
        output_path, anomaly_mask, anomaly_scores, profile,
        description=f"Anomaly detection using methods: {', '.join(detection_methods)}"
    )
    
    return anomaly_mask, anomaly_info


if __name__ == '__main__':
    # Example usage
    print("Satellite Anomaly Detection Module")
    print("Use detect_anomalies_in_geotiff() function for quick analysis")
    print("Or create SatelliteAnomalyDetector instance for advanced usage")
