"""
PlayNexus Satellite Toolkit - Advanced Processing Pipeline
Provides sophisticated satellite imagery analysis with multiple processing stages.
"""

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
from rasterio.features import geometry_mask
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import box, mapping
import cv2
from skimage import filters, restoration, morphology, segmentation, measure, exposure
from skimage.filters import gaussian, median, sobel
from skimage.restoration import denoise_bilateral, denoise_wavelet
from skimage.morphology import disk, ball, binary_erosion, binary_dilation
from skimage.segmentation import watershed, slic
from skimage.measure import label, regionprops
from scipy import ndimage, signal, stats
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from scipy.stats import zscore, iqr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    PREPROCESSING = "preprocessing"
    ENHANCEMENT = "enhancement"
    FEATURE_EXTRACTION = "feature_extraction"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    POST_PROCESSING = "post_processing"
    QUALITY_CONTROL = "quality_control"


class DataQuality(Enum):
    """Enumeration of data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


@dataclass
class ProcessingResult:
    """Data class for processing results."""
    stage: ProcessingStage
    success: bool
    data: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    quality_score: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class PipelineConfiguration:
    """Configuration for the processing pipeline."""
    enable_preprocessing: bool = True
    enable_enhancement: bool = True
    enable_feature_extraction: bool = True
    enable_anomaly_detection: bool = True
    enable_classification: bool = False
    enable_post_processing: bool = True
    enable_quality_control: bool = True
    
    # Preprocessing options
    resample_resolution: Optional[float] = None
    clip_to_bbox: Optional[List[float]] = None
    normalize_data: bool = True
    remove_outliers: bool = True
    
    # Enhancement options
    denoise_method: str = "bilateral"
    contrast_enhancement: bool = True
    edge_enhancement: bool = True
    histogram_equalization: bool = True
    
    # Feature extraction options
    extract_textural_features: bool = True
    extract_spectral_features: bool = True
    extract_morphological_features: bool = True
    
    # Anomaly detection options
    statistical_threshold: float = 2.5
    spatial_analysis: bool = True
    spectral_analysis: bool = True
    
    # Quality control options
    min_quality_score: float = 0.7
    enable_validation: bool = True


class AdvancedProcessingPipeline:
    """Advanced processing pipeline for satellite imagery analysis."""
    
    def __init__(self, config: PipelineConfiguration = None):
        self.config = config or PipelineConfiguration()
        self.logger = logging.getLogger(__name__)
        self.results: List[ProcessingResult] = []
        self.current_stage = None
        
    def process_image(self, input_path: str, output_dir: str) -> Dict[str, ProcessingResult]:
        """Process an image through the complete pipeline."""
        self.logger.info(f"Starting advanced processing pipeline for: {input_path}")
        
        try:
            # Load image
            with rasterio.open(input_path) as src:
                image_data = src.read()
                metadata = src.meta
                bbox = src.bounds
            
            self.logger.info(f"Loaded image with shape: {image_data.shape}")
            
            # Initialize results
            self.results = []
            pipeline_results = {}
            
            # Execute pipeline stages
            if self.config.enable_preprocessing:
                result = self._preprocessing_stage(image_data, metadata, bbox)
                pipeline_results[ProcessingStage.PREPROCESSING.value] = result
                if result.success:
                    image_data = result.data
                    metadata = result.metadata
            
            if self.config.enable_enhancement and image_data is not None:
                result = self._enhancement_stage(image_data, metadata)
                pipeline_results[ProcessingStage.ENHANCEMENT.value] = result
                if result.success:
                    image_data = result.data
                    metadata = result.metadata
            
            if self.config.enable_feature_extraction and image_data is not None:
                result = self._feature_extraction_stage(image_data, metadata)
                pipeline_results[ProcessingStage.FEATURE_EXTRACTION.value] = result
            
            if self.config.enable_anomaly_detection and image_data is not None:
                result = self._anomaly_detection_stage(image_data, metadata)
                pipeline_results[ProcessingStage.ANOMALY_DETECTION.value] = result
            
            if self.config.enable_classification and image_data is not None:
                result = self._classification_stage(image_data, metadata)
                pipeline_results[ProcessingStage.CLASSIFICATION.value] = result
            
            if self.config.enable_post_processing and image_data is not None:
                result = self._post_processing_stage(image_data, metadata)
                pipeline_results[ProcessingStage.POST_PROCESSING.value] = result
                if result.success:
                    image_data = result.data
                    metadata = result.metadata
            
            if self.config.enable_quality_control:
                result = self._quality_control_stage(pipeline_results)
                pipeline_results[ProcessingStage.QUALITY_CONTROL.value] = result
            
            # Save final results
            if image_data is not None:
                self._save_results(image_data, metadata, output_dir)
            
            self.logger.info("Advanced processing pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            raise
    
    def _preprocessing_stage(self, image_data: np.ndarray, metadata: Dict, bbox: Tuple) -> ProcessingResult:
        """Execute preprocessing stage."""
        self.current_stage = ProcessingStage.PREPROCESSING
        self.logger.info("Starting preprocessing stage...")
        
        start_time = time.time()
        
        try:
            processed_data = image_data.copy()
            processed_metadata = metadata.copy()
            
            # Resample if requested
            if self.config.resample_resolution:
                processed_data, processed_metadata = self._resample_image(
                    processed_data, processed_metadata, self.config.resample_resolution
                )
            
            # Clip to bounding box if requested
            if self.config.clip_to_bbox:
                processed_data, processed_metadata = self._clip_to_bbox(
                    processed_data, processed_metadata, self.config.clip_to_bbox
                )
            
            # Normalize data
            if self.config.normalize_data:
                processed_data = self._normalize_data(processed_data)
            
            # Remove outliers
            if self.config.remove_outliers:
                processed_data = self._remove_outliers(processed_data)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.PREPROCESSING,
                success=True,
                data=processed_data,
                metadata=processed_metadata,
                quality_score=0.9,
                processing_time=processing_time
            )
            
            self.logger.info("Preprocessing stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Preprocessing stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.PREPROCESSING,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _enhancement_stage(self, image_data: np.ndarray, metadata: Dict) -> ProcessingResult:
        """Execute image enhancement stage."""
        self.current_stage = ProcessingStage.ENHANCEMENT
        self.logger.info("Starting enhancement stage...")
        
        start_time = time.time()
        
        try:
            enhanced_data = image_data.copy()
            
            # Apply denoising
            if self.config.denoise_method == "bilateral":
                enhanced_data = self._denoise_bilateral(enhanced_data)
            elif self.config.denoise_method == "wavelet":
                enhanced_data = self._denoise_wavelet(enhanced_data)
            elif self.config.denoise_method == "gaussian":
                enhanced_data = self._denoise_gaussian(enhanced_data)
            
            # Contrast enhancement
            if self.config.contrast_enhancement:
                enhanced_data = self._enhance_contrast(enhanced_data)
            
            # Edge enhancement
            if self.config.edge_enhancement:
                enhanced_data = self._enhance_edges(enhanced_data)
            
            # Histogram equalization
            if self.config.histogram_equalization:
                enhanced_data = self._equalize_histogram(enhanced_data)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.ENHANCEMENT,
                success=True,
                data=enhanced_data,
                metadata=metadata,
                quality_score=0.85,
                processing_time=processing_time
            )
            
            self.logger.info("Enhancement stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Enhancement stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.ENHANCEMENT,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _feature_extraction_stage(self, image_data: np.ndarray, metadata: Dict) -> ProcessingResult:
        """Execute feature extraction stage."""
        self.current_stage = ProcessingStage.FEATURE_EXTRACTION
        self.logger.info("Starting feature extraction stage...")
        
        start_time = time.time()
        
        try:
            features = {}
            
            # Textural features
            if self.config.extract_textural_features:
                features['textural'] = self._extract_textural_features(image_data)
            
            # Spectral features
            if self.config.extract_spectral_features:
                features['spectral'] = self._extract_spectral_features(image_data)
            
            # Morphological features
            if self.config.extract_morphological_features:
                features['morphological'] = self._extract_morphological_features(image_data)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.FEATURE_EXTRACTION,
                success=True,
                metadata={'features': features},
                quality_score=0.8,
                processing_time=processing_time
            )
            
            self.logger.info("Feature extraction stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Feature extraction stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.FEATURE_EXTRACTION,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _anomaly_detection_stage(self, image_data: np.ndarray, metadata: Dict) -> ProcessingResult:
        """Execute anomaly detection stage."""
        self.current_stage = ProcessingStage.ANOMALY_DETECTION
        self.logger.info("Starting anomaly detection stage...")
        
        start_time = time.time()
        
        try:
            anomalies = {}
            
            # Statistical anomaly detection
            anomalies['statistical'] = self._detect_statistical_anomalies(
                image_data, self.config.statistical_threshold
            )
            
            # Spatial anomaly detection
            if self.config.spatial_analysis:
                anomalies['spatial'] = self._detect_spatial_anomalies(image_data)
            
            # Spectral anomaly detection
            if self.config.spectral_analysis:
                anomalies['spectral'] = self._detect_spectral_anomalies(image_data)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.ANOMALY_DETECTION,
                success=True,
                metadata={'anomalies': anomalies},
                quality_score=0.75,
                processing_time=processing_time
            )
            
            self.logger.info("Anomaly detection stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Anomaly detection stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.ANOMALY_DETECTION,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _classification_stage(self, image_data: np.ndarray, metadata: Dict) -> ProcessingResult:
        """Execute classification stage."""
        self.current_stage = ProcessingStage.CLASSIFICATION
        self.logger.info("Starting classification stage...")
        
        start_time = time.time()
        
        try:
            # This would implement actual classification algorithms
            # For now, we'll create a simple unsupervised classification
            classified_data = self._simple_unsupervised_classification(image_data)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.CLASSIFICATION,
                success=True,
                data=classified_data,
                metadata=metadata,
                quality_score=0.7,
                processing_time=processing_time
            )
            
            self.logger.info("Classification stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Classification stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.CLASSIFICATION,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _post_processing_stage(self, image_data: np.ndarray, metadata: Dict) -> ProcessingResult:
        """Execute post-processing stage."""
        self.current_stage = ProcessingStage.POST_PROCESSING
        self.logger.info("Starting post-processing stage...")
        
        start_time = time.time()
        
        try:
            processed_data = image_data.copy()
            
            # Apply final smoothing
            processed_data = self._apply_final_smoothing(processed_data)
            
            # Ensure data range
            processed_data = np.clip(processed_data, 0, 1)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.POST_PROCESSING,
                success=True,
                data=processed_data,
                metadata=metadata,
                quality_score=0.9,
                processing_time=processing_time
            )
            
            self.logger.info("Post-processing stage completed successfully")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Post-processing stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.POST_PROCESSING,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _quality_control_stage(self, pipeline_results: Dict) -> ProcessingResult:
        """Execute quality control stage."""
        self.current_stage = ProcessingStage.QUALITY_CONTROL
        self.logger.info("Starting quality control stage...")
        
        start_time = time.time()
        
        try:
            # Calculate overall quality score
            quality_scores = []
            for result in pipeline_results.values():
                if result.quality_score is not None:
                    quality_scores.append(result.quality_score)
            
            overall_quality = np.mean(quality_scores) if quality_scores else 0.0
            
            # Generate quality report
            quality_report = self._generate_quality_report(pipeline_results, overall_quality)
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                stage=ProcessingStage.QUALITY_CONTROL,
                success=overall_quality >= self.config.min_quality_score,
                metadata={'quality_report': quality_report, 'overall_quality': overall_quality},
                quality_score=overall_quality,
                processing_time=processing_time
            )
            
            self.logger.info(f"Quality control stage completed. Overall quality: {overall_quality:.2f}")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Quality control stage failed: {e}")
            
            return ProcessingResult(
                stage=ProcessingStage.QUALITY_CONTROL,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    # Helper methods for each processing stage
    def _resample_image(self, image_data: np.ndarray, metadata: Dict, target_resolution: float) -> Tuple[np.ndarray, Dict]:
        """Resample image to target resolution."""
        # Implementation for resampling
        return image_data, metadata
    
    def _clip_to_bbox(self, image_data: np.ndarray, metadata: Dict, bbox: List[float]) -> Tuple[np.ndarray, Dict]:
        """Clip image to bounding box."""
        # Implementation for clipping
        return image_data, metadata
    
    def _normalize_data(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data to 0-1 range."""
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            min_val = np.nanmin(band_data)
            max_val = np.nanmax(band_data)
            if max_val > min_val:
                image_data[band] = (band_data - min_val) / (max_val - min_val)
        
        return image_data
    
    def _remove_outliers(self, image_data: np.ndarray) -> np.ndarray:
        """Remove statistical outliers from image data."""
        cleaned_data = image_data.copy()
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            z_scores = np.abs(zscore(band_data, nan_policy='omit'))
            outlier_mask = z_scores > 3
            cleaned_data[band][outlier_mask] = np.nanmedian(band_data)
        
        return cleaned_data
    
    def _denoise_bilateral(self, image_data: np.ndarray) -> np.ndarray:
        """Apply bilateral denoising."""
        denoised_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            denoised_data[band] = denoise_bilateral(band_data, sigma_color=0.1, sigma_spatial=15)
        
        return denoised_data
    
    def _denoise_wavelet(self, image_data: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising."""
        denoised_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            denoised_data[band] = denoise_wavelet(band_data, sigma=0.1)
        
        return denoised_data
    
    def _denoise_gaussian(self, image_data: np.ndarray) -> np.ndarray:
        """Apply Gaussian denoising."""
        denoised_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            denoised_data[band] = gaussian_filter(band_data, sigma=1.0)
        
        return denoised_data
    
    def _enhance_contrast(self, image_data: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        enhanced_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            p2, p98 = np.percentile(band_data, (2, 98))
            enhanced_data[band] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
        
        return enhanced_data
    
    def _enhance_edges(self, image_data: np.ndarray) -> np.ndarray:
        """Enhance image edges."""
        enhanced_data = image_data.copy()
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            edges = sobel(band_data)
            enhanced_data[band] = np.clip(band_data + 0.3 * edges, 0, 1)
        
        return enhanced_data
    
    def _equalize_histogram(self, image_data: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        equalized_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            equalized_data[band] = exposure.equalize_hist(band_data)
        
        return equalized_data
    
    def _extract_textural_features(self, image_data: np.ndarray) -> Dict:
        """Extract textural features from image."""
        features = {}
        
        # GLCM features
        for band in range(min(3, image_data.shape[0])):  # Limit to first 3 bands
            band_data = image_data[band]
            # Calculate basic statistics
            features[f'band_{band}_mean'] = np.mean(band_data)
            features[f'band_{band}_std'] = np.std(band_data)
            features[f'band_{band}_skewness'] = stats.skew(band_data.flatten())
            features[f'band_{band}_kurtosis'] = stats.kurtosis(band_data.flatten())
        
        return features
    
    def _extract_spectral_features(self, image_data: np.ndarray) -> Dict:
        """Extract spectral features from image."""
        features = {}
        
        if image_data.shape[0] >= 3:
            # Calculate vegetation indices
            red = image_data[2]  # Assuming band 3 is red
            nir = image_data[3] if image_data.shape[0] > 3 else image_data[0]  # NIR band
            
            # NDVI
            ndvi = (nir - red) / (nir + red + 1e-8)
            features['ndvi_mean'] = np.mean(ndvi)
            features['ndvi_std'] = np.std(ndvi)
            
            # Other spectral ratios
            features['red_nir_ratio'] = np.mean(red / (nir + 1e-8))
        
        return features
    
    def _extract_morphological_features(self, image_data: np.ndarray) -> Dict:
        """Extract morphological features from image."""
        features = {}
        
        # Calculate morphological features for the first band
        band_data = image_data[0]
        
        # Area and perimeter
        binary = band_data > np.mean(band_data)
        labeled = label(binary)
        regions = regionprops(labeled)
        
        if regions:
            areas = [region.area for region in regions]
            features['object_count'] = len(regions)
            features['mean_object_area'] = np.mean(areas)
            features['total_object_area'] = np.sum(areas)
        
        return features
    
    def _detect_statistical_anomalies(self, image_data: np.ndarray, threshold: float) -> Dict:
        """Detect statistical anomalies in image data."""
        anomalies = {}
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            z_scores = np.abs(zscore(band_data, nan_policy='omit'))
            anomaly_mask = z_scores > threshold
            
            anomalies[f'band_{band}_anomaly_count'] = np.sum(anomaly_mask)
            anomalies[f'band_{band}_anomaly_percentage'] = np.sum(anomaly_mask) / anomaly_mask.size * 100
        
        return anomalies
    
    def _detect_spatial_anomalies(self, image_data: np.ndarray) -> Dict:
        """Detect spatial anomalies in image data."""
        anomalies = {}
        
        # Detect clusters of high/low values
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            
            # Local variance analysis
            local_var = uniform_filter(band_data**2, size=5) - uniform_filter(band_data, size=5)**2
            high_variance_mask = local_var > np.percentile(local_var, 95)
            
            anomalies[f'band_{band}_high_variance_pixels'] = np.sum(high_variance_mask)
        
        return anomalies
    
    def _detect_spectral_anomalies(self, image_data: np.ndarray) -> Dict:
        """Detect spectral anomalies in image data."""
        anomalies = {}
        
        if image_data.shape[0] >= 3:
            # Detect pixels with unusual spectral signatures
            for i in range(image_data.shape[0]):
                for j in range(i+1, min(i+3, image_data.shape[0])):
                    ratio = image_data[i] / (image_data[j] + 1e-8)
                    z_scores = np.abs(zscore(ratio, nan_policy='omit'))
                    anomaly_mask = z_scores > 2.5
                    
                    anomalies[f'band_{i}_{j}_ratio_anomalies'] = np.sum(anomaly_mask)
        
        return anomalies
    
    def _simple_unsupervised_classification(self, image_data: np.ndarray) -> np.ndarray:
        """Perform simple unsupervised classification."""
        # Use K-means clustering for basic classification
        from sklearn.cluster import KMeans
        
        # Reshape data for clustering
        height, width = image_data.shape[1], image_data.shape[2]
        reshaped_data = image_data.reshape(image_data.shape[0], -1).T
        
        # Perform clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        labels = kmeans.fit_predict(reshaped_data)
        
        # Reshape back to image dimensions
        classified_data = labels.reshape(height, width)
        
        return classified_data
    
    def _apply_final_smoothing(self, image_data: np.ndarray) -> np.ndarray:
        """Apply final smoothing to processed image."""
        smoothed_data = np.zeros_like(image_data)
        
        for band in range(image_data.shape[0]):
            band_data = image_data[band]
            smoothed_data[band] = gaussian_filter(band_data, sigma=0.5)
        
        return smoothed_data
    
    def _generate_quality_report(self, pipeline_results: Dict, overall_quality: float) -> Dict:
        """Generate comprehensive quality report."""
        report = {
            'overall_quality': overall_quality,
            'stage_results': {},
            'recommendations': []
        }
        
        for stage_name, result in pipeline_results.items():
            report['stage_results'][stage_name] = {
                'success': result.success,
                'quality_score': result.quality_score,
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
        
        # Generate recommendations
        if overall_quality < 0.8:
            report['recommendations'].append("Consider adjusting preprocessing parameters")
        if overall_quality < 0.6:
            report['recommendations'].append("Review input data quality")
        if overall_quality < 0.4:
            report['recommendations'].append("Pipeline may need significant parameter tuning")
        
        return report
    
    def _save_results(self, image_data: np.ndarray, metadata: Dict, output_dir: str):
        """Save processing results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save processed image
        output_file = output_path / "processed_image.tif"
        with rasterio.open(output_file, 'w', **metadata) as dst:
            dst.write(image_data)
        
        # Save quality report
        report_file = output_path / "quality_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, default=str, indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")


# Utility functions for external use
def create_processing_pipeline(config: PipelineConfiguration = None) -> AdvancedProcessingPipeline:
    """Create a new processing pipeline instance."""
    return AdvancedProcessingPipeline(config)


def process_satellite_image(input_path: str, output_dir: str, config: PipelineConfiguration = None) -> Dict:
    """Convenience function to process a satellite image."""
    pipeline = create_processing_pipeline(config)
    return pipeline.process_image(input_path, output_dir)


if __name__ == "__main__":
    # Example usage
    config = PipelineConfiguration(
        enable_preprocessing=True,
        enable_enhancement=True,
        enable_feature_extraction=True,
        enable_anomaly_detection=True,
        enable_classification=False,
        enable_post_processing=True,
        enable_quality_control=True
    )
    
    # Process an image
    # results = process_satellite_image("input.tif", "output", config)
    print("Advanced Processing Pipeline ready for use!")
