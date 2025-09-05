"""
Advanced Image Enhancement for Satellite Imagery
This module provides various image enhancement techniques to improve visibility,
highlight anomalies, and enhance contrast in satellite data.
"""

import numpy as np
import rasterio
from scipy import ndimage
from scipy.signal import wiener
from skimage import filters, exposure, restoration, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
from pathlib import Path


class SatelliteImageEnhancer:
    """Advanced image enhancement for satellite imagery."""
    
    def __init__(self):
        self.enhancement_methods = {
            'histogram_equalization': self.histogram_equalization,
            'adaptive_histogram': self.adaptive_histogram_equalization,
            'contrast_stretching': self.contrast_stretching,
            'gamma_correction': self.gamma_correction,
            'unsharp_masking': self.unsharp_masking,
            'noise_reduction': self.noise_reduction,
            'edge_enhancement': self.edge_enhancement,
            'anomaly_highlighting': self.anomaly_highlighting,
            'multi_scale_enhancement': self.multi_scale_enhancement
        }
    
    def histogram_equalization(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply histogram equalization to enhance image contrast.
        
        Parameters:
        - image: input image array
        - clip_limit: clipping limit for adaptive histogram
        - tile_grid_size: grid size for adaptive histogram
        
        Returns:
        - enhanced_image: enhanced image array
        """
        # Convert to float for processing
        img_float = img_as_float(image)
        
        # Ensure image is in valid range for scikit-image
        if img_float.min() < -1 or img_float.max() > 1:
            # Normalize to [-1, 1] range
            img_float = np.clip(img_float, -1, 1)
        
        # Apply adaptive histogram equalization
        enhanced = exposure.equalize_adapthist(
            img_float, clip_limit=clip_limit, nbins=256
        )
        
        return enhanced
    
    def adaptive_histogram_equalization(self, image, clip_limit=3.0, tile_grid_size=(8, 8)):
        """
        Apply adaptive histogram equalization for better local contrast.
        
        Parameters:
        - image: input image array
        - clip_limit: clipping limit
        - tile_grid_size: grid size for local processing
        
        Returns:
        - enhanced_image: enhanced image array
        """
        img_float = img_as_float(image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced = exposure.equalize_adapthist(
            img_float, clip_limit=clip_limit, nbins=256
        )
        
        return enhanced
    
    def contrast_stretching(self, image, percentiles=(2, 98)):
        """
        Apply contrast stretching to improve image visibility.
        
        Parameters:
        - image: input image array
        - percentiles: lower and upper percentiles for stretching
        
        Returns:
        - enhanced_image: contrast-stretched image
        """
        img_float = img_as_float(image)
        
        # Calculate percentiles
        p2, p98 = np.percentile(img_float, percentiles)
        
        # Apply contrast stretching
        enhanced = exposure.rescale_intensity(img_float, in_range=(p2, p98))
        
        return enhanced
    
    def gamma_correction(self, image, gamma=1.2):
        """
        Apply gamma correction to adjust image brightness.
        
        Parameters:
        - image: input image array
        - gamma: gamma value (gamma > 1 darkens, gamma < 1 brightens)
        
        Returns:
        - enhanced_image: gamma-corrected image
        """
        img_float = img_as_float(image)
        
        # Apply gamma correction
        enhanced = exposure.adjust_gamma(img_float, gamma=gamma)
        
        return enhanced
    
    def unsharp_masking(self, image, radius=1, amount=1.0, threshold=0):
        """
        Apply unsharp masking to enhance image details.
        
        Parameters:
        - image: input image array
        - radius: radius of Gaussian blur
        - amount: strength of enhancement
        - threshold: minimum change threshold
        
        Returns:
        - enhanced_image: sharpened image
        """
        img_float = img_as_float(image)
        
        # Apply unsharp masking
        enhanced = filters.unsharp_mask(
            img_float, radius=radius, amount=amount, threshold=threshold
        )
        
        return enhanced
    
    def noise_reduction(self, image, method='gaussian', **kwargs):
        """
        Reduce noise in satellite imagery.
        
        Parameters:
        - image: input image array
        - method: noise reduction method ('gaussian', 'median', 'wiener', 'bilateral')
        - **kwargs: additional parameters for specific methods
        
        Returns:
        - enhanced_image: denoised image
        """
        img_float = img_as_float(image)
        
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            enhanced = ndimage.gaussian_filter(img_float, sigma=sigma)
            
        elif method == 'median':
            size = kwargs.get('size', 3)
            enhanced = ndimage.median_filter(img_float, size=size)
            
        elif method == 'wiener':
            enhanced = wiener(img_float)
            
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.1)
            sigma_space = kwargs.get('sigma_space', 5)
            enhanced = restoration.denoise_bilateral(
                img_float, sigma_color=sigma_color, sigma_spatial=sigma_space
            )
            
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
        
        return enhanced
    
    def edge_enhancement(self, image, method='sobel', **kwargs):
        """
        Enhance edges in satellite imagery.
        
        Parameters:
        - image: input image array
        - method: edge enhancement method ('sobel', 'canny', 'laplacian')
        - **kwargs: additional parameters for specific methods
        
        Returns:
        - enhanced_image: edge-enhanced image
        """
        img_float = img_as_float(image)
        
        if method == 'sobel':
            # Apply Sobel edge detection
            edge_x = filters.sobel_h(img_float)
            edge_y = filters.sobel_v(img_float)
            edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)
            
            # Enhance edges
            enhanced = img_float + kwargs.get('edge_weight', 0.3) * edge_magnitude
            
        elif method == 'canny':
            # Apply Canny edge detection
            edges = filters.canny(img_float, **kwargs)
            enhanced = img_float + kwargs.get('edge_weight', 0.2) * edges
            
        elif method == 'laplacian':
            # Apply Laplacian edge enhancement
            laplacian = filters.laplace(img_float)
            enhanced = img_float - kwargs.get('edge_weight', 0.1) * laplacian
            
        else:
            raise ValueError(f"Unknown edge enhancement method: {method}")
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    def anomaly_highlighting(self, image, method='statistical', **kwargs):
        """
        Highlight potential anomalies in satellite imagery.
        
        Parameters:
        - image: input image array
        - method: anomaly highlighting method
        - **kwargs: additional parameters
        
        Returns:
        - enhanced_image: image with highlighted anomalies
        """
        img_float = img_as_float(image)
        
        if method == 'statistical':
            # Highlight statistical outliers
            mean_val = np.nanmean(img_float)
            std_val = np.nanstd(img_float)
            
            # Calculate z-scores
            z_scores = np.abs((img_float - mean_val) / (std_val + 1e-8))
            
            # Create anomaly mask
            threshold = kwargs.get('threshold', 2.0)
            anomaly_mask = z_scores > threshold
            
            # Highlight anomalies
            enhanced = img_float.copy()
            enhanced[anomaly_mask] = np.clip(enhanced[anomaly_mask] * 1.5, 0, 1)
            
        elif method == 'local_contrast':
            # Highlight local contrast anomalies
            local_mean = ndimage.uniform_filter(img_float, size=kwargs.get('window_size', 5))
            local_contrast = np.abs(img_float - local_mean)
            
            # Normalize and enhance
            contrast_norm = local_contrast / (np.nanmax(local_contrast) + 1e-8)
            enhanced = img_float + kwargs.get('contrast_weight', 0.3) * contrast_norm
            
        else:
            raise ValueError(f"Unknown anomaly highlighting method: {method}")
        
        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    def multi_scale_enhancement(self, image, scales=[1, 2, 4], weights=[0.5, 0.3, 0.2]):
        """
        Apply multi-scale enhancement for comprehensive image improvement.
        
        Parameters:
        - image: input image array
        - scales: list of scale factors
        - weights: weights for each scale
        
        Returns:
        - enhanced_image: multi-scale enhanced image
        """
        img_float = img_as_float(image)
        enhanced = np.zeros_like(img_float)
        
        for scale, weight in zip(scales, weights):
            if scale == 1:
                # Original scale
                scale_img = img_float
            else:
                # Resample to different scale
                scale_img = ndimage.zoom(img_float, 1/scale, order=1)
                scale_img = ndimage.zoom(scale_img, scale, order=1)
            
            # Apply enhancement to this scale
            scale_enhanced = self.adaptive_histogram_equalization(scale_img)
            scale_enhanced = self.unsharp_masking(scale_enhanced)
            
            # Add weighted contribution
            enhanced += weight * scale_enhanced
        
        # Normalize
        enhanced = enhanced / np.sum(weights)
        
        return enhanced
    
    def enhance_for_anomaly_detection(self, image, enhancement_pipeline=None):
        """
        Apply a comprehensive enhancement pipeline optimized for anomaly detection.
        
        Parameters:
        - image: input image array
        - enhancement_pipeline: list of enhancement methods to apply
        
        Returns:
        - enhanced_image: enhanced image optimized for anomaly detection
        """
        if enhancement_pipeline is None:
            enhancement_pipeline = [
                'noise_reduction',
                'contrast_stretching',
                'adaptive_histogram',
                'edge_enhancement',
                'anomaly_highlighting'
            ]
        
        enhanced = img_as_float(image)
        
        for method in enhancement_pipeline:
            if method in self.enhancement_methods:
                enhanced = self.enhancement_methods[method](enhanced)
            else:
                print(f"Warning: Unknown enhancement method: {method}")
        
        return enhanced
    
    def save_enhanced_image(self, output_path, enhanced_image, profile, 
                           description="Enhanced Satellite Image"):
        """
        Save enhanced image as GeoTIFF.
        
        Parameters:
        - output_path: path to save enhanced image
        - enhanced_image: enhanced image array
        - profile: rasterio profile for georeference
        - description: description of the enhancement
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update profile for enhanced image
        enhanced_profile = profile.copy()
        enhanced_profile.update({
            'count': 1,
            'dtype': 'float32',
            'compress': 'deflate',
            'nodata': np.nan
        })
        
        # Save enhanced image
        with rasterio.open(str(output_path), 'w', **enhanced_profile) as dst:
            dst.write(enhanced_image.astype('float32'), 1)
        
        print(f"Enhanced image saved to {output_path}")
        print(f"Enhancement: {description}")


def enhance_satellite_image(geotiff_path, output_path, enhancement_methods=None, **kwargs):
    """
    Convenience function to enhance a satellite image GeoTIFF.
    
    Parameters:
    - geotiff_path: path to input GeoTIFF
    - output_path: path to save enhanced image
    - enhancement_methods: list of enhancement methods to apply
    - **kwargs: additional parameters for enhancement methods
    
    Returns:
    - enhanced_image: enhanced image array
    """
    # Load GeoTIFF
    with rasterio.open(geotiff_path) as src:
        image = src.read(1)
        profile = src.profile
    
    # Initialize enhancer
    enhancer = SatelliteImageEnhancer()
    
    # Apply enhancement pipeline
    if enhancement_methods is None:
        enhanced = enhancer.enhance_for_anomaly_detection(image)
    else:
        enhanced = image
        for method in enhancement_methods:
            if method in enhancer.enhancement_methods:
                enhanced = enhancer.enhancement_methods[method](enhanced, **kwargs)
            else:
                print(f"Warning: Unknown enhancement method: {method}")
    
    # Save enhanced image
    enhancer.save_enhanced_image(output_path, enhanced, profile)
    
    return enhanced


def create_enhancement_comparison(input_path, output_dir, enhancement_methods=None):
    """
    Create a comparison of different enhancement methods.
    
    Parameters:
    - input_path: path to input GeoTIFF
    - output_dir: directory to save comparison results
    - enhancement_methods: list of enhancement methods to compare
    """
    if enhancement_methods is None:
        enhancement_methods = [
            'histogram_equalization',
            'adaptive_histogram',
            'contrast_stretching',
            'unsharp_masking',
            'edge_enhancement',
            'anomaly_highlighting'
        ]
    
    # Load original image
    with rasterio.open(input_path) as src:
        original = src.read(1)
        profile = src.profile
    
    # Initialize enhancer
    enhancer = SatelliteImageEnhancer()
    
    # Create comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Enhanced versions
    for i, method in enumerate(enhancement_methods[:6]):
        if method in enhancer.enhancement_methods:
            enhanced = enhancer.enhancement_methods[method](original)
            axes[i+1].imshow(enhanced, cmap='viridis')
            axes[i+1].set_title(method.replace('_', ' ').title())
            axes[i+1].axis('off')
            
            # Save individual enhanced image
            output_path = Path(output_dir) / f"enhanced_{method}.tif"
            enhancer.save_enhanced_image(output_path, enhanced, profile, 
                                       description=f"Enhanced using {method}")
    
    # Remove extra subplot if needed
    if len(enhancement_methods) < 6:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = Path(output_dir) / "enhancement_comparison.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhancement comparison saved to {comparison_path}")


if __name__ == '__main__':
    print("Satellite Image Enhancement Module")
    print("Use enhance_satellite_image() function for quick enhancement")
    print("Or create SatelliteImageEnhancer instance for advanced usage")
