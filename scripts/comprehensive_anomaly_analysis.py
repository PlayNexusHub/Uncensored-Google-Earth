#!/usr/bin/env python3
"""
Comprehensive Anomaly Analysis for Satellite Imagery
This script combines image enhancement and anomaly detection to provide
thorough analysis of satellite data for identifying unusual patterns.
"""

import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the scripts directory to the Python path
sys.path.append(str(Path(__file__).parent))

from anomaly_detection import SatelliteAnomalyDetector, detect_anomalies_in_geotiff
from image_enhancement import SatelliteImageEnhancer, enhance_satellite_image


class ComprehensiveAnomalyAnalyzer:
    """Comprehensive analysis combining enhancement and anomaly detection."""
    
    def __init__(self):
        self.enhancer = SatelliteImageEnhancer()
        self.detector = SatelliteAnomalyDetector()
        
    def analyze_satellite_image(self, input_path, output_dir, 
                              enhancement_methods=None, 
                              detection_methods=None,
                              create_comparison=True,
                              **kwargs):
        """
        Perform comprehensive analysis of satellite imagery.
        
        Parameters:
        - input_path: path to input GeoTIFF
        - output_dir: directory to save results
        - enhancement_methods: list of enhancement methods
        - detection_methods: list of detection methods
        - create_comparison: whether to create comparison plots
        - **kwargs: additional parameters
        
        Returns:
        - analysis_results: dictionary with all analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔍 Starting comprehensive analysis of: {input_path}")
        print(f"📁 Results will be saved to: {output_dir}")
        
        # Load original image
        with rasterio.open(input_path) as src:
            original = src.read(1)
            profile = src.profile
        
        print(f"📊 Image loaded: {original.shape} pixels, {original.dtype}")
        
        # Step 1: Image Enhancement
        print("\n🖼️ Step 1: Image Enhancement")
        enhanced_images = {}
        
        if enhancement_methods is None:
            enhancement_methods = [
                'noise_reduction',
                'contrast_stretching', 
                'adaptive_histogram',
                'edge_enhancement',
                'anomaly_highlighting'
            ]
        
        for method in enhancement_methods:
            if method in self.enhancer.enhancement_methods:
                print(f"  Applying {method}...")
                try:
                    enhanced = self.enhancer.enhancement_methods[method](original, **kwargs)
                    enhanced_images[method] = enhanced
                    
                    # Save enhanced image
                    output_path = output_dir / f"enhanced_{method}.tif"
                    self.enhancer.save_enhanced_image(
                        output_path, enhanced, profile, 
                        description=f"Enhanced using {method}"
                    )
                except Exception as e:
                    print(f"    Warning: {method} failed - {e}")
        
        # Step 2: Anomaly Detection
        print("\n🚨 Step 2: Anomaly Detection")
        anomaly_results = {}
        
        if detection_methods is None:
            detection_methods = ['statistical', 'spatial', 'spectral']
        
        # Detect anomalies in original image
        print("  Analyzing original image...")
        original_anomalies, original_info = self.detector.ndvi_anomaly_detection(
            original, method='combined', **kwargs
        )
        anomaly_results['original'] = {
            'mask': original_anomalies,
            'info': original_info
        }
        
        # Detect anomalies in enhanced images
        for method, enhanced_img in enhanced_images.items():
            print(f"  Analyzing {method} enhanced image...")
            try:
                anomalies, info = self.detector.ndvi_anomaly_detection(
                    enhanced_img, method='combined', **kwargs
                )
                anomaly_results[method] = {
                    'mask': anomalies,
                    'info': info
                }
            except Exception as e:
                print(f"    Warning: Anomaly detection on {method} failed - {e}")
        
        # Step 3: Save Results
        print("\n💾 Step 3: Saving Results")
        self._save_analysis_results(output_dir, anomaly_results, profile)
        
        # Step 4: Create Comparison Visualizations
        if create_comparison:
            print("\n📊 Step 4: Creating Visualizations")
            self._create_analysis_comparison(
                output_dir, original, enhanced_images, anomaly_results
            )
        
        # Step 5: Generate Summary Report
        print("\n📋 Step 5: Generating Summary Report")
        summary = self._generate_summary_report(anomaly_results)
        self._save_summary_report(output_dir, summary)
        
        print(f"\n✅ Comprehensive analysis complete! Results saved to {output_dir}")
        
        return {
            'enhanced_images': enhanced_images,
            'anomaly_results': anomaly_results,
            'summary': summary
        }
    
    def _save_analysis_results(self, output_dir, anomaly_results, profile):
        """Save all anomaly detection results."""
        for method_name, result in anomaly_results.items():
            if 'mask' in result and 'info' in result:
                method_dir = output_dir / f"anomalies_{method_name}"
                method_dir.mkdir(exist_ok=True)
                
                # Save anomaly mask
                mask_path = method_dir / "anomaly_mask.tif"
                mask_profile = profile.copy()
                mask_profile.update({
                    'count': 1,
                    'dtype': 'uint8',
                    'compress': 'deflate',
                    'nodata': 255
                })
                
                with rasterio.open(str(mask_path), 'w', **mask_profile) as dst:
                    dst.write(result['mask'].astype('uint8'), 1)
                
                # Save anomaly scores if available
                if 'combined' in result['info'] and 'scores' in result['info']['combined']:
                    scores_path = method_dir / "anomaly_scores.tif"
                    scores_profile = profile.copy()
                    scores_profile.update({
                        'count': 1,
                        'dtype': 'float32',
                        'compress': 'deflate',
                        'nodata': np.nan
                    })
                    
                    with rasterio.open(str(scores_path), 'w', **scores_profile) as dst:
                        dst.write(result['info']['combined']['scores'].astype('float32'), 1)
    
    def _create_analysis_comparison(self, output_dir, original, enhanced_images, anomaly_results):
        """Create comprehensive comparison visualizations."""
        # Create figure with subplots
        n_methods = len(enhanced_images) + 1  # +1 for original
        fig, axes = plt.subplots(3, n_methods, figsize=(5*n_methods, 15))
        
        if n_methods == 1:
            axes = axes.reshape(3, 1)
        
        # Row 1: Original and Enhanced Images
        axes[0, 0].imshow(original, cmap='viridis')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        for i, (method, enhanced) in enumerate(enhanced_images.items()):
            axes[0, i+1].imshow(enhanced, cmap='viridis')
            axes[0, i+1].set_title(f'Enhanced: {method.replace("_", " ").title()}')
            axes[0, i+1].axis('off')
        
        # Row 2: Anomaly Masks
        if 'original' in anomaly_results:
            axes[1, 0].imshow(anomaly_results['original']['mask'], cmap='Reds')
            axes[1, 0].set_title('Original Anomalies')
            axes[1, 0].axis('off')
        
        for i, (method, enhanced) in enumerate(enhanced_images.items()):
            if method in anomaly_results:
                axes[1, i+1].imshow(anomaly_results[method]['mask'], cmap='Reds')
                axes[1, i+1].set_title(f'Anomalies: {method.replace("_", " ").title()}')
                axes[1, i+1].axis('off')
        
        # Row 3: Overlay of anomalies on images
        if 'original' in anomaly_results:
            overlay = original.copy()
            overlay[anomaly_results['original']['mask']] = np.nanmax(original) * 1.2
            axes[2, 0].imshow(overlay, cmap='viridis')
            axes[2, 0].set_title('Original + Anomalies')
            axes[2, 0].axis('off')
        
        for i, (method, enhanced) in enumerate(enhanced_images.items()):
            if method in anomaly_results:
                overlay = enhanced.copy()
                overlay[anomaly_results[method]['mask']] = np.nanmax(enhanced) * 1.2
                axes[2, i+1].imshow(overlay, cmap='viridis')
                axes[2, i+1].set_title(f'{method.replace("_", " ").title()} + Anomalies')
                axes[2, i+1].axis('off')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = output_dir / "comprehensive_analysis_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Comparison visualization saved to {comparison_path}")
    
    def _generate_summary_report(self, anomaly_results):
        """Generate a summary report of all findings."""
        summary = {
            'total_analyses': len(anomaly_results),
            'methods_analyzed': list(anomaly_results.keys()),
            'anomaly_counts': {},
            'anomaly_percentages': {},
            'recommendations': []
        }
        
        for method_name, result in anomaly_results.items():
            if 'mask' in result:
                mask = result['mask']
                total_pixels = mask.size
                anomaly_count = np.sum(mask)
                anomaly_percentage = (anomaly_count / total_pixels) * 100
                
                summary['anomaly_counts'][method_name] = int(anomaly_count)
                summary['anomaly_percentages'][method_name] = float(anomaly_percentage)
        
        # Generate recommendations
        if 'original' in summary['anomaly_percentages']:
            orig_percentage = summary['anomaly_percentages']['original']
            
            if orig_percentage > 10:
                summary['recommendations'].append(
                    "High anomaly rate detected. Consider investigating data quality or environmental changes."
                )
            elif orig_percentage > 5:
                summary['recommendations'].append(
                    "Moderate anomaly rate. Review detected anomalies for potential issues."
                )
            else:
                summary['recommendations'].append(
                    "Low anomaly rate. Data appears to be within normal ranges."
                )
        
        # Compare enhancement methods
        enhanced_methods = [m for m in summary['methods_analyzed'] if m != 'original']
        if len(enhanced_methods) > 0:
            best_method = min(enhanced_methods, 
                            key=lambda x: summary['anomaly_percentages'].get(x, float('inf')))
            summary['recommendations'].append(
                f"Best enhancement method: {best_method} "
                f"({summary['anomaly_percentages'].get(best_method, 0):.2f}% anomalies)"
            )
        
        return summary
    
    def _save_summary_report(self, output_dir, summary):
        """Save the summary report to a text file."""
        report_path = output_dir / "analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE SATELLITE IMAGERY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Analyses Performed: {summary['total_analyses']}\n")
            f.write(f"Methods Analyzed: {', '.join(summary['methods_analyzed'])}\n\n")
            
            f.write("ANOMALY DETECTION RESULTS:\n")
            f.write("-" * 30 + "\n")
            for method in summary['methods_analyzed']:
                count = summary['anomaly_counts'].get(method, 0)
                percentage = summary['anomaly_percentages'].get(method, 0)
                f.write(f"{method.replace('_', ' ').title()}: {count} anomalies ({percentage:.2f}%)\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for rec in summary['recommendations']:
                f.write(f"• {rec}\n")
        
        print(f"  Summary report saved to {report_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Satellite Imagery Anomaly Analysis'
    )
    parser.add_argument('input', help='Input GeoTIFF file path')
    parser.add_argument('output', help='Output directory for results')
    parser.add_argument('--enhancement-methods', nargs='+', 
                       default=['noise_reduction', 'contrast_stretching', 'adaptive_histogram'],
                       help='Enhancement methods to apply')
    parser.add_argument('--detection-methods', nargs='+',
                       default=['statistical', 'spatial'],
                       help='Anomaly detection methods to use')
    parser.add_argument('--no-comparison', action='store_true',
                       help='Skip creating comparison visualizations')
    parser.add_argument('--stat-threshold', type=float, default=1.5,
                       help='Statistical anomaly detection threshold')
    parser.add_argument('--spatial-threshold', type=float, default=2.0,
                       help='Spatial anomaly detection threshold')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComprehensiveAnomalyAnalyzer()
    
    # Run comprehensive analysis
    try:
        results = analyzer.analyze_satellite_image(
            input_path=args.input,
            output_dir=args.output,
            enhancement_methods=args.enhancement_methods,
            detection_methods=args.detection_methods,
            create_comparison=not args.no_comparison,
            stat_threshold=args.stat_threshold,
            spatial_threshold=args.spatial_threshold
        )
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📊 Found anomalies in {len(results['anomaly_results'])} analyses")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
