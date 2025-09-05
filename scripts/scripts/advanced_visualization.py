"""
Advanced Visualization Module for PlayNexus Satellite Toolkit
Provides 3D terrain visualization, interactive dashboards, and enhanced plotting capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import folium
from folium import plugins
import rasterio
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import box, Point
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import logging
from datetime import datetime

from .error_handling import PlayNexusLogger, ValidationError, ProcessingError
from .config import ConfigManager

logger = PlayNexusLogger(__name__)

class AdvancedVisualizer:
    """Advanced visualization capabilities for satellite imagery and analysis results."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the advanced visualizer."""
        self.config = config or ConfigManager()
        self.logger = PlayNexusLogger(__name__)
        self._setup_visualization_config()
        self._setup_colormaps()
    
    def _setup_visualization_config(self):
        """Setup visualization configuration."""
        self.default_figsize = (12, 8)
        self.dpi = 300
        self.style = 'default'
        
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except:
            plt.style.use('default')
        
        # Set seaborn style
        sns.set_theme(style="whitegrid")
    
    def _setup_colormaps(self):
        """Setup custom colormaps for satellite imagery."""
        # NDVI colormap (green to brown)
        self.ndvi_cmap = LinearSegmentedColormap.from_list(
            'ndvi_custom',
            ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
        )
        
        # NDWI colormap (blue to red)
        self.ndwi_cmap = LinearSegmentedColormap.from_list(
            'ndwi_custom',
            ['darkred', 'red', 'orange', 'yellow', 'lightblue', 'blue', 'darkblue']
        )
        
        # Change detection colormap
        self.change_cmap = LinearSegmentedColormap.from_list(
            'change_custom',
            ['darkred', 'red', 'orange', 'yellow', 'white', 'lightgreen', 'green', 'darkgreen']
        )
        
        # Anomaly colormap
        self.anomaly_cmap = LinearSegmentedColormap.from_list(
            'anomaly_custom',
            ['darkblue', 'blue', 'lightblue', 'white', 'yellow', 'orange', 'red', 'darkred']
        )
    
    def create_3d_terrain(
        self,
        elevation_data: np.ndarray,
        image_data: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        dpi: int = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create 3D terrain visualization with optional image overlay."""
        figsize = figsize or self.default_figsize
        dpi = dpi or self.dpi
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids
        height, width = elevation_data.shape
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        # Plot 3D surface
        if image_data is not None:
            # Use image data for color mapping
            surf = ax.plot_surface(
                X, Y, elevation_data,
                facecolors=plt.cm.viridis(image_data),
                alpha=0.8,
                linewidth=0,
                antialiased=True
            )
        else:
            # Use elevation data for color mapping
            surf = ax.plot_surface(
                X, Y, elevation_data,
                cmap='terrain',
                alpha=0.8,
                linewidth=0,
                antialiased=True
            )
        
        # Customize plot
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_zlabel('Elevation')
        ax.set_title('3D Terrain Visualization')
        
        # Add colorbar
        if image_data is not None:
            norm = mcolors.Normalize(vmin=image_data.min(), vmax=image_data.max())
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Image Value')
        else:
            cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Elevation')
        
        # Adjust view
        ax.view_init(elev=30, azim=45)
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_interactive_dashboard(
        self,
        data_dict: Dict[str, np.ndarray],
        titles: Optional[Dict[str, str]] = None,
        colormaps: Optional[Dict[str, str]] = None,
        output_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> go.Figure:
        """Create an interactive Plotly dashboard for multiple datasets."""
        # Setup defaults
        titles = titles or {key: key.replace('_', ' ').title() for key in data_dict.keys()}
        colormaps = colormaps or {key: 'viridis' for key in data_dict.keys()}
        
        # Calculate subplot layout
        n_plots = len(data_dict)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=list(titles.values()),
            specs=[[{"type": "heatmap"} for _ in range(n_cols)] for _ in range(n_rows)]
        )
        
        # Add each dataset as a heatmap
        for idx, (key, data) in enumerate(data_dict.items()):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            # Create heatmap
            heatmap = go.Heatmap(
                z=data,
                colorscale=colormaps[key],
                name=titles[key],
                showscale=True,
                colorbar=dict(title=titles[key])
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title="Interactive Satellite Data Dashboard",
            height=300 * n_rows,
            width=400 * n_cols,
            showlegend=False
        )
        
        # Update axes
        for i in range(1, n_rows + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text="X", row=i, col=j)
                fig.update_yaxes(title_text="Y", row=i, col=j)
        
        if output_path:
            # Save as HTML for interactive viewing
            pyo.plot(fig, filename=str(output_path), auto_open=False)
        
        if show_plot:
            fig.show()
        
        return fig
    
    def create_time_series_plot(
        self,
        time_series_data: np.ndarray,
        dates: List[datetime],
        pixel_coords: Optional[List[Tuple[int, int]]] = None,
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create time series plot for selected pixels or regions."""
        figsize = figsize or self.default_figsize
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if pixel_coords is None:
            # Plot mean time series for entire image
            mean_series = np.nanmean(time_series_data, axis=(1, 2))
            std_series = np.nanstd(time_series_data, axis=(1, 2))
            
            ax.plot(dates, mean_series, 'b-', linewidth=2, label='Mean')
            ax.fill_between(
                dates,
                mean_series - std_series,
                mean_series + std_series,
                alpha=0.3,
                label='±1 Std Dev'
            )
        else:
            # Plot time series for specific pixels
            for i, (row, col) in enumerate(pixel_coords):
                pixel_series = time_series_data[:, row, col]
                ax.plot(dates, pixel_series, '-', linewidth=2, label=f'Pixel ({row}, {col})')
        
        # Customize plot
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Time Series Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_change_detection_visualization(
        self,
        change_maps: List[np.ndarray],
        dates: List[datetime],
        method: str = 'composite',
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create comprehensive change detection visualization."""
        figsize = figsize or (16, 12)
        
        n_changes = len(change_maps)
        n_cols = min(3, n_changes)
        n_rows = (n_changes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each change map
        for idx, (change_map, date) in enumerate(zip(change_maps, dates[1:])):  # Skip first date (reference)
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col]
            
            # Create change visualization
            if method == 'composite':
                # RGB composite: Red for negative changes, Green for positive, Blue for no change
                rgb = np.zeros((*change_map.shape, 3))
                rgb[change_map < 0, 0] = np.abs(change_map[change_map < 0])  # Red for negative
                rgb[change_map > 0, 1] = change_map[change_map > 0]  # Green for positive
                rgb[np.abs(change_map) < 0.1, 2] = 0.5  # Blue for no change
                
                # Normalize
                rgb = np.clip(rgb, 0, 1)
                ax.imshow(rgb)
                title = f'Change: {dates[0].strftime("%Y-%m-%d")} to {date.strftime("%Y-%m-%d")}'
                
            else:
                # Standard heatmap
                im = ax.imshow(change_map, cmap=self.change_cmap, vmin=-1, vmax=1)
                title = f'Change Map: {date.strftime("%Y-%m-%d")}'
                
                # Add colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_changes, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Change Detection Analysis', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_anomaly_visualization(
        self,
        anomaly_data: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create comprehensive anomaly visualization."""
        figsize = figsize or (16, 12)
        
        # Determine what data we have
        available_keys = [key for key, data in anomaly_data.items() if isinstance(data, np.ndarray)]
        
        if not available_keys:
            raise ValidationError("No valid data found in anomaly_data")
        
        n_plots = len(available_keys)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each anomaly dataset
        for idx, key in enumerate(available_keys):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col]
            data = anomaly_data[key]
            
            if key == 'anomalies':
                # Binary anomaly map
                im = ax.imshow(data, cmap='Reds', alpha=0.8)
                title = 'Anomaly Locations'
                
            elif key == 'z_scores':
                # Z-score map
                im = ax.imshow(data, cmap=self.anomaly_cmap, vmin=-3, vmax=3)
                title = 'Z-Score Map'
                
            elif key == 'deviations':
                # Deviation map
                im = ax.imshow(data, cmap=self.anomaly_cmap, vmin=-2, vmax=2)
                title = 'Deviation Map'
                
            else:
                # Generic data
                im = ax.imshow(data, cmap='viridis')
                title = key.replace('_', ' ').title()
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        # Hide unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Anomaly Detection Results', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_geospatial_map(
        self,
        image_data: np.ndarray,
        bounds: Tuple[float, float, float, float],
        overlay_data: Optional[Dict[str, np.ndarray]] = None,
        output_path: Optional[Path] = None,
        show_plot: bool = True
    ) -> folium.Map:
        """Create an interactive geospatial map using Folium."""
        # Calculate center coordinates
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add satellite imagery layer
        folium.raster_layers.ImageOverlay(
            name='Satellite Imagery',
            image=image_data,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.8
        ).add_to(m)
        
        # Add overlay data if provided
        if overlay_data:
            for key, data in overlay_data.items():
                if isinstance(data, np.ndarray):
                    # Convert numpy array to image overlay
                    # This is a simplified approach - in production you'd want proper georeferencing
                    folium.raster_layers.ImageOverlay(
                        name=key.replace('_', ' ').title(),
                        image=data,
                        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                        opacity=0.6
                    ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add fullscreen option
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        if output_path:
            m.save(str(output_path))
        
        if show_plot:
            # Note: Folium maps are typically saved as HTML and opened in browser
            self.logger.info(f"Map saved to {output_path}. Open in web browser to view.")
        
        return m
    
    def create_statistical_summary(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create statistical summary plots for datasets."""
        figsize = figsize or (16, 12)
        
        n_datasets = len(data_dict)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Create plots for each dataset
        for idx, (key, data) in enumerate(data_dict.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            ax = axes[row, col]
            
            # Flatten data for statistics
            flat_data = data.flatten()
            flat_data = flat_data[~np.isnan(flat_data)]
            
            if len(flat_data) > 0:
                # Histogram
                ax.hist(flat_data, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{key.replace("_", " ").title()} Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
                # Add statistics text
                mean_val = np.mean(flat_data)
                std_val = np.std(flat_data)
                median_val = np.median(flat_data)
                
                stats_text = f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nMedian: {median_val:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                       horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{key.replace("_", " ").title()}')
        
        # Hide unused subplots
        for idx in range(n_datasets, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle('Statistical Summary of Datasets', fontsize=16, y=0.95)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def create_correlation_matrix(
        self,
        data_dict: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        figsize: Tuple[int, int] = None,
        show_plot: bool = True
    ) -> plt.Figure:
        """Create correlation matrix visualization for multiple datasets."""
        figsize = figsize or (10, 8)
        
        # Prepare data for correlation analysis
        correlation_data = {}
        for key, data in data_dict.items():
            if isinstance(data, np.ndarray):
                # Use mean values for each dataset to create correlation matrix
                if data.ndim > 2:
                    # For 3D data, take mean across first dimension
                    correlation_data[key] = np.nanmean(data, axis=0).flatten()
                else:
                    correlation_data[key] = data.flatten()
        
        if len(correlation_data) < 2:
            raise ValidationError("At least 2 datasets required for correlation analysis")
        
        # Create DataFrame for correlation
        df = pd.DataFrame(correlation_data)
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        if len(df_clean) == 0:
            raise ValidationError("No valid data for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = df_clean.corr()
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Add correlation values as text
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Customize plot
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        ax.set_title('Dataset Correlation Matrix')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def export_visualization_report(
        self,
        visualizations: Dict[str, plt.Figure],
        output_dir: Path,
        report_name: str = "visualization_report"
    ) -> Path:
        """Export multiple visualizations as a comprehensive report."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual figures
        for name, fig in visualizations.items():
            if isinstance(fig, plt.Figure):
                output_path = output_dir / f"{report_name}_{name}.png"
                fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.info(f"Saved {name} to {output_path}")
        
        # Create HTML report if plotly figures are available
        plotly_figures = {k: v for k, v in visualizations.items() if hasattr(v, 'to_html')}
        if plotly_figures:
            html_report = output_dir / f"{report_name}.html"
            self._create_html_report(plotly_figures, html_report)
            self.logger.info(f"Saved HTML report to {html_report}")
        
        return output_dir
    
    def _create_html_report(self, plotly_figures: Dict[str, Any], output_path: Path):
        """Create HTML report for plotly figures."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PlayNexus Satellite Toolkit - Visualization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .figure { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
            </style>
        </head>
        <body>
            <h1>PlayNexus Satellite Toolkit - Visualization Report</h1>
        """
        
        for name, fig in plotly_figures.items():
            html_content += f"""
            <div class="figure">
                <h2>{name.replace('_', ' ').title()}</h2>
                {fig.to_html(full_html=False)}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

# Convenience functions
def visualize_3d_terrain(
    elevation_data: np.ndarray,
    image_data: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Convenience function for 3D terrain visualization."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_3d_terrain(elevation_data, image_data, output_path)

def create_dashboard(
    data_dict: Dict[str, np.ndarray],
    output_path: Optional[Path] = None
) -> go.Figure:
    """Convenience function for interactive dashboard creation."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_interactive_dashboard(data_dict, output_path=output_path)

def plot_time_series(
    time_series_data: np.ndarray,
    dates: List[datetime],
    pixel_coords: Optional[List[Tuple[int, int]]] = None,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """Convenience function for time series plotting."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_time_series_plot(time_series_data, dates, pixel_coords, output_path)
