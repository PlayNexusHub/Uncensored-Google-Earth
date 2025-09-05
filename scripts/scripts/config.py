"""
PlayNexus Satellite Toolkit - Configuration Management Module
Provides centralized configuration management with environment-specific settings,
validation, and secure defaults for production deployment.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import platform as platform_module


@dataclass
class ProcessingConfig:
    """Configuration for image processing operations."""
    
    # Image enhancement settings
    histogram_clip_limit: float = 2.0
    histogram_nbins: int = 256
    contrast_percentiles: tuple = (2, 98)
    gamma_correction: float = 1.2
    noise_reduction_sigma: float = 1.0
    edge_enhancement_weight: float = 0.3
    
    # Anomaly detection settings
    statistical_threshold: float = 2.0
    spatial_threshold: float = 2.0
    spectral_contamination: float = 0.1
    neighborhood_size: int = 5
    water_threshold: float = 0.3
    
    # Performance settings
    max_image_size: int = 8192  # Maximum image dimension
    chunk_size: int = 1024      # Processing chunk size
    memory_limit_gb: float = 4.0  # Memory usage limit
    
    # Output settings
    output_format: str = "GeoTIFF"
    compression: str = "deflate"
    dpi: int = 300


@dataclass
class DownloadConfig:
    """Configuration for satellite data download."""
    
    # API settings
    stac_api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/"
    max_cloud_cover: float = 30.0
    max_scenes: int = 10
    download_timeout: int = 300  # seconds
    
    # Data sources
    preferred_collections: List[str] = field(default_factory=lambda: [
        "sentinel-2-l2a",
        "landsat-c2-l2"
    ])
    
    # Storage settings
    cache_directory: Optional[str] = None
    max_cache_size_gb: float = 10.0


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_rotation: str = "daily"  # daily, weekly, monthly
    max_log_files: int = 30
    
    # Progress tracking
    progress_update_interval: float = 0.1  # seconds
    show_progress_bar: bool = True
    
    # Error reporting
    enable_error_reporting: bool = True
    error_reporting_url: Optional[str] = None


@dataclass
class SecurityConfig:
    """Configuration for security and privacy."""
    
    # Data handling
    enable_telemetry: bool = False
    anonymize_data: bool = True
    data_retention_days: int = 30
    
    # API security
    api_key_rotation_days: int = 90
    max_api_requests_per_minute: int = 60
    
    # File security
    validate_file_signatures: bool = True
    scan_for_malware: bool = False  # Placeholder for future implementation


@dataclass
class PlayNexusConfig:
    """Main configuration class for PlayNexus Satellite Toolkit."""
    
    # Branding
    brand_name: str = "PlayNexus"
    product_name: str = "Satellite Imagery Educational Toolkit"
    version: str = "1.0.0"
    owner: str = "Nortaq"
    contact_email: str = "playnexushq@gmail.com"
    
    # Processing configuration
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    # Download configuration
    download: DownloadConfig = field(default_factory=DownloadConfig)
    
    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Security configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Platform-specific settings
    platform: str = platform_module.system().lower()
    is_windows: bool = platform_module.system().lower() == "windows"
    is_macos: bool = platform_module.system().lower() == "darwin"
    is_linux: bool = platform_module.system().lower() == "linux"
    
    def __post_init__(self):
        """Initialize platform-specific settings after object creation."""
        self._setup_platform_specifics()
    
    def _setup_platform_specifics(self):
        """Configure platform-specific settings."""
        if self.is_windows:
            self._setup_windows()
        elif self.is_macos:
            self._setup_macos()
        elif self.is_linux:
            self._setup_linux()
    
    def _setup_windows(self):
        """Configure Windows-specific settings."""
        # Windows paths
        if not self.download.cache_directory:
            self.download.cache_directory = str(Path.home() / "AppData" / "Local" / "PlayNexus" / "Cache")
        
        if not self.logging.log_file:
            self.logging.log_file = str(Path.home() / "AppData" / "Local" / "PlayNexus" / "Logs" / "satellite_toolkit.log")
    
    def _setup_macos(self):
        """Configure macOS-specific settings."""
        # macOS paths
        if not self.download.cache_directory:
            self.download.cache_directory = str(Path.home() / "Library" / "Application Support" / "PlayNexus" / "Cache")
        
        if not self.logging.log_file:
            self.logging.log_file = str(Path.home() / "Library" / "Logs" / "PlayNexus" / "satellite_toolkit.log")
    
    def _setup_linux(self):
        """Configure Linux-specific settings."""
        # Linux paths
        if not self.download.cache_directory:
            self.download.cache_directory = str(Path.home() / ".local" / "share" / "playnexus" / "cache")
        
        if not self.logging.log_file:
            self.logging.log_file = str(Path.home() / ".local" / "share" / "playnexus" / "logs" / "satellite_toolkit.log")


class ConfigManager:
    """Manages configuration loading, validation, and persistence."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_path()
        self.config = PlayNexusConfig()
        self._load_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        if platform_module.system().lower() == "windows":
            config_dir = Path.home() / "AppData" / "Local" / "PlayNexus"
        elif platform_module.system().lower() == "darwin":
            config_dir = Path.home() / "Library" / "Application Support" / "PlayNexus"
        else:
            config_dir = Path.home() / ".config" / "playnexus"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "satellite_toolkit.json")
    
    def _load_config(self):
        """Load configuration from file or create default."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._update_config_from_dict(config_data)
            else:
                self.save_config()  # Save default config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_file}: {e}")
            print("Using default configuration.")
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary data."""
        # Update main config
        for key, value in config_data.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(getattr(self.config, key), '__dict__'):
                    # Update nested config object
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_data = self._config_to_dict()
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save config to {self.config_file}: {e}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for key, value in self.config.__dict__.items():
            if key.startswith('_'):
                continue
            
            if hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, tuple)):
                # Convert nested config objects
                config_dict[key] = {k: v for k, v in value.__dict__.items() if not k.startswith('_')}
            else:
                config_dict[key] = value
        
        return config_dict
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config_obj = self.config
        
        # Navigate to the parent object
        for k in keys[:-1]:
            if not hasattr(config_obj, k):
                raise ValueError(f"Invalid configuration key: {key}")
            config_obj = getattr(config_obj, k)
        
        # Set the value
        setattr(config_obj, keys[-1], value)
    
    def validate(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        issues = []
        
        # Validate processing config
        if self.config.processing.max_image_size <= 0:
            issues.append("max_image_size must be positive")
        
        if self.config.processing.memory_limit_gb <= 0:
            issues.append("memory_limit_gb must be positive")
        
        # Validate download config
        if self.config.download.max_cloud_cover < 0 or self.config.download.max_cloud_cover > 100:
            issues.append("max_cloud_cover must be between 0 and 100")
        
        if self.config.download.max_scenes <= 0:
            issues.append("max_scenes must be positive")
        
        # Validate logging config
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.config.logging.log_level.upper() not in valid_log_levels:
            issues.append(f"log_level must be one of: {', '.join(valid_log_levels)}")
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = PlayNexusConfig()
        self.save_config()
    
    def export_config(self, export_path: str):
        """Export configuration to specified path."""
        config_data = self._config_to_dict()
        with open(export_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    def import_config(self, import_path: str):
        """Import configuration from specified path."""
        with open(import_path, 'r') as f:
            config_data = json.load(f)
        self._update_config_from_dict(config_data)
        self.save_config()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> PlayNexusConfig:
    """Get the global configuration instance."""
    return config_manager.config


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value by key."""
    return config_manager.get(key, default)


def set_config_value(key: str, value: Any):
    """Set a configuration value by key."""
    config_manager.set(key, value)
    config_manager.save_config()


if __name__ == "__main__":
    # Test configuration management
    print("Testing PlayNexus Configuration Management")
    print("=" * 50)
    
    # Load config
    config = get_config()
    print(f"Brand: {config.brand_name}")
    print(f"Product: {config.product_name}")
    print(f"Version: {config.version}")
    print(f"Owner: {config.owner}")
    print(f"Contact: {config.contact_email}")
    print(f"Platform: {config.platform}")
    
    # Test validation
    issues = config_manager.validate()
    if issues:
        print(f"Configuration issues found: {issues}")
    else:
        print("Configuration validation passed")
    
    # Test getting/setting values
    print(f"Current max image size: {get_config_value('processing.max_image_size')}")
    set_config_value('processing.max_image_size', 4096)
    print(f"Updated max image size: {get_config_value('processing.max_image_size')}")
    
    # Reset to defaults
    config_manager.reset_to_defaults()
    print("Configuration reset to defaults")
    
    print("Configuration management test completed!")
