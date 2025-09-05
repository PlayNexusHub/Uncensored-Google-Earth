"""
PlayNexus Satellite Toolkit - Error Handling & Validation Module
Provides comprehensive error handling, input validation, and graceful degradation
for production-ready satellite imagery analysis.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError, RasterioError


class PlayNexusError(Exception):
    """Base exception class for PlayNexus Satellite Toolkit."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = None  # Will be set by logger


class ValidationError(PlayNexusError):
    """Raised when input validation fails."""
    pass


class ProcessingError(PlayNexusError):
    """Raised when image processing fails."""
    pass


class DataError(PlayNexusError):
    """Raised when satellite data is invalid or corrupted."""
    pass


class PlayNexusLogger:
    """Centralized logging for PlayNexus Satellite Toolkit."""
    
    def __init__(self, name: str = "PlayNexus-Satellite", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler for persistent logging
        log_dir = Path.home() / ".playnexus" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "satellite_toolkit.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error message with optional exception details."""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['traceback'] = traceback.format_exc()
        
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, error: Exception = None, **kwargs):
        """Log critical error message."""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['traceback'] = traceback.format_exc()
        
        self.logger.critical(message, extra=kwargs)


class InputValidator:
    """Validates inputs for satellite imagery processing."""
    
    @staticmethod
    def validate_geotiff_path(file_path: Union[str, Path]) -> Path:
        """Validate GeoTIFF file path and existence."""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ValidationError(f"File does not exist: {file_path}")
            
            if not path.is_file():
                raise ValidationError(f"Path is not a file: {file_path}")
            
            if not path.suffix.lower() in ['.tif', '.tiff']:
                raise ValidationError(f"File is not a GeoTIFF: {file_path}")
            
            return path
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(f"Invalid file path: {file_path}", details={'original_error': str(e)})
            raise
    
    @staticmethod
    def validate_bounding_box(bbox: Union[List, Tuple], expected_length: int = 4) -> Tuple[float, ...]:
        """Validate bounding box coordinates."""
        try:
            if not isinstance(bbox, (list, tuple)):
                raise ValidationError("Bounding box must be a list or tuple")
            
            if len(bbox) != expected_length:
                raise ValidationError(f"Bounding box must have {expected_length} coordinates")
            
            # Convert to floats and validate
            coords = tuple(float(x) for x in bbox)
            
            # Basic coordinate validation
            if expected_length == 4:  # minx, miny, maxx, maxy
                minx, miny, maxx, maxy = coords
                if minx >= maxx or miny >= maxy:
                    raise ValidationError("Invalid bounding box: min coordinates must be less than max coordinates")
                
                if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
                    raise ValidationError("Longitude coordinates must be between -180 and 180")
                
                if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
                    raise ValidationError("Latitude coordinates must be between -90 and 90")
            
            return coords
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(f"Invalid bounding box: {bbox}", details={'original_error': str(e)})
            raise
    
    @staticmethod
    def validate_date_string(date_str: str) -> str:
        """Validate date string format (YYYY-MM-DD)."""
        try:
            from datetime import datetime
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError as e:
            raise ValidationError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD", details={'original_error': str(e)})
    
    @staticmethod
    def validate_numpy_array(array: np.ndarray, expected_dtype: str = None, min_dimensions: int = 2) -> np.ndarray:
        """Validate numpy array for processing."""
        try:
            if not isinstance(array, np.ndarray):
                raise ValidationError("Input must be a numpy array")
            
            if array.size == 0:
                raise ValidationError("Array cannot be empty")
            
            if array.ndim < min_dimensions:
                raise ValidationError(f"Array must have at least {min_dimensions} dimensions")
            
            if expected_dtype and array.dtype != expected_dtype:
                # Try to convert if possible
                try:
                    array = array.astype(expected_dtype)
                except (ValueError, TypeError):
                    raise ValidationError(f"Array cannot be converted to {expected_dtype}")
            
            return array
        except Exception as e:
            if not isinstance(e, ValidationError):
                raise ValidationError(f"Invalid array: {type(array)}", details={'original_error': str(e)})
            raise


class ErrorBoundary:
    """Context manager for graceful error handling."""
    
    def __init__(self, logger: PlayNexusLogger, operation_name: str, fallback_value: Any = None):
        self.logger = logger
        self.operation_name = operation_name
        self.fallback_value = fallback_value
        self.error_occurred = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.logger.error(
                f"Error in {self.operation_name}",
                error=exc_val,
                operation=self.operation_name
            )
            return False  # Don't suppress the exception
        return True
    
    def execute(self, func, *args, **kwargs):
        """Execute function with error boundary."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_occurred = True
            self.logger.error(
                f"Error executing {func.__name__} in {self.operation_name}",
                error=e,
                operation=self.operation_name,
                function=func.__name__
            )
            if self.fallback_value is not None:
                return self.fallback_value
            raise


def safe_geotiff_operation(operation_name: str, fallback_value: Any = None):
    """Decorator for safe GeoTIFF operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = PlayNexusLogger()
            with ErrorBoundary(logger, operation_name, fallback_value) as boundary:
                return boundary.execute(func, *args, **kwargs)
        return wrapper
    return decorator


# Global logger instance
logger = PlayNexusLogger()


def handle_runtime_error(error: Exception, context: str = "Unknown operation") -> Dict[str, Any]:
    """Handle runtime errors and return structured error information."""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'context': context,
        'timestamp': None,  # Will be set by logger
        'suggested_action': 'Check input data and try again'
    }
    
    # Log the error
    logger.error(f"Runtime error in {context}", error=error)
    
    # Provide specific suggestions based on error type
    if isinstance(error, RasterioIOError):
        error_info['suggested_action'] = 'Verify file path and ensure GeoTIFF is not corrupted'
    elif isinstance(error, ValidationError):
        error_info['suggested_action'] = 'Check input parameters and data format'
    elif isinstance(error, MemoryError):
        error_info['suggested_action'] = 'Reduce image size or close other applications'
    elif isinstance(error, ValueError):
        error_info['suggested_action'] = 'Verify input data contains valid numerical values'
    
    return error_info


def validate_environment() -> Dict[str, Any]:
    """Validate the runtime environment for satellite processing."""
    validation_results = {
        'python_version': sys.version,
        'numpy_version': None,
        'rasterio_version': None,
        'scipy_version': None,
        'skimage_version': None,
        'matplotlib_backend': None,
        'platform': sys.platform,
        'all_valid': True,
        'warnings': []
    }
    
    try:
        import numpy as np
        validation_results['numpy_version'] = np.__version__
    except ImportError:
        validation_results['all_valid'] = False
        validation_results['warnings'].append("NumPy not available")
    
    try:
        import rasterio
        validation_results['rasterio_version'] = rasterio.__version__
    except ImportError:
        validation_results['all_valid'] = False
        validation_results['warnings'].append("Rasterio not available")
    
    try:
        import scipy
        validation_results['scipy_version'] = scipy.__version__
    except ImportError:
        validation_results['warnings'].append("SciPy not available")
    
    try:
        import skimage
        validation_results['skimage_version'] = skimage.__version__
    except ImportError:
        validation_results['warnings'].append("scikit-image not available")
    
    try:
        import matplotlib
        validation_results['matplotlib_backend'] = matplotlib.get_backend()
    except ImportError:
        validation_results['warnings'].append("Matplotlib not available")
    
    return validation_results


if __name__ == '__main__':
    # Test the error handling system
    logger.info("Testing PlayNexus error handling system")
    
    # Test validation
    try:
        InputValidator.validate_bounding_box([-122.6, 37.6, -122.3, 37.9])
        logger.info("Bounding box validation working")
    except Exception as e:
        logger.error("Bounding box validation failed", error=e)
    
    # Test environment validation
    env_status = validate_environment()
    logger.info(f"Environment validation: {'PASS' if env_status['all_valid'] else 'FAIL'}")
    if env_status['warnings']:
        for warning in env_status['warnings']:
            logger.warning(warning)
