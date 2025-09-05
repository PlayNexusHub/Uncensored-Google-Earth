"""
Enhancement Controller for PlayNexus Satellite Toolkit
Handles image enhancement operations.
"""
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path
import sys
import threading
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gui.components.controllers.base_controller import BaseController, ControllerConfig
from gui.components.models.base_model import BaseModel
from scripts.advanced_processing import AdvancedImageProcessor
from scripts.progress_tracker import ProgressTracker
from scripts.validation import validate_processing_request, ValidationError
from scripts.security import SecurityValidator

class EnhancementMethod(Enum):
    """Available enhancement methods."""
    NOISE_REDUCTION = "noise_reduction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    EDGE_ENHANCEMENT = "edge_enhancement"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    ADAPTIVE_ENHANCEMENT = "adaptive_enhancement"

class EnhancementRequest(BaseModel):
    """Model for enhancement request data."""
    def __init__(self, **kwargs):
        self.input_file = kwargs.get('input_file', '')
        self.output_dir = kwargs.get('output_dir', '')
        self.methods = kwargs.get('methods', [])
        self.noise_reduction_strength = kwargs.get('noise_reduction_strength', 0.5)
        self.contrast_factor = kwargs.get('contrast_factor', 1.2)
        self.edge_threshold = kwargs.get('edge_threshold', 0.1)
        self.preserve_original = kwargs.get('preserve_original', True)
        super().__init__(**kwargs)

class EnhancementController(BaseController[EnhancementRequest]):
    """Controller for image enhancement operations."""
    
    def __init__(self):
        super().__init__(ControllerConfig(
            name="EnhancementController",
            enable_logging=True
        ))
        self.processor = AdvancedImageProcessor()
        self.security_validator = SecurityValidator()
        self.current_request: Optional[EnhancementRequest] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._is_processing = False
        
    def get_available_methods(self) -> List[Dict[str, str]]:
        """
        Get list of available enhancement methods.
        
        Returns:
            List of method dictionaries with name and description
        """
        return [
            {
                'id': EnhancementMethod.NOISE_REDUCTION.value,
                'name': 'Noise Reduction',
                'description': 'Reduces noise using advanced filtering techniques'
            },
            {
                'id': EnhancementMethod.CONTRAST_ENHANCEMENT.value,
                'name': 'Contrast Enhancement',
                'description': 'Improves image contrast and brightness'
            },
            {
                'id': EnhancementMethod.EDGE_ENHANCEMENT.value,
                'name': 'Edge Enhancement',
                'description': 'Sharpens edges and fine details'
            },
            {
                'id': EnhancementMethod.HISTOGRAM_EQUALIZATION.value,
                'name': 'Histogram Equalization',
                'description': 'Equalizes image histogram for better contrast'
            },
            {
                'id': EnhancementMethod.ADAPTIVE_ENHANCEMENT.value,
                'name': 'Adaptive Enhancement',
                'description': 'Applies adaptive enhancement based on image content'
            }
        ]
    
    def validate_enhancement_request(self, request: EnhancementRequest) -> bool:
        """
        Validate an enhancement request.
        
        Args:
            request: The enhancement request to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if input file exists
            input_path = Path(request.input_file)
            if not input_path.exists():
                self.logger.error(f"Input file does not exist: {request.input_file}")
                return False
            
            # Validate file type
            if not self.security_validator.validate_file_type(
                request.input_file, 
                allowed_extensions=['.tif', '.tiff', '.jpg', '.jpeg', '.png']
            ):
                self.logger.error("Invalid file type for enhancement")
                return False
            
            # Validate output directory
            if not self.security_validator.validate_output_path(request.output_dir):
                self.logger.error("Invalid output directory path")
                return False
            
            # Check if at least one method is selected
            if not request.methods:
                self.logger.error("No enhancement methods selected")
                return False
            
            # Validate method parameters
            if request.noise_reduction_strength < 0 or request.noise_reduction_strength > 1:
                self.logger.error("Invalid noise reduction strength (must be 0-1)")
                return False
            
            if request.contrast_factor < 0.1 or request.contrast_factor > 5.0:
                self.logger.error("Invalid contrast factor (must be 0.1-5.0)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False
    
    def start_enhancement(
        self,
        request: EnhancementRequest,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        completion_callback: Optional[Callable[[bool, str, List[str]], None]] = None
    ) -> bool:
        """
        Start image enhancement processing.
        
        Args:
            request: The enhancement request
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for completion with output files
            
        Returns:
            True if processing started successfully, False otherwise
        """
        if self._is_processing:
            self.logger.warning("Enhancement already in progress")
            return False
            
        if not self.validate_enhancement_request(request):
            self.logger.error("Invalid enhancement request")
            return False
            
        self.current_request = request
        self._is_processing = True
        
        # Create progress tracker
        self.progress_tracker = ProgressTracker(
            total_steps=len(request.methods),
            description="Enhancing image"
        )
        
        # Start processing in separate thread
        self._processing_thread = threading.Thread(
            target=self._enhancement_worker,
            args=(request, progress_callback, completion_callback),
            daemon=True
        )
        self._processing_thread.start()
        
        self.logger.info(f"Started enhancement for: {request.input_file}")
        return True
    
    def cancel_enhancement(self) -> bool:
        """
        Cancel the current enhancement operation.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self._is_processing:
            self.logger.warning("No enhancement in progress to cancel")
            return False
            
        self._is_processing = False
        
        if self.progress_tracker:
            self.progress_tracker.cancel()
            
        self.logger.info("Enhancement cancellation requested")
        return True
    
    def get_enhancement_status(self) -> Dict[str, Any]:
        """
        Get the current enhancement status.
        
        Returns:
            Dictionary containing enhancement status information
        """
        status = {
            'is_processing': self._is_processing,
            'current_request': self.current_request.to_dict() if self.current_request else None,
            'progress': None
        }
        
        if self.progress_tracker:
            status['progress'] = {
                'current_step': self.progress_tracker.current_step,
                'total_steps': self.progress_tracker.total_steps,
                'percentage': self.progress_tracker.get_percentage(),
                'description': self.progress_tracker.description,
                'eta': self.progress_tracker.get_eta()
            }
            
        return status
    
    def _enhancement_worker(
        self,
        request: EnhancementRequest,
        progress_callback: Optional[Callable[[str, float], None]],
        completion_callback: Optional[Callable[[bool, str, List[str]], None]]
    ) -> None:
        """
        Worker method for image enhancement.
        
        Args:
            request: The enhancement request
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for completion
        """
        success = False
        message = ""
        output_files = []
        
        try:
            self.logger.info("Starting enhancement worker")
            
            # Create output directory
            output_path = Path(request.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load input image
            if progress_callback:
                progress_callback("Loading input image...", 0.1)
            
            input_path = Path(request.input_file)
            image_data = self.processor.load_image(str(input_path))
            
            if image_data is None:
                message = "Failed to load input image"
                self.logger.error(message)
                return
            
            self.logger.info(f"Loaded image with shape: {image_data.shape}")
            
            # Apply each enhancement method
            enhanced_image = image_data.copy()
            
            for i, method in enumerate(request.methods):
                if not self._is_processing:
                    message = "Enhancement cancelled by user"
                    self.logger.info(message)
                    return
                
                try:
                    # Update progress
                    progress = (i + 1) / len(request.methods)
                    method_name = method.replace('_', ' ').title()
                    
                    if progress_callback:
                        progress_callback(f"Applying {method_name}...", progress)
                    
                    if self.progress_tracker:
                        self.progress_tracker.update_step(
                            i + 1,
                            f"Applying {method_name}"
                        )
                    
                    # Apply enhancement method
                    if method == EnhancementMethod.NOISE_REDUCTION.value:
                        enhanced_image = self.processor.reduce_noise(
                            enhanced_image,
                            strength=request.noise_reduction_strength
                        )
                    elif method == EnhancementMethod.CONTRAST_ENHANCEMENT.value:
                        enhanced_image = self.processor.enhance_contrast(
                            enhanced_image,
                            factor=request.contrast_factor
                        )
                    elif method == EnhancementMethod.EDGE_ENHANCEMENT.value:
                        enhanced_image = self.processor.enhance_edges(
                            enhanced_image,
                            threshold=request.edge_threshold
                        )
                    elif method == EnhancementMethod.HISTOGRAM_EQUALIZATION.value:
                        enhanced_image = self.processor.equalize_histogram(enhanced_image)
                    elif method == EnhancementMethod.ADAPTIVE_ENHANCEMENT.value:
                        enhanced_image = self.processor.adaptive_enhancement(enhanced_image)
                    
                    self.logger.info(f"Applied {method_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply {method}: {e}")
                    continue
            
            # Save enhanced image
            if progress_callback:
                progress_callback("Saving enhanced image...", 0.9)
            
            # Generate output filename
            base_name = input_path.stem
            output_filename = f"{base_name}_enhanced{input_path.suffix}"
            output_file_path = output_path / output_filename
            
            # Save the enhanced image
            self.processor.save_image(enhanced_image, str(output_file_path))
            output_files.append(str(output_file_path))
            
            # Save original if requested
            if request.preserve_original:
                original_filename = f"{base_name}_original{input_path.suffix}"
                original_file_path = output_path / original_filename
                self.processor.save_image(image_data, str(original_file_path))
                output_files.append(str(original_file_path))
            
            success = True
            message = f"Successfully enhanced image with {len(request.methods)} methods"
            self.logger.info(message)
            
        except Exception as e:
            message = f"Enhancement failed: {str(e)}"
            self.logger.error(message)
            
        finally:
            self._is_processing = False
            if completion_callback:
                completion_callback(success, message, output_files)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._is_processing:
            self.cancel_enhancement()
            
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
            
        super().cleanup()
