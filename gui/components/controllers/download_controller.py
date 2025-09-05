"""
Download Controller for PlayNexus Satellite Toolkit
Handles satellite data download operations.
"""
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import sys
import asyncio
import threading
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gui.components.controllers.base_controller import BaseController, ControllerConfig
from gui.components.models.base_model import BaseModel
from scripts.advanced_data_sources import AdvancedDataSources
from scripts.progress_tracker import ProgressTracker
from scripts.validation import validate_download_request, ValidationError
from scripts.security import SecurityValidator

class DownloadRequest(BaseModel):
    """Model for download request data."""
    def __init__(self, **kwargs):
        self.collection = kwargs.get('collection', 'sentinel-2-l2a')
        self.bbox = kwargs.get('bbox', [])
        self.start_date = kwargs.get('start_date', '')
        self.end_date = kwargs.get('end_date', '')
        self.output_dir = kwargs.get('output_dir', '')
        self.max_cloud_cover = kwargs.get('max_cloud_cover', 20)
        self.max_items = kwargs.get('max_items', 10)
        super().__init__(**kwargs)

class DownloadController(BaseController[DownloadRequest]):
    """Controller for satellite data download operations."""
    
    def __init__(self):
        super().__init__(ControllerConfig(
            name="DownloadController",
            enable_logging=True
        ))
        self.data_sources = AdvancedDataSources()
        self.security_validator = SecurityValidator()
        self.current_request: Optional[DownloadRequest] = None
        self.progress_tracker: Optional[ProgressTracker] = None
        self._download_thread: Optional[threading.Thread] = None
        self._is_downloading = False
        
    def validate_download_request(self, request: DownloadRequest) -> bool:
        """
        Validate a download request.
        
        Args:
            request: The download request to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate using existing validation system
            validate_download_request({
                'collection': request.collection,
                'bbox': request.bbox,
                'start_date': request.start_date,
                'end_date': request.end_date,
                'output_dir': request.output_dir,
                'max_cloud_cover': request.max_cloud_cover,
                'max_items': request.max_items
            })
            
            # Additional security validation
            if not self.security_validator.validate_output_path(request.output_dir):
                self.logger.error("Invalid output directory path")
                return False
                
            return True
            
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected validation error: {e}")
            return False
    
    def start_download(
        self, 
        request: DownloadRequest,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        completion_callback: Optional[Callable[[bool, str], None]] = None
    ) -> bool:
        """
        Start a satellite data download.
        
        Args:
            request: The download request
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for completion
            
        Returns:
            True if download started successfully, False otherwise
        """
        if self._is_downloading:
            self.logger.warning("Download already in progress")
            return False
            
        if not self.validate_download_request(request):
            self.logger.error("Invalid download request")
            return False
            
        self.current_request = request
        self._is_downloading = True
        
        # Create progress tracker
        self.progress_tracker = ProgressTracker(
            total_steps=request.max_items,
            description="Downloading satellite data"
        )
        
        # Start download in separate thread
        self._download_thread = threading.Thread(
            target=self._download_worker,
            args=(request, progress_callback, completion_callback),
            daemon=True
        )
        self._download_thread.start()
        
        self.logger.info(f"Started download for collection: {request.collection}")
        return True
    
    def cancel_download(self) -> bool:
        """
        Cancel the current download operation.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        if not self._is_downloading:
            self.logger.warning("No download in progress to cancel")
            return False
            
        self._is_downloading = False
        
        if self.progress_tracker:
            self.progress_tracker.cancel()
            
        self.logger.info("Download cancellation requested")
        return True
    
    def get_download_status(self) -> Dict[str, Any]:
        """
        Get the current download status.
        
        Returns:
            Dictionary containing download status information
        """
        status = {
            'is_downloading': self._is_downloading,
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
    
    def _download_worker(
        self,
        request: DownloadRequest,
        progress_callback: Optional[Callable[[str, float], None]],
        completion_callback: Optional[Callable[[bool, str], None]]
    ) -> None:
        """
        Worker method for downloading satellite data.
        
        Args:
            request: The download request
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback for completion
        """
        success = False
        message = ""
        
        try:
            self.logger.info("Starting download worker")
            
            # Convert bbox to proper format if needed
            bbox = request.bbox
            if isinstance(bbox, list) and len(bbox) == 4:
                bbox = [float(x) for x in bbox]
            
            # Parse dates
            start_date = datetime.fromisoformat(request.start_date.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(request.end_date.replace('Z', '+00:00'))
            
            # Create output directory
            output_path = Path(request.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Search for items
            self.logger.info("Searching for satellite items")
            if progress_callback:
                progress_callback("Searching for items...", 0.1)
                
            items = self.data_sources.search_stac_items(
                collection=request.collection,
                bbox=bbox,
                datetime_range=(start_date, end_date),
                max_cloud_cover=request.max_cloud_cover,
                limit=request.max_items
            )
            
            if not items:
                message = "No items found matching the criteria"
                self.logger.warning(message)
                return
                
            self.logger.info(f"Found {len(items)} items to download")
            
            # Download items
            downloaded_count = 0
            for i, item in enumerate(items):
                if not self._is_downloading:
                    message = "Download cancelled by user"
                    self.logger.info(message)
                    return
                    
                try:
                    # Update progress
                    progress = (i + 1) / len(items)
                    if progress_callback:
                        progress_callback(f"Downloading item {i+1}/{len(items)}", progress)
                    
                    if self.progress_tracker:
                        self.progress_tracker.update_step(
                            i + 1,
                            f"Downloading {item.id}"
                        )
                    
                    # Download the item
                    self.data_sources.download_stac_item(item, output_path)
                    downloaded_count += 1
                    
                    self.logger.info(f"Downloaded item: {item.id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to download item {item.id}: {e}")
                    continue
            
            success = downloaded_count > 0
            message = f"Successfully downloaded {downloaded_count}/{len(items)} items"
            self.logger.info(message)
            
        except Exception as e:
            message = f"Download failed: {str(e)}"
            self.logger.error(message)
            
        finally:
            self._is_downloading = False
            if completion_callback:
                completion_callback(success, message)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self._is_downloading:
            self.cancel_download()
            
        if self._download_thread and self._download_thread.is_alive():
            self._download_thread.join(timeout=5.0)
            
        super().cleanup()
