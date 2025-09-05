"""
Main Application Controller for PlayNexus Satellite Toolkit
Orchestrates the overall application flow and component interactions.
"""
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys
import tkinter as tk

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gui.components.controllers.base_controller import BaseController, ControllerConfig
from gui.components.controllers.download_controller import DownloadController
from gui.components.controllers.enhancement_controller import EnhancementController
from gui.components.models.base_model import BaseModel
from scripts.config import get_config
from scripts.state_manager import get_state_manager

class ApplicationState(BaseModel):
    """Model for application state."""
    def __init__(self, **kwargs):
        self.current_tab = kwargs.get('current_tab', 'welcome')
        self.theme = kwargs.get('theme', 'light')
        self.window_geometry = kwargs.get('window_geometry', '1200x800')
        self.last_input_dir = kwargs.get('last_input_dir', str(Path.home()))
        self.last_output_dir = kwargs.get('last_output_dir', str(Path.home()))
        super().__init__(**kwargs)

class MainAppController(BaseController[ApplicationState]):
    """Main application controller that orchestrates all components."""
    
    def __init__(self):
        super().__init__(ControllerConfig(
            name="MainAppController",
            enable_logging=True
        ))
        
        # Initialize application state
        self.app_state = ApplicationState()
        
        # Initialize configuration and state management
        self.config = get_config()
        self.state_manager = get_state_manager()
        
        # Initialize sub-controllers
        self.download_controller = DownloadController()
        self.enhancement_controller = EnhancementController()
        
        # Track active operations
        self._active_operations: Dict[str, bool] = {
            'download': False,
            'enhancement': False
        }
        
        self.logger.info("MainAppController initialized")
    
    def get_application_info(self) -> Dict[str, Any]:
        """
        Get application information and status.
        
        Returns:
            Dictionary containing application info
        """
        return {
            'name': self.config.product_name,
            'version': getattr(self.config, 'version', '1.1.0'),
            'description': 'Advanced satellite imagery analysis toolkit',
            'current_tab': self.app_state.current_tab,
            'theme': self.app_state.theme,
            'active_operations': self._active_operations.copy(),
            'controllers': {
                'download': self.download_controller.get_download_status(),
                'enhancement': self.enhancement_controller.get_enhancement_status()
            }
        }
    
    def set_current_tab(self, tab_name: str) -> None:
        """
        Set the current active tab.
        
        Args:
            tab_name: Name of the tab to activate
        """
        self.app_state.current_tab = tab_name
        self.logger.info(f"Switched to tab: {tab_name}")
    
    def set_theme(self, theme_name: str) -> None:
        """
        Set the application theme.
        
        Args:
            theme_name: Name of the theme to apply
        """
        self.app_state.theme = theme_name
        self.logger.info(f"Theme changed to: {theme_name}")
    
    def update_window_geometry(self, geometry: str) -> None:
        """
        Update the window geometry setting.
        
        Args:
            geometry: Window geometry string (e.g., "1200x800+100+100")
        """
        self.app_state.window_geometry = geometry
    
    def get_recent_directories(self) -> Dict[str, str]:
        """
        Get recently used directories.
        
        Returns:
            Dictionary with input and output directory paths
        """
        return {
            'input': self.app_state.last_input_dir,
            'output': self.app_state.last_output_dir
        }
    
    def update_recent_directory(self, dir_type: str, path: str) -> None:
        """
        Update a recently used directory.
        
        Args:
            dir_type: Type of directory ('input' or 'output')
            path: Directory path
        """
        if dir_type == 'input':
            self.app_state.last_input_dir = path
        elif dir_type == 'output':
            self.app_state.last_output_dir = path
        
        self.logger.info(f"Updated {dir_type} directory: {path}")
    
    def start_download_workflow(self, **kwargs) -> bool:
        """
        Start a download workflow.
        
        Args:
            **kwargs: Download parameters
            
        Returns:
            True if workflow started successfully
        """
        if self._active_operations['download']:
            self.logger.warning("Download already in progress")
            return False
        
        self._active_operations['download'] = True
        self.logger.info("Started download workflow")
        return True
    
    def start_enhancement_workflow(self, **kwargs) -> bool:
        """
        Start an enhancement workflow.
        
        Args:
            **kwargs: Enhancement parameters
            
        Returns:
            True if workflow started successfully
        """
        if self._active_operations['enhancement']:
            self.logger.warning("Enhancement already in progress")
            return False
        
        self._active_operations['enhancement'] = True
        self.logger.info("Started enhancement workflow")
        return True
    
    def cancel_all_operations(self) -> None:
        """Cancel all active operations."""
        if self._active_operations['download']:
            self.download_controller.cancel_download()
            self._active_operations['download'] = False
        
        if self._active_operations['enhancement']:
            self.enhancement_controller.cancel_enhancement()
            self._active_operations['enhancement'] = False
        
        self.logger.info("Cancelled all active operations")
    
    def get_workflow_suggestions(self, context: str = '') -> List[Dict[str, str]]:
        """
        Get workflow suggestions based on current context.
        
        Args:
            context: Current context or state
            
        Returns:
            List of workflow suggestions
        """
        suggestions = []
        
        # Basic workflows
        suggestions.extend([
            {
                'id': 'download_enhance',
                'title': 'Download & Enhance',
                'description': 'Download satellite data and apply enhancements',
                'steps': ['Download satellite imagery', 'Apply enhancement filters', 'Save results']
            },
            {
                'id': 'enhance_only',
                'title': 'Enhance Existing Images',
                'description': 'Apply enhancements to existing image files',
                'steps': ['Select input images', 'Configure enhancement methods', 'Process images']
            },
            {
                'id': 'batch_download',
                'title': 'Batch Download',
                'description': 'Download multiple satellite datasets',
                'steps': ['Configure search parameters', 'Review available datasets', 'Download selected items']
            }
        ])
        
        # Context-specific suggestions
        if context == 'no_data':
            suggestions.insert(0, {
                'id': 'get_started',
                'title': 'Get Started',
                'description': 'Download sample satellite data to begin analysis',
                'steps': ['Select area of interest', 'Choose date range', 'Download sample data']
            })
        
        return suggestions
    
    def save_application_state(self) -> bool:
        """
        Save the current application state.
        
        Returns:
            True if state was saved successfully
        """
        try:
            state_data = self.app_state.to_dict()
            self.state_manager.save_state('application', state_data)
            self.logger.info("Application state saved")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save application state: {e}")
            return False
    
    def load_application_state(self) -> bool:
        """
        Load the application state.
        
        Returns:
            True if state was loaded successfully
        """
        try:
            state_data = self.state_manager.load_state('application')
            if state_data:
                # Update application state with loaded data
                for key, value in state_data.items():
                    if hasattr(self.app_state, key):
                        setattr(self.app_state, key, value)
                
                self.logger.info("Application state loaded")
                return True
            else:
                self.logger.info("No saved application state found")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load application state: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary containing system status information
        """
        return {
            'application': {
                'name': self.config.product_name,
                'version': getattr(self.config, 'version', '1.1.0'),
                'state': self.app_state.to_dict()
            },
            'controllers': {
                'download': {
                    'active': self._active_operations['download'],
                    'status': self.download_controller.get_download_status()
                },
                'enhancement': {
                    'active': self._active_operations['enhancement'],
                    'status': self.enhancement_controller.get_enhancement_status()
                }
            },
            'resources': {
                'memory_usage': self._get_memory_usage(),
                'active_threads': self._get_active_threads()
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _get_active_threads(self) -> int:
        """Get number of active threads."""
        try:
            import threading
            return threading.active_count()
        except Exception:
            return -1
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.logger.info("Cleaning up MainAppController")
        
        # Cancel all operations
        self.cancel_all_operations()
        
        # Save application state
        self.save_application_state()
        
        # Clean up sub-controllers
        self.download_controller.cleanup()
        self.enhancement_controller.cleanup()
        
        super().cleanup()
        self.logger.info("MainAppController cleanup completed")
