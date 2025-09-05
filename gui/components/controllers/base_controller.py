"""
Base Controller for PlayNexus Satellite Toolkit
Provides common functionality for all controllers in the application.
"""
from typing import Any, Dict, Optional, TypeVar, Generic
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.error_handling import PlayNexusLogger
from scripts.config import get_config
from scripts.state_manager import get_state_manager

T = TypeVar('T')

@dataclass
class ControllerConfig:
    """Configuration for controller initialization."""
    name: str
    logger_name: Optional[str] = None
    debug: bool = False

class BaseController(Generic[T]):
    """
    Base controller class providing common functionality for all controllers.
    """
    
    def __init__(self, config: ControllerConfig):
        """
        Initialize the base controller.
        
        Args:
            config: Controller configuration
        """
        self.config = get_config()
        self.state = get_state_manager()
        self.name = config.name
        self.debug = config.debug
        
        # Set up logging
        logger_name = config.logger_name or f"controller.{self.name.lower()}"
        self.logger = PlayNexusLogger(logger_name)
        
        # Initialize state
        self._initialized = False
        self._data: Optional[T] = None
        
    def initialize(self) -> bool:
        """
        Initialize the controller and its dependencies.
        
        Returns:
            bool: True if initialization was successful
        """
        if self._initialized:
            return True
            
        try:
            self._setup_dependencies()
            self._initialized = True
            self.logger.info(f"{self.name} controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name} controller: {str(e)}")
            return False
    
    def _setup_dependencies(self) -> None:
        """Set up any required dependencies."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources used by the controller."""
        self._initialized = False
        self._data = None
        
    def get_data(self) -> Optional[T]:
        """
        Get the current data managed by this controller.
        
        Returns:
            The current data or None if not available
        """
        return self._data
    
    def update_data(self, data: T) -> None:
        """
        Update the data managed by this controller.
        
        Args:
            data: The new data to manage
        """
        self._data = data
        self._on_data_updated()
    
    def _on_data_updated(self) -> None:
        """Called when the data is updated."""
        pass
    
    def __del__(self):
        """Ensure resources are cleaned up when the controller is destroyed."""
        self.cleanup()
