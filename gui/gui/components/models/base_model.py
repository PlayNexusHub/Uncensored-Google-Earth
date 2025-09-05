"""
Base Model for PlayNexus Satellite Toolkit
Provides common functionality for all data models.
"""
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import logging
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.error_handling import PlayNexusLogger
from scripts.config import get_config

T = TypeVar('T')

class ModelUpdate:
    """Represents a model update with change information."""
    def __init__(self, model: 'BaseModel', field: str, old_value: Any, new_value: Any):
        self.model = model
        self.field = field
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the update to a dictionary."""
        return {
            'model': self.model.__class__.__name__,
            'field': self.field,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'timestamp': self.timestamp.isoformat()
        }

class BaseModel:
    """
    Base model class providing common functionality for all data models.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the base model with the given attributes.
        
        Args:
            **kwargs: Model attributes as keyword arguments
        """
        self._logger = PlayNexusLogger(f"model.{self.__class__.__name__.lower()}")
        self._listeners: List[Callable[[ModelUpdate], None]] = []
        self._initialized = False
        
        # Set attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
        self._initialized = True
        
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute and notify listeners if the value changed.
        
        Args:
            name: The attribute name
            value: The new value
        """
        old_value = getattr(self, name, None) if hasattr(self, name) else None
        
        # Set the attribute
        super().__setattr__(name, value)
        
        # Notify listeners if the value changed and the model is initialized
        if self._initialized and old_value != value:
            self._notify_listeners(name, old_value, value)
            
    def _notify_listeners(self, field: str, old_value: Any, new_value: Any) -> None:
        """
        Notify all registered listeners of a model update.
        
        Args:
            field: The name of the field that changed
            old_value: The old value of the field
            new_value: The new value of the field
        """
        update = ModelUpdate(self, field, old_value, new_value)
        for listener in self._listeners:
            try:
                listener(update)
            except Exception as e:
                self._logger.error(f"Error in model update listener: {str(e)}")
                
    def add_listener(self, callback: Callable[[ModelUpdate], None]) -> None:
        """
        Register a callback to be notified of model updates.
        
        Args:
            callback: The function to call when the model is updated
        """
        if callback not in self._listeners:
            self._listeners.append(callback)
            
    def remove_listener(self, callback: Callable[[ModelUpdate], None]) -> None:
        """
        Unregister a model update callback.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.
        
        Returns:
            A dictionary representation of the model
        """
        result = {}
        for key, value in self.__dict__.items():
            # Skip private attributes and callables
            if not key.startswith('_') and not callable(value):
                # Handle nested models
                if hasattr(value, 'to_dict'):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """
        Create a model instance from a dictionary.
        
        Args:
            data: Dictionary containing model data
            
        Returns:
            A new model instance
        """
        return cls(**data)
    
    def to_json(self, **kwargs) -> str:
        """
        Convert the model to a JSON string.
        
        Args:
            **kwargs: Additional arguments to pass to json.dumps()
            
        Returns:
            A JSON string representation of the model
        """
        return json.dumps(self.to_dict(), **kwargs)
    
    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> 'BaseModel':
        """
        Create a model instance from a JSON string.
        
        Args:
            json_str: JSON string containing model data
            **kwargs: Additional arguments to pass to json.loads()
            
        Returns:
            A new model instance
        """
        data = json.loads(json_str, **kwargs)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: str, **kwargs) -> None:
        """
        Save the model to a file.
        
        Args:
            file_path: Path to the output file
            **kwargs: Additional arguments to pass to to_json()
        """
        with open(file_path, 'w') as f:
            f.write(self.to_json(**kwargs))
    
    @classmethod
    def load_from_file(cls, file_path: str, **kwargs) -> 'BaseModel':
        """
        Load a model from a file.
        
        Args:
            file_path: Path to the input file
            **kwargs: Additional arguments to pass to from_json()
            
        Returns:
            A new model instance
        """
        with open(file_path, 'r') as f:
            return cls.from_json(f.read(), **kwargs)
    
    def copy(self) -> 'BaseModel':
        """
        Create a deep copy of the model.
        
        Returns:
            A new model instance with the same data
        """
        return self.__class__.from_dict(self.to_dict())
    
    def __eq__(self, other: Any) -> bool:
        """Check if two models are equal by comparing their dictionary representations."""
        if not isinstance(other, self.__class__):
            return False
        return self.to_dict() == other.to_dict()
    
    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"<{self.__class__.__name__} {self.to_dict()}>"
