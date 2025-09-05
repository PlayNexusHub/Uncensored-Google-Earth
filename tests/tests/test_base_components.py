"""Tests for base components."""
import pytest
import tkinter as tk
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gui.components.controllers.base_controller import BaseController, ControllerConfig
from gui.components.views.base_view import BaseView, ViewConfig
from gui.components.models.base_model import BaseModel
from gui.components.widgets.base_widget import BaseWidget

class TestBaseModel:
    """Test BaseModel functionality."""
    
    def test_model_initialization(self):
        """Test model initialization with default values."""
        model = BaseModel()
        assert model is not None
        assert hasattr(model, '_listeners')
        assert isinstance(model._listeners, list)
    
    def test_model_property_setter(self):
        """Test property setting with change notification."""
        model = BaseModel()
        callback = MagicMock()
        model.add_listener(callback)
        
        # Test setting a new property
        model.some_property = "test"
        assert model.some_property == "test"
        callback.assert_called_once()
    
    def test_model_serialization(self):
        """Test model serialization to dictionary."""
        model = BaseModel()
        model.name = "Test"
        model.value = 42
        
        data = model.to_dict()
        assert 'name' in data
        assert 'value' in data
        assert data['name'] == "Test"
        assert data['value'] == 42

class TestBaseController:
    """Test BaseController functionality."""
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        config = ControllerConfig(name="TestController")
        controller = BaseController(config)
        assert controller.config == config
        assert controller.logger is not None
    
    def test_controller_cleanup(self):
        """Test controller cleanup."""
        controller = BaseController(ControllerConfig(name="TestController"))
        controller.cleanup()  # Should not raise

class TestBaseView:
    """Test BaseView functionality."""
    
    def test_view_initialization(self):
        """Test view initialization."""
        controller = MagicMock()
        config = ViewConfig(title="Test View", width=400, height=300)
        
        view = BaseView(controller, config)
        
        assert view.controller == controller
        assert view.config == config
        assert view.root is not None
        assert view.root.title() == "Test View"
        
        # Cleanup
        view.root.destroy()
    
    def test_view_show_hide(self):
        """Test view show and hide methods."""
        view = BaseView(MagicMock(), ViewConfig())
        
        # Initially not visible
        assert not view.root.winfo_ismapped()
        
        # Show the view
        view.show()
        assert view.root.winfo_ismapped()
        
        # Hide the view
        view.hide()
        assert not view.root.winfo_ismapped()
        
        # Cleanup
        view.root.destroy()

class TestBaseWidget:
    """Test BaseWidget functionality."""
    
    def test_widget_initialization(self):
        """Test widget initialization."""
        parent = tk.Tk()
        try:
            widget = BaseWidget(parent, "test_widget")
            assert widget is not None
            assert widget.widget_id == "test_widget"
            assert widget.parent == parent
            assert widget._callbacks == {}
        finally:
            parent.destroy()
    
    def test_widget_callbacks(self):
        """Test widget callback registration and triggering."""
        parent = tk.Tk()
        try:
            widget = BaseWidget(parent, "test_widget")
            
            # Test callback registration
            mock_callback = MagicMock()
            widget.register_callback("test_event", mock_callback)
            assert "test_event" in widget._callbacks
            
            # Test callback triggering
            test_data = {"key": "value"}
            widget.trigger_callback("test_event", test_data)
            mock_callback.assert_called_once_with(test_data)
            
            # Test callback removal
            widget.unregister_callback("test_event", mock_callback)
            assert "test_event" not in widget._callbacks
            
        finally:
            parent.destroy()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
