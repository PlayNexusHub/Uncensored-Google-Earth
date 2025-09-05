"""Pytest configuration and fixtures."""
import pytest
import tkinter as tk
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="function")
def tk_root():
    """Create a Tk root window for testing."""
    root = tk.Tk()
    root.withdraw()  # Don't show the window
    yield root
    root.destroy()

@pytest.fixture(scope="function")
def mock_widget():
    """Create a mock widget for testing."""
    class MockWidget:
        def __init__(self):
            self.attributes_calls = []
            self.place_calls = []
            self.pack_calls = []
            self.grid_calls = []
            self._x = 0
            self._y = 0
            self._alpha = 1.0
            self._state = "normal"
            
        def attributes(self, **kwargs):
            self.attributes_calls.append(kwargs)
            if '-alpha' in kwargs:
                self._alpha = kwargs['-alpha']
            if '-state' in kwargs:
                self._state = kwargs['-state']
                
        def winfo_x(self):
            return self._x
            
        def winfo_y(self):
            return self._y
            
        def place(self, **kwargs):
            self.place_calls.append(kwargs)
            if 'x' in kwargs:
                self._x = kwargs['x']
            if 'y' in kwargs:
                self._y = kwargs['y']
                
        def pack(self, **kwargs):
            self.pack_calls.append(kwargs)
            
        def grid(self, **kwargs):
            self.grid_calls.append(kwargs)
            
        def winfo_ismapped(self):
            return hasattr(self, '_mapped') and self._mapped
            
        def deiconify(self):
            self._mapped = True
            
        def withdraw(self):
            self._mapped = False
            
        def update(self):
            pass
    
    return MockWidget()

@pytest.fixture(scope="function")
def mock_controller():
    """Create a mock controller for testing."""
    class MockController:
        def __init__(self):
            self.calls = []
            
        def handle_event(self, event, data=None):
            self.calls.append((event, data))
            
        def cleanup(self):
            self.calls.append(('cleanup', None))
    
    return MockController()

@pytest.fixture(scope="function")
def mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self):
            self._listeners = []
            self.data = {}
            
        def add_listener(self, callback):
            self._listeners.append(callback)
            
        def remove_listener(self, callback):
            self._listeners.remove(callback)
            
        def to_dict(self):
            return self.data.copy()
    
    return MockModel()
