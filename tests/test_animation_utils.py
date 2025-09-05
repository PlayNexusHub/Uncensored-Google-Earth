"""Tests for animation utilities."""
import pytest
import tkinter as tk
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gui.utils.animation_utils import (
    AnimationManager,
    EasingType,
    AnimationConfig,
    fade_in,
    fade_out,
)

class MockWidget:
    """Mock widget for testing animations."""
    def __init__(self):
        self.attributes_calls = []
        self.place_calls = []
        self._x = 0
        self._y = 0
        self._alpha = 1.0
        
    def attributes(self, **kwargs):
        self.attributes_calls.append(kwargs)
        if '-alpha' in kwargs:
            self._alpha = kwargs['-alpha']
    
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

def test_animation_manager_init():
    """Test AnimationManager initialization."""
    root = tk.Tk()
    try:
        manager = AnimationManager(root)
        assert manager is not None
    finally:
        root.destroy()

def test_animation_manager_animate():
    """Test animation manager's animate method."""
    root = tk.Tk()
    try:
        manager = AnimationManager(root)
        widget = MockWidget()
        
        # Test basic animation
        anim_id = manager.animate(
            widget,
            {'_x': (0, 100), '_y': (0, 200)},
            AnimationConfig(duration=100, easing=EasingType.LINEAR)
        )
        
        assert isinstance(anim_id, str)
        assert anim_id in manager.active_animations
        
        # Let the animation run
        root.update()
        
        # Stop the animation
        manager.stop_animation(anim_id)
        assert anim_id not in manager.active_animations
        
    finally:
        root.destroy()

def test_easing_functions():
    """Test easing functions."""
    manager = AnimationManager(tk.Tk())
    
    # Test linear easing
    assert manager._easing_function(0.5, EasingType.LINEAR) == 0.5
    
    # Test ease in
    assert 0 < manager._easing_function(0.5, EasingType.EASE_IN) < 0.5
    
    # Test ease out
    assert 0.5 < manager._easing_function(0.5, EasingType.EASE_OUT) < 1.0
    
    # Test ease in-out
    result = manager._easing_function(0.5, EasingType.EASE_IN_OUT)
    assert 0.0 < result < 1.0
    
    # Test bounce
    result = manager._easing_function(0.5, EasingType.BOUNCE)
    assert 0.0 <= result <= 1.0
    
    # Test elastic
    result = manager._easing_function(0.5, EasingType.ELASTIC)
    assert isinstance(result, float)

def test_fade_in():
    """Test fade in animation."""
    root = tk.Tk()
    try:
        widget = MockWidget()
        fade_in(widget, duration=100)
        
        # Should set initial alpha to 0
        assert widget.attributes_calls[0]['-alpha'] == 0.0
        
        # Should call deiconify
        assert len(widget.place_calls) > 0
        
    finally:
        root.destroy()

def test_fade_out():
    """Test fade out animation."""
    root = tk.Tk()
    try:
        widget = MockWidget()
        fade_out(widget, duration=100)
        
        # Should have started fade out
        assert len(widget.attributes_calls) > 0
        
    finally:
        root.destroy()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
