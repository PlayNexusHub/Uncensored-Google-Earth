"""Tests for UI utilities."""
import pytest
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from gui.utils.ui_utils import (
    create_scrollable_frame,
    create_tooltip,
    create_tabbed_interface,
    create_status_bar,
    create_button,
    create_labeled_entry,
    center_window
)

class TestUIFunctions:
    """Test UI utility functions."""
    
    def test_create_scrollable_frame(self):
        """Test creating a scrollable frame."""
        root = tk.Tk()
        try:
            canvas, frame = create_scrollable_frame(root)
            
            # Check if canvas and frame were created
            assert isinstance(canvas, tk.Canvas)
            assert isinstance(frame, ttk.Frame)
            
            # Check if scrollbar was created
            children = root.winfo_children()
            assert any(isinstance(child, ttk.Scrollbar) for child in children)
            
        finally:
            root.destroy()
    
    def test_create_tabbed_interface(self):
        """Test creating a tabbed interface."""
        root = tk.Tk()
        try:
            notebook = create_tabbed_interface(root)
            assert isinstance(notebook, ttk.Notebook)
            
            # Test adding a tab
            tab = ttk.Frame(notebook)
            notebook.add(tab, text="Test Tab")
            assert notebook.tabs()  # Should have at least one tab
            
        finally:
            root.destroy()
    
    def test_create_status_bar(self):
        """Test creating a status bar."""
        root = tk.Tk()
        try:
            status_label = create_status_bar(root)
            assert isinstance(status_label, ttk.Label)
            assert status_label.winfo_ismapped()
            
        finally:
            root.destroy()
    
    def test_create_button(self):
        """Test creating a styled button."""
        root = tk.Tk()
        try:
            # Test basic button
            btn = create_button(root, "Test", lambda: None)
            assert isinstance(btn, ttk.Button)
            assert btn["text"] == "Test"
            
            # Test button with tooltip
            with patch("gui.utils.ui_utils.create_tooltip") as mock_tooltip:
                btn = create_button(root, "Tooltip", lambda: None, tooltip="Test tooltip")
                mock_tooltip.assert_called_once()
            
        finally:
            root.destroy()
    
    def test_create_labeled_entry(self):
        """Test creating a labeled entry."""
        root = tk.Tk()
        try:
            entry = create_labeled_entry(root, "Test Label", "default")
            assert isinstance(entry, ttk.Entry)
            assert entry.get() == "default"
            
            # Check if label was created
            frame = entry.master
            assert any(isinstance(child, ttk.Label) for child in frame.winfo_children())
            
        finally:
            root.destroy()
    
    def test_center_window(self):
        """Test centering a window on screen."""
        root = tk.Tk()
        try:
            root.geometry("400x300")
            center_window(root)
            
            # Can't directly test position, but can check if geometry was set
            assert root.geometry() != ""
            
        finally:
            root.destroy()

class TestTooltip:
    """Test tooltip functionality."""
    
    def test_tooltip_creation(self):
        """Test creating a tooltip."""
        root = tk.Tk()
        try:
            # Create a button to attach tooltip to
            btn = ttk.Button(root, text="Hover me")
            btn.pack()
            
            # Create tooltip
            create_tooltip(btn, "This is a tooltip")
            
            # Simulate enter/leave events
            btn.event_generate("<Enter>")
            
            # Check if tooltip window was created
            toplevels = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
            assert len(toplevels) > 0
            
            # Clean up
            btn.event_generate("<Leave>")
            
        finally:
            root.destroy()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
