"""
Base Widget for PlayNexus Satellite Toolkit
Provides common functionality for all custom widgets.
"""
from typing import Any, Dict, Optional, Tuple, TypeVar, Generic, Callable
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.error_handling import PlayNexusLogger
from scripts.config import get_config

T = TypeVar('T')

@dataclass
class WidgetConfig:
    """Configuration for widget initialization."""
    style: str = "TFrame"
    padding: Tuple[int, int, int, int] = (5, 5, 5, 5)
    sticky: str = "nsew"
    row: int = 0
    column: int = 0
    rowspan: int = 1
    columnspan: int = 1
    extra_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

class BaseWidget(ttk.Frame, Generic[T]):
    """
    Base widget class providing common functionality for all custom widgets.
    """
    
    def __init__(self, parent: tk.Widget, config: Optional[WidgetConfig] = None, **kwargs):
        """
        Initialize the base widget.
        
        Args:
            parent: The parent widget
            config: Widget configuration
            **kwargs: Additional arguments to pass to the ttk.Frame constructor
        """
        self.config = config or WidgetConfig(**kwargs)
        super().__init__(parent, style=self.config.style)
        
        self.logger = PlayNexusLogger(f"widget.{self.__class__.__name__.lower()}")
        self._callbacks: Dict[str, Callable[..., Any]] = {}
        
        # Configure grid
        self.grid(
            row=self.config.row,
            column=self.config.column,
            rowspan=self.config.rowspan,
            columnspan=self.config.columnspan,
            padx=self.config.padding[0],
            pady=self.config.padding[1],
            sticky=self.config.sticky
        )
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self._create_widgets()
        self._setup_bindings()
        
    def _create_widgets(self) -> None:
        """Create and arrange child widgets."""
        # To be implemented by subclasses
        pass
        
    def _setup_bindings(self) -> None:
        """Set up event bindings."""
        # To be implemented by subclasses
        pass
        
    def update_widget(self, data: Optional[T] = None) -> None:
        """
        Update the widget with new data.
        
        Args:
            data: Optional data to update the widget with
        """
        if data is not None:
            self._update_display(data)
            
    def _update_display(self, data: T) -> None:
        """
        Update the widget's display with new data.
        
        Args:
            data: The data to display
        """
        # To be implemented by subclasses
        pass
        
    def register_callback(self, event: str, callback: Callable[..., Any]) -> None:
        """
        Register a callback function for a widget event.
        
        Args:
            event: The event name
            callback: The function to call when the event occurs
        """
        self._callbacks[event] = callback
        
    def _trigger_callback(self, event: str, *args: Any) -> None:
        """
        Trigger a registered callback.
        
        Args:
            event: The event name
            *args: Arguments to pass to the callback
        """
        if event in self._callbacks:
            try:
                self._callbacks[event](*args)
            except Exception as e:
                self.logger.error(f"Error in {event} callback: {str(e)}")
                
    def set_enabled(self, enabled: bool = True) -> None:
        """
        Enable or disable the widget.
        
        Args:
            enabled: Whether to enable or disable the widget
        """
        state = "normal" if enabled else "disabled"
        for child in self.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                # Some widgets might not support state configuration
                pass
                
    def show(self) -> None:
        """Show the widget."""
        self.grid()
        
    def hide(self) -> None:
        """Hide the widget."""
        self.grid_remove()
        
    def destroy(self) -> None:
        """Clean up resources used by the widget."""
        # Clean up any resources
        self._callbacks.clear()
        super().destroy()
