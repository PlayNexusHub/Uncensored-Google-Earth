"""
Base View for PlayNexus Satellite Toolkit
Provides common functionality for all views in the application.
"""
from typing import Any, Dict, Optional, Protocol, TypeVar, Tuple, Generic
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from scripts.error_handling import PlayNexusLogger
from scripts.config import get_config
from gui.components.controllers.base_controller import BaseController

T = TypeVar('T')

@dataclass
class ViewConfig:
    """Configuration for view initialization."""
    title: str = ""
    width: int = 800
    height: int = 600
    resizable: Tuple[bool, bool] = (True, True)
    theme: str = "default"
    parent: Optional[tk.Widget] = None

class BaseView(Generic[T]):
    """
    Base view class providing common functionality for all views.
    """
    
    def __init__(self, controller: BaseController[T], config: ViewConfig):
        """
        Initialize the base view.
        
        Args:
            controller: The controller managing this view
            config: View configuration
        """
        self.controller = controller
        self.config = config
        self.logger = PlayNexusLogger(f"view.{self.__class__.__name__.lower()}")
        
        # Create the root window if no parent provided
        if config.parent:
            self.root = tk.Toplevel(config.parent)
            self.is_main_window = False
        else:
            self.root = tk.Tk()
            self.is_main_window = True
            
        self._setup_window()
        self._create_widgets()
        self._setup_bindings()
        self._setup_style()
        
    def _setup_window(self) -> None:
        """Set up the window properties."""
        self.root.title(self.config.title or "PlayNexus Satellite Toolkit")
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.minsize(800, 600)
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def _create_widgets(self) -> None:
        """Create and arrange widgets in the view."""
        # To be implemented by subclasses
        pass
        
    def _setup_bindings(self) -> None:
        """Set up event bindings."""
        # Close handler for main window
        if self.is_main_window:
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
    def _setup_style(self) -> None:
        """Set up the visual style of the view."""
        style = ttk.Style()
        
        # Configure default styles
        style.theme_use(self.config.theme)
        
        # Configure common styles
        style.configure("TFrame", background="white")
        style.configure("TLabel", font=('Segoe UI', 10))
        style.configure("TButton", font=('Segoe UI', 10))
        style.configure("TEntry", font=('Segoe UI', 10))
        
    def update_view(self, data: Optional[T] = None) -> None:
        """
        Update the view with new data.
        
        Args:
            data: Optional data to update the view with
        """
        if data is not None:
            self._update_widgets(data)
            
    def _update_widgets(self, data: T) -> None:
        """
        Update individual widgets with new data.
        
        Args:
            data: The data to update widgets with
        """
        # To be implemented by subclasses
        pass
        
    def show(self) -> None:
        """Show the view."""
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
    def hide(self) -> None:
        """Hide the view."""
        self.root.withdraw()
        
    def on_close(self) -> None:
        """Handle window close event."""
        self.controller.cleanup()
        if self.is_main_window:
            self.root.quit()
        else:
            self.root.destroy()
            
    def run(self) -> None:
        """Run the main loop of the view."""
        if self.is_main_window:
            self.root.mainloop()
            
    def show_error(self, title: str, message: str) -> None:
        """
        Show an error message dialog.
        
        Args:
            title: The title of the error dialog
            message: The error message to display
        """
        tk.messagebox.showerror(title, message)
        
    def show_info(self, title: str, message: str) -> None:
        """
        Show an info message dialog.
        
        Args:
            title: The title of the info dialog
            message: The info message to display
        """
        tk.messagebox.showinfo(title, message)
        
    def show_warning(self, title: str, message: str) -> None:
        """
        Show a warning message dialog.
        
        Args:
            title: The title of the warning dialog
            message: The warning message to display
        """
        tk.messagebox.showwarning(title, message)
