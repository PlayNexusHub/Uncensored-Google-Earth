"""
Download View for PlayNexus Satellite Toolkit
Provides UI for satellite data download operations.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable
from pathlib import Path
import sys
from datetime import datetime, timedelta
from tkcalendar import Calendar

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gui.components.views.base_view import BaseView, ViewConfig
from gui.components.controllers.download_controller import DownloadController, DownloadRequest
from gui.utils.ui_utils import create_labeled_entry, create_button, create_tooltip
from gui.utils.animation_utils import AnimationManager, EasingType, AnimationConfig

class DownloadView(BaseView[DownloadRequest]):
    """View for satellite data download operations."""
    
    def __init__(self, controller: DownloadController, parent: Optional[tk.Widget] = None):
        config = ViewConfig(
            title="Satellite Data Download",
            width=800,
            height=600,
            resizable=(True, True)
        )
        super().__init__(controller, config, parent)
        self.animation_manager = AnimationManager(self.root)
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready")
        self._setup_variables()
        
    def _setup_variables(self) -> None:
        """Setup tkinter variables for form fields."""
        self.collection_var = tk.StringVar(value="sentinel-2-l2a")
        self.bbox_vars = {
            'west': tk.DoubleVar(value=-74.0),
            'south': tk.DoubleVar(value=40.7),
            'east': tk.DoubleVar(value=-73.9),
            'north': tk.DoubleVar(value=40.8)
        }
        self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"))
        self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "satellite_data"))
        self.max_cloud_var = tk.IntVar(value=20)
        self.max_items_var = tk.IntVar(value=10)
        
    def _create_widgets(self) -> None:
        """Create and arrange widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Satellite Data Download",
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook for organized sections
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Collection and Area tab
        area_frame = ttk.Frame(notebook, padding=15)
        notebook.add(area_frame, text="Collection & Area")
        self._create_area_widgets(area_frame)
        
        # Time and Filters tab
        time_frame = ttk.Frame(notebook, padding=15)
        notebook.add(time_frame, text="Time & Filters")
        self._create_time_widgets(time_frame)
        
        # Output and Progress tab
        output_frame = ttk.Frame(notebook, padding=15)
        notebook.add(output_frame, text="Output & Progress")
        self._create_output_widgets(output_frame)
        
        # Control buttons
        self._create_control_buttons(main_frame)
        
    def _create_area_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for collection and area selection."""
        # Collection selection
        collection_frame = ttk.LabelFrame(parent, text="Data Collection", padding=10)
        collection_frame.pack(fill='x', pady=(0, 10))
        
        collections = [
            ("Sentinel-2 L2A", "sentinel-2-l2a"),
            ("Landsat Collection 2 L2", "landsat-c2-l2"),
            ("Sentinel-1 GRD", "sentinel-1-grd")
        ]
        
        for i, (label, value) in enumerate(collections):
            ttk.Radiobutton(
                collection_frame,
                text=label,
                variable=self.collection_var,
                value=value
            ).grid(row=0, column=i, padx=10, sticky='w')
        
        # Bounding box
        bbox_frame = ttk.LabelFrame(parent, text="Area of Interest (Bounding Box)", padding=10)
        bbox_frame.pack(fill='x', pady=(0, 10))
        
        # Coordinate inputs
        coord_frame = ttk.Frame(bbox_frame)
        coord_frame.pack(fill='x')
        
        # North
        ttk.Label(coord_frame, text="North:").grid(row=0, column=1, padx=5)
        north_entry = ttk.Entry(coord_frame, textvariable=self.bbox_vars['north'], width=12)
        north_entry.grid(row=0, column=2, padx=5)
        
        # West and East
        ttk.Label(coord_frame, text="West:").grid(row=1, column=0, padx=5)
        west_entry = ttk.Entry(coord_frame, textvariable=self.bbox_vars['west'], width=12)
        west_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(coord_frame, text="East:").grid(row=1, column=2, padx=5)
        east_entry = ttk.Entry(coord_frame, textvariable=self.bbox_vars['east'], width=12)
        east_entry.grid(row=1, column=3, padx=5)
        
        # South
        ttk.Label(coord_frame, text="South:").grid(row=2, column=1, padx=5)
        south_entry = ttk.Entry(coord_frame, textvariable=self.bbox_vars['south'], width=12)
        south_entry.grid(row=2, column=2, padx=5)
        
        # Preset locations
        preset_frame = ttk.Frame(bbox_frame)
        preset_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(preset_frame, text="Presets:").pack(side='left')
        
        presets = [
            ("New York", [-74.0, 40.7, -73.9, 40.8]),
            ("London", [-0.2, 51.4, 0.0, 51.6]),
            ("Tokyo", [139.6, 35.6, 139.8, 35.7])
        ]
        
        for name, coords in presets:
            btn = ttk.Button(
                preset_frame,
                text=name,
                command=lambda c=coords: self._set_bbox_preset(c)
            )
            btn.pack(side='left', padx=5)
            create_tooltip(btn, f"Set bounding box to {name}")
    
    def _create_time_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for time range and filters."""
        # Date range
        date_frame = ttk.LabelFrame(parent, text="Date Range", padding=10)
        date_frame.pack(fill='x', pady=(0, 10))
        
        # Start date
        start_frame = ttk.Frame(date_frame)
        start_frame.pack(fill='x', pady=5)
        ttk.Label(start_frame, text="Start Date:").pack(side='left')
        start_entry = ttk.Entry(start_frame, textvariable=self.start_date_var, width=15)
        start_entry.pack(side='left', padx=(10, 5))
        ttk.Button(
            start_frame,
            text="📅",
            width=3,
            command=lambda: self._select_date(self.start_date_var)
        ).pack(side='left')
        
        # End date
        end_frame = ttk.Frame(date_frame)
        end_frame.pack(fill='x', pady=5)
        ttk.Label(end_frame, text="End Date:").pack(side='left')
        end_entry = ttk.Entry(end_frame, textvariable=self.end_date_var, width=15)
        end_entry.pack(side='left', padx=(10, 5))
        ttk.Button(
            end_frame,
            text="📅",
            width=3,
            command=lambda: self._select_date(self.end_date_var)
        ).pack(side='left')
        
        # Filters
        filter_frame = ttk.LabelFrame(parent, text="Filters", padding=10)
        filter_frame.pack(fill='x', pady=(0, 10))
        
        # Cloud cover
        cloud_frame = ttk.Frame(filter_frame)
        cloud_frame.pack(fill='x', pady=5)
        ttk.Label(cloud_frame, text="Max Cloud Cover (%):").pack(side='left')
        cloud_scale = ttk.Scale(
            cloud_frame,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.max_cloud_var,
            length=200
        )
        cloud_scale.pack(side='left', padx=(10, 5))
        cloud_label = ttk.Label(cloud_frame, textvariable=self.max_cloud_var)
        cloud_label.pack(side='left')
        
        # Max items
        items_frame = ttk.Frame(filter_frame)
        items_frame.pack(fill='x', pady=5)
        ttk.Label(items_frame, text="Max Items:").pack(side='left')
        items_spin = ttk.Spinbox(
            items_frame,
            from_=1,
            to=100,
            textvariable=self.max_items_var,
            width=10
        )
        items_spin.pack(side='left', padx=(10, 0))
    
    def _create_output_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for output directory and progress."""
        # Output directory
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding=10)
        output_frame.pack(fill='x', pady=(0, 10))
        
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill='x')
        
        ttk.Entry(
            dir_frame,
            textvariable=self.output_dir_var,
            width=50
        ).pack(side='left', fill='x', expand=True)
        
        ttk.Button(
            dir_frame,
            text="Browse...",
            command=self._browse_output_dir
        ).pack(side='right', padx=(5, 0))
        
        # Progress
        progress_frame = ttk.LabelFrame(parent, text="Download Progress", padding=10)
        progress_frame.pack(fill='both', expand=True)
        
        # Status label
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        # Log area
        log_frame = ttk.Frame(progress_frame)
        log_frame.pack(fill='both', expand=True)
        
        ttk.Label(log_frame, text="Download Log:").pack(anchor='w')
        
        self.log_text = tk.Text(log_frame, height=8, wrap='word')
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
    
    def _create_control_buttons(self, parent: ttk.Frame) -> None:
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(side='bottom', pady=10)
        
        self.download_btn = create_button(
            button_frame,
            text="Start Download",
            command=self._start_download,
            tooltip="Start downloading satellite data"
        )
        self.download_btn.pack(side='left', padx=5)
        
        self.cancel_btn = create_button(
            button_frame,
            text="Cancel",
            command=self._cancel_download,
            tooltip="Cancel current download"
        )
        self.cancel_btn.pack(side='left', padx=5)
        self.cancel_btn.configure(state='disabled')
        
        create_button(
            button_frame,
            text="Clear Log",
            command=self._clear_log,
            tooltip="Clear the download log"
        ).pack(side='left', padx=5)
    
    def _set_bbox_preset(self, coords: list) -> None:
        """Set bounding box to preset coordinates."""
        self.bbox_vars['west'].set(coords[0])
        self.bbox_vars['south'].set(coords[1])
        self.bbox_vars['east'].set(coords[2])
        self.bbox_vars['north'].set(coords[3])
    
    def _select_date(self, date_var: tk.StringVar) -> None:
        """Open date picker dialog."""
        def on_date_select():
            selected = cal.selection_get()
            date_var.set(selected.strftime("%Y-%m-%d"))
            date_window.destroy()
        
        date_window = tk.Toplevel(self.root)
        date_window.title("Select Date")
        date_window.geometry("300x250")
        date_window.transient(self.root)
        date_window.grab_set()
        
        cal = Calendar(date_window, selectmode='day')
        cal.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Button(
            date_window,
            text="Select",
            command=on_date_select
        ).pack(pady=5)
    
    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        if directory:
            self.output_dir_var.set(directory)
    
    def _start_download(self) -> None:
        """Start the download process."""
        # Create download request
        request = DownloadRequest(
            collection=self.collection_var.get(),
            bbox=[
                self.bbox_vars['west'].get(),
                self.bbox_vars['south'].get(),
                self.bbox_vars['east'].get(),
                self.bbox_vars['north'].get()
            ],
            start_date=self.start_date_var.get(),
            end_date=self.end_date_var.get(),
            output_dir=self.output_dir_var.get(),
            max_cloud_cover=self.max_cloud_var.get(),
            max_items=self.max_items_var.get()
        )
        
        # Start download
        success = self.controller.start_download(
            request,
            progress_callback=self._on_progress_update,
            completion_callback=self._on_download_complete
        )
        
        if success:
            self.download_btn.configure(state='disabled')
            self.cancel_btn.configure(state='normal')
            self._log_message("Download started...")
        else:
            self.show_error("Download Error", "Failed to start download. Check the log for details.")
    
    def _cancel_download(self) -> None:
        """Cancel the current download."""
        if self.controller.cancel_download():
            self.download_btn.configure(state='normal')
            self.cancel_btn.configure(state='disabled')
            self._log_message("Download cancelled by user.")
    
    def _clear_log(self) -> None:
        """Clear the download log."""
        self.log_text.delete(1.0, tk.END)
    
    def _on_progress_update(self, message: str, progress: float) -> None:
        """Handle progress updates."""
        self.status_var.set(message)
        self.progress_var.set(progress * 100)
        self._log_message(f"Progress: {message}")
        self.root.update_idletasks()
    
    def _on_download_complete(self, success: bool, message: str) -> None:
        """Handle download completion."""
        self.download_btn.configure(state='normal')
        self.cancel_btn.configure(state='disabled')
        
        if success:
            self.status_var.set("Download completed successfully")
            self.progress_var.set(100)
            self.show_info("Download Complete", message)
        else:
            self.status_var.set("Download failed")
            self.progress_var.set(0)
            self.show_error("Download Failed", message)
        
        self._log_message(f"Download completed: {message}")
    
    def _log_message(self, message: str) -> None:
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
