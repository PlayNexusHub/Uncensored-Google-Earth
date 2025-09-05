"""
Enhancement View for PlayNexus Satellite Toolkit
Provides UI for image enhancement operations.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Dict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from gui.components.views.base_view import BaseView, ViewConfig
from gui.components.controllers.enhancement_controller import EnhancementController, EnhancementRequest
from gui.utils.ui_utils import create_button, create_tooltip
from gui.utils.animation_utils import AnimationManager, EasingType, AnimationConfig

class EnhancementView(BaseView[EnhancementRequest]):
    """View for image enhancement operations."""
    
    def __init__(self, controller: EnhancementController, parent: Optional[tk.Widget] = None):
        config = ViewConfig(
            title="Image Enhancement",
            width=700,
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
        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value=str(Path.home() / "enhanced_images"))
        self.preserve_original_var = tk.BooleanVar(value=True)
        
        # Enhancement method variables
        self.method_vars = {}
        for method in self.controller.get_available_methods():
            self.method_vars[method['id']] = tk.BooleanVar()
        
        # Parameter variables
        self.noise_strength_var = tk.DoubleVar(value=0.5)
        self.contrast_factor_var = tk.DoubleVar(value=1.2)
        self.edge_threshold_var = tk.DoubleVar(value=0.1)
        
    def _create_widgets(self) -> None:
        """Create and arrange widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Image Enhancement",
            font=('Segoe UI', 16, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Create notebook for organized sections
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Input/Output tab
        io_frame = ttk.Frame(notebook, padding=15)
        notebook.add(io_frame, text="Input/Output")
        self._create_io_widgets(io_frame)
        
        # Enhancement Methods tab
        methods_frame = ttk.Frame(notebook, padding=15)
        notebook.add(methods_frame, text="Enhancement Methods")
        self._create_methods_widgets(methods_frame)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook, padding=15)
        notebook.add(params_frame, text="Parameters")
        self._create_parameters_widgets(params_frame)
        
        # Progress tab
        progress_frame = ttk.Frame(notebook, padding=15)
        notebook.add(progress_frame, text="Progress")
        self._create_progress_widgets(progress_frame)
        
        # Control buttons
        self._create_control_buttons(main_frame)
        
    def _create_io_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for input/output file selection."""
        # Input file selection
        input_frame = ttk.LabelFrame(parent, text="Input Image", padding=10)
        input_frame.pack(fill='x', pady=(0, 10))
        
        file_frame = ttk.Frame(input_frame)
        file_frame.pack(fill='x')
        
        ttk.Entry(
            file_frame,
            textvariable=self.input_file_var,
            width=50
        ).pack(side='left', fill='x', expand=True)
        
        create_button(
            file_frame,
            text="Browse...",
            command=self._browse_input_file,
            tooltip="Select input image file"
        ).pack(side='right', padx=(5, 0))
        
        # Supported formats info
        formats_label = ttk.Label(
            input_frame,
            text="Supported formats: TIFF, JPEG, PNG",
            font=('Segoe UI', 9),
            foreground='gray'
        )
        formats_label.pack(anchor='w', pady=(5, 0))
        
        # Output directory selection
        output_frame = ttk.LabelFrame(parent, text="Output Directory", padding=10)
        output_frame.pack(fill='x', pady=(0, 10))
        
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill='x')
        
        ttk.Entry(
            dir_frame,
            textvariable=self.output_dir_var,
            width=50
        ).pack(side='left', fill='x', expand=True)
        
        create_button(
            dir_frame,
            text="Browse...",
            command=self._browse_output_dir,
            tooltip="Select output directory"
        ).pack(side='right', padx=(5, 0))
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options", padding=10)
        options_frame.pack(fill='x')
        
        ttk.Checkbutton(
            options_frame,
            text="Preserve original image",
            variable=self.preserve_original_var
        ).pack(anchor='w')
        
    def _create_methods_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for enhancement method selection."""
        methods_frame = ttk.LabelFrame(parent, text="Enhancement Methods", padding=10)
        methods_frame.pack(fill='both', expand=True)
        
        # Create checkboxes for each method
        for method in self.controller.get_available_methods():
            method_frame = ttk.Frame(methods_frame)
            method_frame.pack(fill='x', pady=2)
            
            checkbox = ttk.Checkbutton(
                method_frame,
                text=method['name'],
                variable=self.method_vars[method['id']]
            )
            checkbox.pack(side='left')
            
            # Add tooltip with description
            create_tooltip(checkbox, method['description'])
        
        # Quick selection buttons
        quick_frame = ttk.Frame(methods_frame)
        quick_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Label(quick_frame, text="Quick Select:").pack(side='left')
        
        create_button(
            quick_frame,
            text="All",
            command=self._select_all_methods,
            tooltip="Select all enhancement methods"
        ).pack(side='left', padx=5)
        
        create_button(
            quick_frame,
            text="None",
            command=self._select_no_methods,
            tooltip="Deselect all enhancement methods"
        ).pack(side='left', padx=5)
        
        create_button(
            quick_frame,
            text="Basic",
            command=self._select_basic_methods,
            tooltip="Select basic enhancement methods"
        ).pack(side='left', padx=5)
        
    def _create_parameters_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for enhancement parameters."""
        # Noise Reduction Parameters
        noise_frame = ttk.LabelFrame(parent, text="Noise Reduction", padding=10)
        noise_frame.pack(fill='x', pady=(0, 10))
        
        noise_control_frame = ttk.Frame(noise_frame)
        noise_control_frame.pack(fill='x')
        
        ttk.Label(noise_control_frame, text="Strength:").pack(side='left')
        noise_scale = ttk.Scale(
            noise_control_frame,
            from_=0.0,
            to=1.0,
            orient='horizontal',
            variable=self.noise_strength_var,
            length=200
        )
        noise_scale.pack(side='left', padx=(10, 5))
        
        noise_value_label = ttk.Label(noise_control_frame, text="0.5")
        noise_value_label.pack(side='left')
        
        # Update label when scale changes
        def update_noise_label(value):
            noise_value_label.config(text=f"{float(value):.2f}")
        noise_scale.config(command=update_noise_label)
        
        # Contrast Enhancement Parameters
        contrast_frame = ttk.LabelFrame(parent, text="Contrast Enhancement", padding=10)
        contrast_frame.pack(fill='x', pady=(0, 10))
        
        contrast_control_frame = ttk.Frame(contrast_frame)
        contrast_control_frame.pack(fill='x')
        
        ttk.Label(contrast_control_frame, text="Factor:").pack(side='left')
        contrast_scale = ttk.Scale(
            contrast_control_frame,
            from_=0.1,
            to=5.0,
            orient='horizontal',
            variable=self.contrast_factor_var,
            length=200
        )
        contrast_scale.pack(side='left', padx=(10, 5))
        
        contrast_value_label = ttk.Label(contrast_control_frame, text="1.2")
        contrast_value_label.pack(side='left')
        
        def update_contrast_label(value):
            contrast_value_label.config(text=f"{float(value):.2f}")
        contrast_scale.config(command=update_contrast_label)
        
        # Edge Enhancement Parameters
        edge_frame = ttk.LabelFrame(parent, text="Edge Enhancement", padding=10)
        edge_frame.pack(fill='x')
        
        edge_control_frame = ttk.Frame(edge_frame)
        edge_control_frame.pack(fill='x')
        
        ttk.Label(edge_control_frame, text="Threshold:").pack(side='left')
        edge_scale = ttk.Scale(
            edge_control_frame,
            from_=0.01,
            to=1.0,
            orient='horizontal',
            variable=self.edge_threshold_var,
            length=200
        )
        edge_scale.pack(side='left', padx=(10, 5))
        
        edge_value_label = ttk.Label(edge_control_frame, text="0.1")
        edge_value_label.pack(side='left')
        
        def update_edge_label(value):
            edge_value_label.config(text=f"{float(value):.3f}")
        edge_scale.config(command=update_edge_label)
        
    def _create_progress_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for progress tracking."""
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill='x', pady=(0, 10))
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack()
        
        # Progress bar
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.pack(fill='x', pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill='x')
        
        # Log area
        log_frame = ttk.LabelFrame(parent, text="Processing Log", padding=10)
        log_frame.pack(fill='both', expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, wrap='word')
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
    def _create_control_buttons(self, parent: ttk.Frame) -> None:
        """Create control buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(side='bottom', pady=10)
        
        self.enhance_btn = create_button(
            button_frame,
            text="Start Enhancement",
            command=self._start_enhancement,
            tooltip="Start image enhancement process"
        )
        self.enhance_btn.pack(side='left', padx=5)
        
        self.cancel_btn = create_button(
            button_frame,
            text="Cancel",
            command=self._cancel_enhancement,
            tooltip="Cancel current enhancement"
        )
        self.cancel_btn.pack(side='left', padx=5)
        self.cancel_btn.configure(state='disabled')
        
        create_button(
            button_frame,
            text="Clear Log",
            command=self._clear_log,
            tooltip="Clear the processing log"
        ).pack(side='left', padx=5)
        
        create_button(
            button_frame,
            text="Open Output",
            command=self._open_output_dir,
            tooltip="Open output directory in file explorer"
        ).pack(side='left', padx=5)
    
    def _browse_input_file(self) -> None:
        """Browse for input image file."""
        filetypes = [
            ("Image files", "*.tif *.tiff *.jpg *.jpeg *.png"),
            ("TIFF files", "*.tif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=filetypes
        )
        
        if filename:
            self.input_file_var.set(filename)
    
    def _browse_output_dir(self) -> None:
        """Browse for output directory."""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        if directory:
            self.output_dir_var.set(directory)
    
    def _select_all_methods(self) -> None:
        """Select all enhancement methods."""
        for var in self.method_vars.values():
            var.set(True)
    
    def _select_no_methods(self) -> None:
        """Deselect all enhancement methods."""
        for var in self.method_vars.values():
            var.set(False)
    
    def _select_basic_methods(self) -> None:
        """Select basic enhancement methods."""
        self._select_no_methods()
        basic_methods = ['noise_reduction', 'contrast_enhancement']
        for method_id in basic_methods:
            if method_id in self.method_vars:
                self.method_vars[method_id].set(True)
    
    def _start_enhancement(self) -> None:
        """Start the enhancement process."""
        # Validate input
        if not self.input_file_var.get():
            self.show_error("Input Error", "Please select an input image file.")
            return
        
        # Get selected methods
        selected_methods = [
            method_id for method_id, var in self.method_vars.items()
            if var.get()
        ]
        
        if not selected_methods:
            self.show_error("Method Error", "Please select at least one enhancement method.")
            return
        
        # Create enhancement request
        request = EnhancementRequest(
            input_file=self.input_file_var.get(),
            output_dir=self.output_dir_var.get(),
            methods=selected_methods,
            noise_reduction_strength=self.noise_strength_var.get(),
            contrast_factor=self.contrast_factor_var.get(),
            edge_threshold=self.edge_threshold_var.get(),
            preserve_original=self.preserve_original_var.get()
        )
        
        # Start enhancement
        success = self.controller.start_enhancement(
            request,
            progress_callback=self._on_progress_update,
            completion_callback=self._on_enhancement_complete
        )
        
        if success:
            self.enhance_btn.configure(state='disabled')
            self.cancel_btn.configure(state='normal')
            self._log_message("Enhancement started...")
        else:
            self.show_error("Enhancement Error", "Failed to start enhancement. Check the log for details.")
    
    def _cancel_enhancement(self) -> None:
        """Cancel the current enhancement."""
        if self.controller.cancel_enhancement():
            self.enhance_btn.configure(state='normal')
            self.cancel_btn.configure(state='disabled')
            self._log_message("Enhancement cancelled by user.")
    
    def _clear_log(self) -> None:
        """Clear the processing log."""
        self.log_text.delete(1.0, tk.END)
    
    def _open_output_dir(self) -> None:
        """Open output directory in file explorer."""
        import subprocess
        import platform
        
        output_dir = self.output_dir_var.get()
        if not Path(output_dir).exists():
            self.show_warning("Directory Not Found", f"Output directory does not exist: {output_dir}")
            return
        
        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", output_dir])
            else:  # Linux
                subprocess.run(["xdg-open", output_dir])
        except Exception as e:
            self.show_error("Error", f"Failed to open directory: {e}")
    
    def _on_progress_update(self, message: str, progress: float) -> None:
        """Handle progress updates."""
        self.status_var.set(message)
        self.progress_var.set(progress * 100)
        self._log_message(f"Progress: {message}")
        self.root.update_idletasks()
    
    def _on_enhancement_complete(self, success: bool, message: str, output_files: List[str]) -> None:
        """Handle enhancement completion."""
        self.enhance_btn.configure(state='normal')
        self.cancel_btn.configure(state='disabled')
        
        if success:
            self.status_var.set("Enhancement completed successfully")
            self.progress_var.set(100)
            
            # Show completion dialog with output files
            file_list = "\n".join([Path(f).name for f in output_files])
            self.show_info(
                "Enhancement Complete", 
                f"{message}\n\nOutput files:\n{file_list}"
            )
        else:
            self.status_var.set("Enhancement failed")
            self.progress_var.set(0)
            self.show_error("Enhancement Failed", message)
        
        self._log_message(f"Enhancement completed: {message}")
    
    def _log_message(self, message: str) -> None:
        """Add a message to the log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
