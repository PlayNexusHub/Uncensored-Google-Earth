"""
PlayNexus Satellite Toolkit - Advanced GUI Styling
Provides modern themes, animations, and professional UI elements.
"""

import tkinter as tk
from tkinter import ttk
import colorsys


class PlayNexusThemeManager:
    """Manages advanced themes and styling for the PlayNexus toolkit."""
    
    def __init__(self):
        self.current_theme = "playnexus_dark"
        self.themes = {
            "playnexus_dark": self._create_dark_theme(),
            "playnexus_light": self._create_light_theme(),
            "professional_blue": self._create_professional_blue_theme(),
            "earth_tone": self._create_earth_tone_theme(),
            "high_contrast": self._create_high_contrast_theme()
        }
    
    def _create_dark_theme(self):
        """Create the signature PlayNexus dark theme."""
        return {
            "bg_primary": "#1a1a1a",
            "bg_secondary": "#2d2d2d",
            "bg_tertiary": "#404040",
            "accent_primary": "#00d4ff",
            "accent_secondary": "#ff6b35",
            "accent_success": "#00ff88",
            "accent_warning": "#ffaa00",
            "accent_error": "#ff4757",
            "text_primary": "#ffffff",
            "text_secondary": "#b0b0b0",
            "text_muted": "#808080",
            "border": "#555555",
            "shadow": "#000000"
        }
    
    def _create_light_theme(self):
        """Create the PlayNexus light theme."""
        return {
            "bg_primary": "#ffffff",
            "bg_secondary": "#f8f9fa",
            "bg_tertiary": "#e9ecef",
            "accent_primary": "#007bff",
            "accent_secondary": "#fd7e14",
            "accent_success": "#28a745",
            "accent_warning": "#ffc107",
            "accent_error": "#dc3545",
            "text_primary": "#212529",
            "text_secondary": "#6c757d",
            "text_muted": "#adb5bd",
            "border": "#dee2e6",
            "shadow": "#000000"
        }
    
    def _create_professional_blue_theme(self):
        """Create a professional blue theme."""
        return {
            "bg_primary": "#f8fafc",
            "bg_secondary": "#e2e8f0",
            "bg_tertiary": "#cbd5e1",
            "accent_primary": "#1e40af",
            "accent_secondary": "#3b82f6",
            "accent_success": "#059669",
            "accent_warning": "#d97706",
            "accent_error": "#dc2626",
            "text_primary": "#1e293b",
            "text_secondary": "#475569",
            "text_muted": "#64748b",
            "border": "#94a3b8",
            "shadow": "#000000"
        }
    
    def _create_earth_tone_theme(self):
        """Create an earth-tone theme suitable for satellite imagery."""
        return {
            "bg_primary": "#faf6f1",
            "bg_secondary": "#f0e6d9",
            "bg_tertiary": "#e6d5c3",
            "accent_primary": "#8b4513",
            "accent_secondary": "#cd853f",
            "accent_success": "#228b22",
            "accent_warning": "#daa520",
            "accent_error": "#b22222",
            "text_primary": "#2f1b14",
            "text_secondary": "#5d4037",
            "text_muted": "#8d6e63",
            "border": "#a1887f",
            "shadow": "#000000"
        }
    
    def _create_high_contrast_theme(self):
        """Create a high-contrast theme for accessibility."""
        return {
            "bg_primary": "#000000",
            "bg_secondary": "#ffffff",
            "bg_tertiary": "#000000",
            "accent_primary": "#ffff00",
            "accent_secondary": "#00ffff",
            "accent_success": "#00ff00",
            "accent_warning": "#ff8000",
            "accent_error": "#ff0000",
            "text_primary": "#ffffff",
            "text_secondary": "#ffffff",
            "text_muted": "#ffffff",
            "border": "#ffffff",
            "shadow": "#ffffff"
        }
    
    def apply_theme(self, style, theme_name="playnexus_dark"):
        """Apply a specific theme to the ttk style."""
        if theme_name not in self.themes:
            theme_name = "playnexus_dark"
        
        self.current_theme = theme_name
        theme = self.themes[theme_name]
        
        # Configure base styles
        style.configure("TFrame", background=theme["bg_primary"])
        style.configure("TLabel", background=theme["bg_primary"], foreground=theme["text_primary"])
        style.configure("TButton", background=theme["bg_secondary"], foreground=theme["text_primary"])
        style.configure("TEntry", fieldbackground=theme["bg_secondary"], foreground=theme["text_primary"])
        style.configure("TCombobox", fieldbackground=theme["bg_secondary"], foreground=theme["text_primary"])
        style.configure("TNotebook", background=theme["bg_primary"])
        style.configure("TNotebook.Tab", background=theme["bg_secondary"], foreground=theme["text_primary"])
        style.configure("TProgressbar", background=theme["accent_primary"], troughcolor=theme["bg_tertiary"])
        style.configure("TScale", background=theme["bg_primary"], troughcolor=theme["bg_tertiary"])
        style.configure("TCheckbutton", background=theme["bg_primary"], foreground=theme["text_primary"])
        style.configure("TRadiobutton", background=theme["bg_primary"], foreground=theme["text_primary"])
        
        # Configure custom styles
        style.configure("Title.TLabel", 
                       font=("Segoe UI", 24, "bold"), 
                       foreground=theme["accent_primary"],
                       background=theme["bg_primary"])
        
        style.configure("Header.TLabel", 
                       font=("Segoe UI", 16, "bold"), 
                       foreground=theme["accent_secondary"],
                       background=theme["bg_primary"])
        
        style.configure("Subtitle.TLabel", 
                       font=("Segoe UI", 12, "italic"), 
                       foreground=theme["text_secondary"],
                       background=theme["bg_primary"])
        
        style.configure("Accent.TButton", 
                       font=("Segoe UI", 10, "bold"),
                       background=theme["accent_primary"],
                       foreground=theme["bg_primary"])
        
        style.configure("Success.TButton", 
                       font=("Segoe UI", 10, "bold"),
                       background=theme["accent_success"],
                       foreground=theme["bg_primary"])
        
        style.configure("Warning.TButton", 
                       font=("Segoe UI", 10, "bold"),
                       background=theme["accent_warning"],
                       foreground=theme["bg_primary"])
        
        style.configure("Error.TButton", 
                       font=("Segoe UI", 10, "bold"),
                       background=theme["accent_error"],
                       foreground=theme["bg_primary"])
        
        # Configure notebook tab styles
        style.map("TNotebook.Tab",
                 background=[("selected", theme["accent_primary"]),
                           ("active", theme["bg_tertiary"])],
                 foreground=[("selected", theme["bg_primary"]),
                           ("active", theme["text_primary"])])
        
        # Configure button styles
        style.map("TButton",
                 background=[("active", theme["accent_secondary"]),
                           ("pressed", theme["bg_tertiary"])],
                 foreground=[("active", theme["bg_primary"]),
                           ("pressed", theme["text_primary"])])
        
        return theme
    
    def get_current_theme(self):
        """Get the currently applied theme."""
        return self.themes[self.current_theme]


class PlayNexusAnimations:
    """Provides smooth animations and transitions for the GUI."""
    
    @staticmethod
    def fade_in(widget, duration=300):
        """Fade in a widget with smooth animation."""
        widget.attributes('-alpha', 0.0)
        
        def animate(alpha=0.0):
            if alpha < 1.0:
                widget.attributes('-alpha', alpha)
                widget.after(10, lambda: animate(alpha + 0.05))
        
        animate()
    
    @staticmethod
    def slide_in(widget, direction="left", duration=300):
        """Slide in a widget from a specific direction."""
        original_x = widget.winfo_x()
        original_y = widget.winfo_y()
        
        if direction == "left":
            widget.place(x=original_x - 100, y=original_y)
            target_x = original_x
        elif direction == "right":
            widget.place(x=original_x + 100, y=original_y)
            target_x = original_x
        elif direction == "top":
            widget.place(x=original_x, y=original_y - 100)
            target_y = original_y
        elif direction == "bottom":
            widget.place(x=original_x, y=original_y + 100)
            target_y = original_y
        
        def animate(step=0):
            if step < duration:
                progress = step / duration
                if direction in ["left", "right"]:
                    current_x = original_x + (target_x - original_x) * progress
                    widget.place(x=current_x, y=original_y)
                else:
                    current_y = original_y + (target_y - original_y) * progress
                    widget.place(x=original_x, y=current_y)
                widget.after(10, lambda: animate(step + 10))
            else:
                widget.place(x=original_x, y=original_y)
        
        animate()
    
    @staticmethod
    def pulse(widget, duration=1000):
        """Create a pulsing effect on a widget."""
        original_bg = widget.cget("background")
        
        def pulse_effect(step=0):
            if step < duration:
                progress = step / duration
                # Create a sine wave effect
                intensity = 0.5 + 0.5 * (1 + (progress * 2 * 3.14159))
                # Modify background color
                try:
                    r, g, b = widget.winfo_rgb(original_bg)
                    r = int(r * intensity / 65535)
                    g = int(g * intensity / 65535)
                    b = int(b * intensity / 65535)
                    new_color = f"#{r:02x}{g:02x}{b:02x}"
                    widget.configure(background=new_color)
                except:
                    pass
                widget.after(50, lambda: pulse_effect(step + 50))
            else:
                widget.configure(background=original_bg)
        
        pulse_effect()


class PlayNexusIcons:
    """Provides icon management for the toolkit."""
    
    @staticmethod
    def get_icon_symbol(icon_type):
        """Get Unicode symbols for icons."""
        icons = {
            "satellite": "🛰️",
            "earth": "🌍",
            "analysis": "📊",
            "download": "📥",
            "upload": "📤",
            "settings": "⚙️",
            "help": "❓",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "play": "▶️",
            "pause": "⏸️",
            "stop": "⏹️",
            "refresh": "🔄",
            "search": "🔍",
            "filter": "🔧",
            "export": "📤",
            "import": "📥",
            "save": "💾",
            "load": "📂",
            "new": "🆕",
            "edit": "✏️",
            "delete": "🗑️",
            "copy": "📋",
            "paste": "📋",
            "cut": "✂️",
            "undo": "↶",
            "redo": "↷",
            "zoom_in": "🔍+",
            "zoom_out": "🔍-",
            "home": "🏠",
            "back": "⬅️",
            "forward": "➡️",
            "up": "⬆️",
            "down": "⬇️",
            "calendar": "📅",
            "clock": "🕐",
            "location": "📍",
            "map": "🗺️",
            "camera": "📷",
            "image": "🖼️",
            "video": "🎥",
            "audio": "🎵",
            "file": "📄",
            "folder": "📁",
            "database": "🗄️",
            "network": "🌐",
            "cloud": "☁️",
            "lock": "🔒",
            "unlock": "🔓",
            "key": "🔑",
            "user": "👤",
            "users": "👥",
            "star": "⭐",
            "heart": "❤️",
            "like": "👍",
            "dislike": "👎",
            "share": "📤",
            "link": "🔗",
            "mail": "✉️",
            "phone": "📞",
            "message": "💬",
            "notification": "🔔",
            "alert": "🚨",
            "fire": "🔥",
            "water": "💧",
            "leaf": "🍃",
            "tree": "🌳",
            "flower": "🌸",
            "sun": "☀️",
            "moon": "🌙",
            "rain": "🌧️",
            "snow": "❄️",
            "storm": "⛈️",
            "wind": "💨",
            "temperature": "🌡️",
            "humidity": "💧",
            "pressure": "📊",
            "altitude": "📈",
            "speed": "🏃",
            "distance": "📏",
            "time": "⏱️",
            "battery": "🔋",
            "signal": "📶",
            "wifi": "📶",
            "bluetooth": "📶",
            "gps": "📍",
            "compass": "🧭",
            "ruler": "📏",
            "calculator": "🧮",
            "chart": "📊",
            "graph": "📈",
            "pie": "🥧",
            "bar": "📊",
            "line": "📈",
            "scatter": "🔵",
            "histogram": "📊",
            "heatmap": "🔥",
            "contour": "🗺️",
            "surface": "🏔️",
            "volume": "📦",
            "cube": "🧊",
            "sphere": "⚪",
            "cylinder": "🥫",
            "cone": "🍦",
            "pyramid": "🔺",
            "prism": "🔷",
            "torus": "🍩",
            "mesh": "🕸️",
            "wireframe": "🔲",
            "solid": "🔳",
            "transparent": "👻",
            "opaque": "🕶️",
            "reflection": "💎",
            "refraction": "💠",
            "diffraction": "✨",
            "interference": "🌈",
            "polarization": "💫",
            "absorption": "⚫",
            "emission": "💡",
            "scattering": "💨",
            "transmission": "🔮",
            "absorption": "⚫",
            "reflection": "💎",
            "refraction": "💠",
            "diffraction": "✨",
            "interference": "🌈",
            "polarization": "💫",
            "absorption": "⚫",
            "emission": "💡",
            "scattering": "💨",
            "transmission": "🔮"
        }
        return icons.get(icon_type, "❓")


class PlayNexusTooltips:
    """Provides advanced tooltip functionality."""
    
    def __init__(self, parent):
        self.parent = parent
        self.tooltip_window = None
        self.tooltip_text = ""
        self.tooltip_delay = 1000  # milliseconds
        self.tooltip_timer = None
    
    def create_tooltip(self, widget, text, delay=None):
        """Create a tooltip for a widget."""
        if delay is None:
            delay = self.tooltip_delay
        
        def show_tooltip(event):
            self._show_tooltip(event, text)
        
        def hide_tooltip(event):
            self._hide_tooltip()
        
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
        widget.bind("<Button-1>", hide_tooltip)
    
    def _show_tooltip(self, event, text):
        """Show the tooltip window."""
        self._hide_tooltip()
        
        x, y, _, _ = event.widget.bbox("insert")
        x += event.widget.winfo_rootx() + 25
        y += event.widget.winfo_rooty() + 20
        
        self.tooltip_window = tk.Toplevel(self.parent)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(self.tooltip_window, text=text, 
                        justify=tk.LEFT,
                        background="#ffffe0", 
                        relief=tk.SOLID, 
                        borderwidth=1,
                        font=("Segoe UI", 8, "normal"))
        label.pack(ipadx=5, ipady=2)
        
        self.tooltip_text = text
    
    def _hide_tooltip(self):
        """Hide the tooltip window."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
            self.tooltip_text = ""


class PlayNexusStatusBar:
    """Provides an enhanced status bar with multiple indicators."""
    
    def __init__(self, parent, theme_manager):
        self.parent = parent
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_current_theme()
        
        self.status_frame = ttk.Frame(parent)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        
        # Configure grid weights
        self.status_frame.grid_columnconfigure(1, weight=1)
        
        # Status indicators
        self.status_label = ttk.Label(self.status_frame, text="Ready", 
                                     style="Subtitle.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w", padx=5)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(self.status_frame, mode="determinate")
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=(10, 5))
        
        # System info
        self.system_info_frame = ttk.Frame(self.status_frame)
        self.system_info_frame.grid(row=0, column=2, sticky="e", padx=5)
        
        self.cpu_label = ttk.Label(self.system_info_frame, text="CPU: --", 
                                  style="Subtitle.TLabel")
        self.cpu_label.pack(side="right", padx=2)
        
        self.memory_label = ttk.Label(self.system_info_frame, text="RAM: --", 
                                     style="Subtitle.TLabel")
        self.memory_label.pack(side="right", padx=2)
        
        self.status_bar = self.status_frame
    
    def update_status(self, text, progress=None):
        """Update the status bar text and progress."""
        self.status_label.config(text=text)
        if progress is not None:
            self.progress_bar["value"] = progress
    
    def update_system_info(self, cpu_percent, memory_percent):
        """Update system information display."""
        self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")
        self.memory_label.config(text=f"RAM: {memory_percent:.1f}%")
    
    def show_progress(self, show=True):
        """Show or hide the progress bar."""
        if show:
            self.progress_bar.pack(side="right", fill="x", expand=True, padx=(10, 5))
        else:
            self.progress_bar.pack_forget()


class PlayNexusMenuBar:
    """Provides an enhanced menu bar with modern styling."""
    
    def __init__(self, parent, theme_manager):
        self.parent = parent
        self.theme_manager = theme_manager
        self.theme = theme_manager.get_current_theme()
        
        self.menubar = tk.Menu(parent)
        parent.config(menu=self.menubar)
        
        self._create_file_menu()
        self._create_processing_menu()
        self._create_analysis_menu()
        self._create_tools_menu()
        self._create_view_menu()
        self._create_help_menu()
    
    def _create_file_menu(self):
        """Create the file menu."""
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="📁 File", menu=file_menu)
        
        file_menu.add_command(label="🆕 New Project", accelerator="Ctrl+N")
        file_menu.add_command(label="📂 Open Project", accelerator="Ctrl+O")
        file_menu.add_command(label="💾 Save Project", accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="📤 Export Results", accelerator="Ctrl+E")
        file_menu.add_command(label="📥 Import Data", accelerator="Ctrl+I")
        file_menu.add_separator()
        file_menu.add_command(label="⚙️ Project Settings")
        file_menu.add_separator()
        file_menu.add_command(label="❌ Exit", accelerator="Alt+F4")
    
    def _create_processing_menu(self):
        """Create the processing menu."""
        processing_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="🔧 Processing", menu=processing_menu)
        
        processing_menu.add_command(label="🖼️ Image Enhancement")
        processing_menu.add_command(label="🚨 Anomaly Detection")
        processing_menu.add_command(label="📊 Comprehensive Analysis")
        processing_menu.add_separator()
        processing_menu.add_command(label="🔄 Batch Processing")
        processing_menu.add_command(label="⏱️ Processing Queue")
    
    def _create_analysis_menu(self):
        """Create the analysis menu."""
        analysis_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="📊 Analysis", menu=analysis_menu)
        
        analysis_menu.add_command(label="🌱 NDVI Analysis")
        analysis_menu.add_command(label="💧 NDWI Analysis")
        analysis_menu.add_command(label="🕒 Time Series Analysis")
        analysis_menu.add_command(label="🔄 Change Detection")
        analysis_menu.add_separator()
        analysis_menu.add_command(label="🤖 Machine Learning")
        analysis_menu.add_command(label="📈 Statistical Analysis")
    
    def _create_tools_menu(self):
        """Create the tools menu."""
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="🛠️ Tools", menu=tools_menu)
        
        tools_menu.add_command(label="⚙️ Configuration")
        tools_menu.add_command(label="📊 Progress Monitor")
        tools_menu.add_command(label="🔍 Data Explorer")
        tools_menu.add_separator()
        tools_menu.add_command(label="💻 System Information")
        tools_menu.add_command(label="🔧 Performance Tuning")
        tools_menu.add_separator()
        tools_menu.add_command(label="📋 Script Editor")
        tools_menu.add_command(label="🔄 Workflow Manager")
    
    def _create_view_menu(self):
        """Create the view menu."""
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="👁️ View", menu=view_menu)
        
        view_menu.add_command(label="🎨 Themes")
        view_menu.add_command(label="📱 Layout")
        view_menu.add_separator()
        view_menu.add_command(label="🔍 Zoom In", accelerator="Ctrl++")
        view_menu.add_command(label="🔍 Zoom Out", accelerator="Ctrl+-")
        view_menu.add_command(label="🔍 Reset Zoom", accelerator="Ctrl+0")
        view_menu.add_separator()
        view_menu.add_command(label="📊 Show Status Bar")
        view_menu.add_command(label="📊 Show Toolbar")
    
    def _create_help_menu(self):
        """Create the help menu."""
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="❓ Help", menu=help_menu)
        
        help_menu.add_command(label="📚 Documentation")
        help_menu.add_command(label="🎥 Tutorials")
        help_menu.add_command(label="❓ Quick Help", accelerator="F1")
        help_menu.add_separator()
        help_menu.add_command(label="🐛 Report Bug")
        help_menu.add_command(label="💡 Feature Request")
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ About PlayNexus")
        help_menu.add_command(label="📞 Support")
