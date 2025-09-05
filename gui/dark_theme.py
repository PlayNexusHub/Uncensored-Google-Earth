"""
Dark Theme Configuration for PlayNexus Satellite Toolkit
Provides better contrast and readability for the GUI.
"""

import tkinter as tk
from tkinter import ttk


def apply_dark_theme(root: tk.Tk):
    """Apply a dark theme with good contrast for better readability."""
    
    # Configure the style
    style = ttk.Style()
    
    # Set the theme to a base theme that supports customization
    try:
        style.theme_use('clam')
    except:
        style.theme_use('default')
    
    # Dark color scheme with good contrast
    colors = {
        'bg': '#2d3748',           # Dark blue-gray background
        'fg': '#e2e8f0',           # Light gray text
        'select_bg': '#4a5568',    # Darker selection background
        'select_fg': '#ffffff',    # White selected text
        'entry_bg': '#4a5568',     # Entry background
        'entry_fg': '#ffffff',     # Entry text
        'button_bg': '#4299e1',    # Blue button background
        'button_fg': '#ffffff',    # White button text
        'frame_bg': '#2d3748',     # Frame background
        'label_fg': '#e2e8f0',     # Label text color
        'border': '#4a5568'        # Border color
    }
    
    # Configure root window
    root.configure(bg=colors['bg'])
    
    # Configure ttk styles
    style.configure('TFrame', 
                   background=colors['frame_bg'],
                   borderwidth=1,
                   relief='flat')
    
    style.configure('TLabel',
                   background=colors['bg'],
                   foreground=colors['label_fg'],
                   font=('Segoe UI', 10))
    
    style.configure('TButton',
                   background=colors['button_bg'],
                   foreground=colors['button_fg'],
                   borderwidth=1,
                   focuscolor='none',
                   font=('Segoe UI', 10))
    
    style.map('TButton',
              background=[('active', '#3182ce'),
                         ('pressed', '#2c5282')])
    
    style.configure('TEntry',
                   fieldbackground=colors['entry_bg'],
                   foreground=colors['entry_fg'],
                   borderwidth=1,
                   insertcolor=colors['entry_fg'],
                   font=('Segoe UI', 10))
    
    style.configure('TLabelFrame',
                   background=colors['bg'],
                   foreground=colors['label_fg'],
                   borderwidth=2,
                   relief='groove',
                   font=('Segoe UI', 10, 'bold'))
    
    style.configure('TLabelFrame.Label',
                   background=colors['bg'],
                   foreground=colors['label_fg'],
                   font=('Segoe UI', 10, 'bold'))
    
    style.configure('TNotebook',
                   background=colors['bg'],
                   borderwidth=1)
    
    style.configure('TNotebook.Tab',
                   background=colors['select_bg'],
                   foreground=colors['fg'],
                   padding=[10, 5],
                   font=('Segoe UI', 10))
    
    style.map('TNotebook.Tab',
              background=[('selected', colors['button_bg']),
                         ('active', colors['select_bg'])])
    
    style.configure('Vertical.TScrollbar',
                   background=colors['select_bg'],
                   troughcolor=colors['bg'],
                   borderwidth=1,
                   arrowcolor=colors['fg'],
                   darkcolor=colors['border'],
                   lightcolor=colors['border'])
    
    style.configure('TProgressbar',
                   background=colors['button_bg'],
                   troughcolor=colors['select_bg'],
                   borderwidth=1,
                   lightcolor=colors['button_bg'],
                   darkcolor=colors['button_bg'])
    
    # Configure Text widget colors (for scrolled text areas)
    text_config = {
        'bg': colors['entry_bg'],
        'fg': colors['entry_fg'],
        'insertbackground': colors['entry_fg'],
        'selectbackground': colors['button_bg'],
        'selectforeground': colors['select_fg'],
        'font': ('Consolas', 10)
    }
    
    return colors, text_config


def configure_text_widget(text_widget: tk.Text, text_config: dict):
    """Configure a Text widget with the dark theme colors."""
    text_widget.configure(**text_config)


def create_dark_canvas(parent: tk.Widget, **kwargs) -> tk.Canvas:
    """Create a canvas with dark theme colors."""
    default_config = {
        'bg': '#2d3748',
        'highlightthickness': 0,
        'selectbackground': '#4299e1',
        'selectforeground': '#ffffff'
    }
    default_config.update(kwargs)
    return tk.Canvas(parent, **default_config)
