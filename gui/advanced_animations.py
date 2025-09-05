"""
PlayNexus Satellite Toolkit - Advanced Animation System
Provides smooth transitions, micro-interactions, and professional animations
inspired by top-tier design agencies like Apple, Notion, and Figma.
"""

import tkinter as tk
from tkinter import ttk
import math
import time
from typing import Callable, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import threading


# --- Easing Functions ---
def linear(t: float) -> float:
    """Linear easing: no acceleration or deceleration."""
    return t

def ease_in_quad(t: float) -> float:
    """Ease in (quadratic): slow start, fast end."""
    return t * t

def ease_out_quad(t: float) -> float:
    """Ease out (quadratic): fast start, slow end."""
    return 1 - (1 - t) * (1 - t)

def ease_in_out_quad(t: float) -> float:
    """Ease in out (quadratic): slow start and end, fast middle."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2

def ease_out_bounce(t: float) -> float:
    """Bounce easing: bounces at the end."""
    n1 = 7.5625
    d1 = 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375

def ease_out_elastic(t: float) -> float:
    """Elastic easing: elastic effect at the end."""
    c4 = (2 * math.pi) / 3
    if t == 0:
        return 0
    if t == 1:
        return 1
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1

EASING_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "linear": linear,
    "ease_in": ease_in_quad,
    "ease_out": ease_out_quad,
    "ease_in_out": ease_in_out_quad,
    "bounce": ease_out_bounce,
    "elastic": ease_out_elastic,
}

# --- Animation Configuration ---
@dataclass
class AnimationConfig:
    """Configuration for animations."""
    duration: float = 300  # milliseconds
    easing: str = "ease_out"
    delay: float = 0  # milliseconds
    on_complete: Optional[Callable[[], None]] = None


# --- Core Animation Engine ---
class AdvancedAnimations:
    """A powerful, refactored animation system for tkinter."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self._active_animations: Dict[str, bool] = {}
        self._animation_id_counter = 0

    def _generate_id(self, prefix: str) -> str:
        self._animation_id_counter += 1
        # Create a unique ID based on the widget and a counter
        return f"{prefix}_{id(prefix)}_{self._animation_id_counter}"

    def stop_animation(self, animation_id: str):
        """Stops a specific running animation."""
        if animation_id in self._active_animations:
            self._active_animations[animation_id] = False

    def stop_all_animations_for_widget(self, widget: tk.Widget):
        """Stops all animations associated with a specific widget."""
        widget_id_prefix = f"{id(widget)}"
        for anim_id in list(self._active_animations.keys()):
            if anim_id.startswith(widget_id_prefix):
                self.stop_animation(anim_id)

    def _animate(
        self,
        widget: tk.Widget,
        config: AnimationConfig,
        prop_configs: Dict[str, Tuple[float, float]],
        animation_id: str,
    ):
        """The core animation loop with frame timing and cleanup."""
        if not widget.winfo_exists():
            return None
            
        start_time = time.time()
        last_frame_time = start_time
        frame_delay = max(1, 1000 // 60)  # Target 60 FPS (16.67ms per frame)
        easing_func = EASING_FUNCTIONS.get(config.easing, ease_out_quad)
        
        # Pre-calculate value ranges for performance
        value_ranges = {
            prop: (start, end - start)
            for prop, (start, end) in prop_configs.items()
        }

        def tick():
            nonlocal last_frame_time
            
            if not widget.winfo_exists() or not self._active_animations.get(animation_id, False):
                self._cleanup_animation(animation_id)
                return

            current_time = time.time()
            elapsed = (current_time - start_time) * 1000  # Convert to ms
            
            if elapsed < config.delay:
                self.root.after(10, tick)
                return

            progress = min((elapsed - config.delay) / max(1, config.duration), 1.0)
            eased_progress = easing_func(progress)

            # Calculate new values using pre-computed ranges
            updates = {
                prop: start + (delta * eased_progress)
                for prop, (start, delta) in value_ranges.items()
            }

            self._apply_properties(widget, updates)

            if progress < 1.0:
                # Calculate dynamic frame delay based on actual frame time
                frame_time = (time.time() - last_frame_time) * 1000  # in ms
                delay = max(1, int(frame_delay - (frame_time - frame_delay)))
                last_frame_time = time.time()
                self.root.after(delay, tick)
            else:
                # Ensure final state is exactly as requested
                final_updates = {prop: vals[1] for prop, vals in prop_configs.items()}
                self._apply_properties(widget, final_updates)
                self._cleanup_animation(animation_id, config.on_complete)

        self._active_animations[animation_id] = True
        self.root.after(int(config.delay), tick)
        return animation_id
        
    def _cleanup_animation(self, animation_id: str, on_complete: Optional[Callable] = None):
        """Safely clean up animation resources."""
        if animation_id in self._active_animations:
            del self._active_animations[animation_id]
        if on_complete:
            try:
                on_complete()
            except Exception as e:
                print(f"Error in animation completion callback: {e}")
        
    def _apply_properties(self, widget: tk.Widget, props: Dict[str, float]):
        """Applies calculated properties to a widget with error handling."""
        if not widget.winfo_exists():
            return
            
        try:
            place_props = {}
            for k, v in props.items():
                if k in ('x', 'y', 'width', 'height'):
                    place_props[k] = int(round(v))
                
            if place_props:
                widget.place_configure(**place_props)

            if "alpha" in props:
                try:
                    alpha = max(0.0, min(1.0, float(props["alpha"])))
                    widget.attributes("-alpha", alpha)
                except (tk.TclError, ValueError):
                    pass  # Alpha not supported on all platforms/widgets
                    
        except Exception as e:
            import traceback
            print(f"Error applying properties: {e}")
            traceback.print_exc()

    # --- Public Animation Methods ---

    def fade(self, widget: tk.Widget, start: Optional[float] = None, end: float = 1.0, config: Optional[AnimationConfig] = None) -> Optional[str]:
        """Fade a widget by animating its alpha attribute.
        
        Args:
            widget: The widget to animate
            start: Starting alpha (0.0 to 1.0). If None, uses current alpha.
            end: Target alpha (0.0 to 1.0)
            config: Optional animation configuration
            
        Returns:
            Animation ID if successful, None if widget is invalid
        """
        if not widget.winfo_exists():
            return None
            
        config = config or AnimationConfig()
        self.stop_all_animations_for_widget(widget)
        animation_id = self._generate_id(f"{id(widget)}_fade")
        
        try:
            current_alpha = widget.attributes("-alpha")
        except (tk.TclError, AttributeError):
            current_alpha = 1.0
            
        start_alpha = max(0.0, min(1.0, float(start if start is not None else current_alpha)))
        end_alpha = max(0.0, min(1.0, float(end)))
        
        # Skip if no change needed
        if abs(start_alpha - end_alpha) < 0.01:
            widget.attributes("-alpha", end_alpha)
            if config.on_complete:
                self.root.after(10, config.on_complete)
            return None
            
        prop_configs = {"alpha": (start_alpha, end_alpha)}
        return self._animate(widget, config, prop_configs, animation_id)

    def slide(self, widget: tk.Widget, start_pos: Optional[Tuple[int, int]] = None, 
              end_pos: Optional[Tuple[int, int]] = None, config: Optional[AnimationConfig] = None) -> Optional[str]:
        """Slide a widget from a start to an end position.
        
        Args:
            widget: The widget to animate
            start_pos: Optional (x, y) starting position. If None, uses current position.
            end_pos: (x, y) target position
            config: Optional animation configuration
            
        Returns:
            Animation ID if successful, None if widget is invalid
        """
        if not widget.winfo_exists():
            return None
            
        config = config or AnimationConfig()
        self.stop_all_animations_for_widget(widget)
        animation_id = self._generate_id(f"{id(widget)}_slide")
        
        # Get current position if start_pos not provided
        if start_pos is None:
            try:
                start_x = widget.winfo_x()
                start_y = widget.winfo_y()
            except tk.TclError:
                start_x, start_y = 0, 0
        else:
            start_x, start_y = start_pos
            
        # Apply start position if provided
        if start_pos is not None:
            widget.place(x=start_x, y=start_y)
            
        if end_pos is None:
            end_x, end_y = start_x, start_y
        else:
            end_x, end_y = end_pos
            
        # Skip if no movement needed
        if (abs(start_x - end_x) < 1 and abs(start_y - end_y) < 1):
            if config.on_complete:
                self.root.after(10, config.on_complete)
            return None
            
        prop_configs = {
            "x": (start_x, end_x), 
            "y": (start_y, end_y)
        }
        return self._animate(widget, config, prop_configs, animation_id)
        
    def scale(self, widget: tk.Widget, start_scale: Optional[float] = None, 
             end_scale: float = 1.0, config: Optional[AnimationConfig] = None) -> Optional[str]:
        """Scale a widget in or out, centered on its current position.
        
        Args:
            widget: The widget to animate
            start_scale: Starting scale factor. If None, uses current size.
            end_scale: Target scale factor
            config: Optional animation configuration
            
        Returns:
            Animation ID if successful, None if widget is invalid
        """
        if not widget.winfo_exists():
            return None
            
        config = config or AnimationConfig()
        self.stop_all_animations_for_widget(widget)
        animation_id = self._generate_id(f"{id(widget)}_scale")

        try:
            orig_w, orig_h = widget.winfo_width(), widget.winfo_height()
            orig_x, orig_y = widget.winfo_x(), widget.winfo_y()
        except tk.TclError:
            return None
            
        if start_scale is None:
            # If start_scale not provided, calculate based on current size
            if orig_w > 0 and orig_h > 0:
                start_scale = 1.0
            else:
                start_scale = 0.0
                
        # Skip if no scaling needed
        if abs(start_scale - end_scale) < 0.01:
            if config.on_complete:
                self.root.after(10, config.on_complete)
            return None

        prop_configs = {
            "width": (max(1, orig_w * start_scale), max(1, orig_w * end_scale)),
            "height": (max(1, orig_h * start_scale), max(1, orig_h * end_scale)),
            "x": (orig_x + (orig_w - orig_w * start_scale) / 2, 
                  orig_x + (orig_w - orig_w * end_scale) / 2),
            "y": (orig_y + (orig_h - orig_h * start_scale) / 2, 
                  orig_y + (orig_h - orig_h * end_scale) / 2),
        }
        return self._animate(widget, config, prop_configs, animation_id)
        
    def shake(self, widget: tk.Widget, intensity: float = 5.0, 
             frequency: float = 10.0, decay: bool = True,
             config: Optional[AnimationConfig] = None) -> Optional[str]:
        """Shake a widget with a decaying oscillation.
        
        Args:
            widget: The widget to animate
            intensity: Maximum shake distance in pixels
            frequency: Oscillations per second
            decay: Whether the shake should decay over time
            config: Optional animation configuration
            
        Returns:
            Animation ID if successful, None if widget is invalid
        """
        if not widget.winfo_exists():
            return None
            
        config = config or AnimationConfig(duration=400, easing="linear")
        self.stop_all_animations_for_widget(widget)
        animation_id = self._generate_id(f"{id(widget)}_shake")

        try:
            orig_x, orig_y = widget.winfo_x(), widget.winfo_y()
        except tk.TclError:
            return None

        def on_complete():
            if widget.winfo_exists():
                widget.place(x=orig_x, y=orig_y)
            if config and config.on_complete:
                config.on_complete()
        
        shake_config = AnimationConfig(
            duration=config.duration,
            easing=config.easing,
            delay=config.delay,
            on_complete=on_complete
        )

        def update_shake(progress: float):
            if not widget.winfo_exists():
                return
                
            decay_factor = (1 - progress) if decay else 1.0
            offset = intensity * decay_factor * math.sin(progress * math.pi * frequency * 2)
            widget.place(x=orig_x + int(round(offset)), y=orig_y)

        return self._custom_animate(shake_config, update_shake, animation_id) or animation_id

    def _custom_animate(self, config: AnimationConfig, update_func: Callable[[float], None], animation_id: str):
        """A generic animator for custom effects like shake."""
        start_time = time.time()
        easing_func = EASING_FUNCTIONS.get(config.easing, linear)

        def tick():
            if not self._active_animations.get(animation_id, False):
                return

            elapsed = (time.time() - start_time) * 1000
            if elapsed < config.delay:
                self.root.after(10, tick)
                return

            progress = min((elapsed - config.delay) / config.duration, 1.0)
            eased_progress = easing_func(progress)
            
            update_func(eased_progress)

            if progress < 1.0:
                self.root.after(16, tick)
            else:
                update_func(1.0) # Ensure final state
                if animation_id in self._active_animations:
                    del self._active_animations[animation_id]
                if config.on_complete:
                    config.on_complete()

        self._active_animations[animation_id] = True
        self.root.after(int(config.delay), tick)


# --- Micro-interactions ---
class MicroInteractions:
    """Micro-interactions for enhanced user experience."""
    
    def __init__(self, animations: AdvancedAnimations):
        self.animations = animations
    
    def add_hover_effect(self, widget: tk.Widget, effect: str = "scale"):
        """Add hover effect to a widget."""
        def on_enter(event):
            if effect == "scale":
                self.animations.scale(widget, 1.0, 1.1, AnimationConfig(duration=150))
        
        def on_leave(event):
            if effect == "scale":
                self.animations.scale(widget, 1.1, 1.0, AnimationConfig(duration=150))
        
        widget.bind("<Enter>", on_enter, "+")
        widget.bind("<Leave>", on_leave, "+")
    
    def add_click_effect(self, widget: tk.Widget, effect: str = "shake"):
        """Add click effect to a widget."""
        def on_click(event):
            if effect == "shake":
                self.animations.shake(widget)
            elif effect == "pulse":
                # A pulse can be a quick scale in and out
                self.animations.scale(widget, 1.0, 1.2, AnimationConfig(duration=100, on_complete=lambda:
                    self.animations.scale(widget, 1.2, 1.0, AnimationConfig(duration=100))))
        
        widget.bind("<Button-1>", on_click, "+")

# --- Utility Functions ---
def create_advanced_animations(root: tk.Tk) -> AdvancedAnimations:
    """Create a new advanced animations instance."""
    return AdvancedAnimations(root)

def create_micro_interactions(animations: AdvancedAnimations) -> MicroInteractions:
    """Create micro-interactions instance."""
    return MicroInteractions(animations)


# if __name__ == "__main__":
#     # --- Demo Window --- 
#     class AnimationDemo(tk.Tk):
#         def __init__(self):
#             super().__init__()
#             self.title("Advanced Animation Demo")
#             self.geometry("600x500")
#             self.configure(bg="#1e1e1e")
#
#             # Style for themed widgets
#             style = ttk.Style(self)
#             style.theme_use('clam')
#             style.configure("TButton", padding=10, font=("Segoe UI", 10, "bold"), background="#333", foreground="#fff")
#             style.map("TButton", background=[('active', '#444')])
#
#             # Animation and micro-interaction handlers
#             self.animator = create_advanced_animations(self)
#             self.interactions = create_micro_interactions(self.animator)
#
#             # Target widget for animations
#             self.target = tk.Label(self, text="Animate Me!", bg="#007acc", fg="white", font=("Segoe UI", 16, "bold"), relief="raised", borderwidth=2, padx=20, pady=10)
#             self.target.place(x=220, y=200)
#
#             # Animation control buttons
#             self._create_controls()
#
#         def _create_controls(self):
#             controls_frame = tk.Frame(self, bg="#1e1e1e")
#             controls_frame.pack(side="bottom", fill="x", pady=20, padx=20)
#
#             buttons = {
#                 "Fade In/Out": self.toggle_fade,
#                 "Slide Up/Down": self.toggle_slide,
#                 "Scale In/Out": self.toggle_scale,
#                 "Shake": self.trigger_shake
#             }
#
#             for i, (text, command) in enumerate(buttons.items()):
#                 btn = ttk.Button(controls_frame, text=text, command=command)
#                 btn.pack(side="left", expand=True, fill="x", padx=10)
#                 self.interactions.add_hover_effect(btn)
#                 self.interactions.add_click_effect(btn, effect="pulse")
#
#             self.fade_state = True
#             self.slide_state = True
#             self.scale_state = True
#
#         def toggle_fade(self):
#             start, end = (1.0, 0.0) if self.fade_state else (0.0, 1.0)
#             self.animator.fade(self.target, start=start, end=end, config=AnimationConfig(duration=500, easing="ease_in_out"))
#             self.fade_state = not self.fade_state
#
#         def toggle_slide(self):
#             start_pos, end_pos = ((220, 200), (220, -100)) if self.slide_state else ((220, -100), (220, 200))
#             self.animator.slide(self.target, start_pos, end_pos, config=AnimationConfig(duration=500, easing="elastic"))
#             self.slide_state = not self.slide_state
#
#         def toggle_scale(self):
#             start, end = (1.0, 0.0) if self.scale_state else (0.0, 1.0)
#             self.animator.scale(self.target, start_scale=start, end_scale=end, config=AnimationConfig(duration=400, easing="bounce"))
#             self.scale_state = not self.scale_state
#
#         def trigger_shake(self):
#             self.animator.shake(self.target, intensity=8)
#
#     # Run the demo
#     app = AnimationDemo()
#     app.mainloop()
