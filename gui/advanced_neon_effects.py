"""
Advanced Neon Effects System for PlayNexus Satellite Toolkit
Provides ultra-polished holographic reflections, energy fields, and advanced cyberpunk animations.
"""

import tkinter as tk
import math
import time
import random
from typing import List, Tuple, Optional

class HolographicReflection:
    """Creates stunning holographic reflection effects."""
    
    def __init__(self, canvas: tk.Canvas, x: int, y: int, width: int, height: int, color: str):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.animation_angle = 0
        
        # Create holographic elements
        self._create_holographic_elements()
        
        # Start animation
        self._start_animation()
    
    def _create_holographic_elements(self):
        """Create holographic reflection elements."""
        # Main reflection with enhanced glow
        self.reflection = self.canvas.create_rectangle(
            self.x + 3, self.y + 3, self.x + self.width - 3, self.y + self.height - 3,
            fill="",
            outline=self.color,
            width=2,
            tags="holographic"
        )
        
        # Corner reflections with enhanced styling
        self.corner_reflections = []
        corner_size = 10
        
        # Top-left corner
        self.corner_reflections.append(self.canvas.create_rectangle(
            self.x, self.y, self.x + corner_size, self.y + corner_size,
            fill=self.color,
            outline="",
            tags="holographic"
        ))
        
        # Top-right corner
        self.corner_reflections.append(self.canvas.create_rectangle(
            self.x + self.width - corner_size, self.y, self.x + self.width, self.y + corner_size,
            fill=self.color,
            outline="",
            tags="holographic"
        ))
        
        # Bottom-left corner
        self.corner_reflections.append(self.canvas.create_rectangle(
            self.x, self.y + self.height - corner_size, self.x + corner_size, self.y + self.height,
            fill=self.color,
            outline="",
            tags="holographic"
        ))
        
        # Bottom-right corner
        self.corner_reflections.append(self.canvas.create_rectangle(
            self.x + self.width - corner_size, self.y + self.height - corner_size, 
            self.x + self.width, self.y + self.height,
            fill=self.color,
            outline="",
            tags="holographic"
        ))
        
        # Add ultra-bright center dots
        for accent in self.corner_reflections:
            center_x = self.canvas.coords(accent)[0] + corner_size // 2
            center_y = self.canvas.coords(accent)[1] + corner_size // 2
            
            self.canvas.create_oval(
                center_x - 2, center_y - 2, center_x + 2, center_y + 2,
                fill="#ffffff",
                outline="",
                tags="holographic"
            )
    
    def _start_animation(self):
        """Start holographic animation."""
        self._animate_holographic()
    
    def _animate_holographic(self):
        """Animate holographic effects."""
        if not self.canvas.winfo_exists():
            return
        
        self.animation_angle += 0.18
        
        # Update reflection opacity with enhanced wave effect
        reflection_alpha = 0.3 + 0.4 * math.sin(self.animation_angle)
        reflection_color = self._adjust_color_intensity(self.color, reflection_alpha)
        self.canvas.itemconfig(self.reflection, outline=reflection_color)
        
        # Update corner reflections with enhanced pulsing
        for i, corner in enumerate(self.corner_reflections):
            corner_alpha = 0.4 + 0.6 * math.sin(self.animation_angle * 2.5 + i * 1.8)
            corner_color = self._adjust_color_intensity(self.color, corner_alpha)
            self.canvas.itemconfig(corner, fill=corner_color)
        
        # Schedule next animation
        self.canvas.after(35, self._animate_holographic)
    
    def _adjust_color_intensity(self, color: str, intensity: float) -> str:
        """Adjust color intensity for enhanced glow effects."""
        try:
            color = color.lstrip('#')
            if len(color) != 6:
                return color  # Return original color if invalid format
            
            r = int(color[0:2], 16)
            g = int(color[1:3], 16)
            b = int(color[2:4], 16)
            
            # Ensure intensity is within valid range
            intensity = max(0.0, min(2.0, intensity))
            
            r = min(255, max(0, int(r * intensity)))
            g = min(255, max(0, int(g * intensity)))
            b = min(255, max(0, int(b * intensity)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            # Return original color if parsing fails
            return color
    
    def destroy(self):
        """Clean up holographic elements."""
        self.canvas.delete("holographic")

class EnergyField:
    """Creates stunning energy field effects around elements."""
    
    def __init__(self, canvas: tk.Canvas, x: int, y: int, width: int, height: int, color: str):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.energy_level = 0.0
        self.animation_angle = 0
        
        # Create energy field elements
        self._create_energy_field()
        
        # Start animation
        self._start_animation()
    
    def _create_energy_field(self):
        """Create energy field visual elements."""
        # Outer energy ring with enhanced glow
        self.outer_ring = self.canvas.create_oval(
            self.x - 15, self.y - 15, 
            self.x + self.width + 15, self.y + self.height + 15,
            fill="",
            outline=self.color,
            width=3,
            tags="energy_field"
        )
        
        # Inner energy ring
        self.inner_ring = self.canvas.create_oval(
            self.x - 8, self.y - 8, 
            self.x + self.width + 8, self.y + self.height + 8,
            fill="",
            outline=self.color,
            width=2,
            tags="energy_field"
        )
        
        # Energy particles with enhanced styling
        self.energy_particles = []
        for _ in range(8):
            particle = self.canvas.create_oval(
                0, 0, 4, 4,
                fill=self.color,
                outline="",
                tags="energy_field"
            )
            self.energy_particles.append(particle)
        
        # Add energy wave lines
        self.energy_waves = []
        for _ in range(4):
            wave = self.canvas.create_line(
                0, 0, 0, 0,
                fill=self.color,
                width=2,
                tags="energy_field"
            )
            self.energy_waves.append(wave)
    
    def _start_animation(self):
        """Start energy field animation."""
        self._animate_energy_field()
    
    def _animate_energy_field(self):
        """Animate energy field effects."""
        if not self.canvas.winfo_exists():
            return
        
        self.animation_angle += 0.12
        
        # Update energy level with enhanced wave
        self.energy_level = 0.6 + 0.4 * math.sin(self.animation_angle)
        
        # Update outer ring with enhanced effects
        outer_alpha = 0.4 + 0.6 * self.energy_level
        outer_color = self._adjust_color_intensity(self.color, outer_alpha)
        self.canvas.itemconfig(self.outer_ring, outline=outer_color, 
                             width=int(3 + self.energy_level * 4))
        
        # Update inner ring
        inner_alpha = 0.6 + 0.4 * self.energy_level
        inner_color = self._adjust_color_intensity(self.color, inner_alpha)
        self.canvas.itemconfig(self.inner_ring, outline=inner_color)
        
        # Update energy particles with enhanced orbits
        for i, particle in enumerate(self.energy_particles):
            angle = self.animation_angle + i * math.pi / 4
            radius = 18 + self.energy_level * 12
            
            px = self.x + self.width // 2 + radius * math.cos(angle)
            py = self.y + self.height // 2 + radius * math.sin(angle)
            
            self.canvas.coords(particle, px - 2, py - 2, px + 2, py + 2)
            
            particle_alpha = 0.5 + 0.5 * math.sin(self.animation_angle * 4 + i)
            particle_color = self._adjust_color_intensity(self.color, particle_alpha)
            self.canvas.itemconfig(particle, fill=particle_color)
        
        # Update energy waves
        for i, wave in enumerate(self.energy_waves):
            wave_angle = self.animation_angle * 2 + i * math.pi / 2
            wave_radius = 25 + self.energy_level * 15
            
            start_x = self.x + self.width // 2 + (wave_radius - 10) * math.cos(wave_angle)
            start_y = self.y + self.height // 2 + (wave_radius - 10) * math.sin(wave_angle)
            end_x = self.x + self.width // 2 + (wave_radius + 10) * math.cos(wave_angle)
            end_y = self.y + self.height // 2 + (wave_radius + 10) * math.sin(wave_angle)
            
            self.canvas.coords(wave, start_x, start_y, end_x, end_y)
            
            wave_alpha = 0.3 + 0.7 * math.sin(self.animation_angle * 3 + i)
            wave_color = self._adjust_color_intensity(self.color, wave_alpha)
            self.canvas.itemconfig(wave, fill=wave_color)
        
        # Schedule next animation
        self.canvas.after(25, self._animate_energy_field)
    
    def _adjust_color_intensity(self, color: str, intensity: float) -> str:
        """Adjust color intensity for enhanced glow effects."""
        try:
            color = color.lstrip('#')
            if len(color) != 6:
                return color  # Return original color if invalid format
            
            r = int(color[0:2], 16)
            g = int(color[1:3], 16)
            b = int(color[2:4], 16)
            
            # Ensure intensity is within valid range
            intensity = max(0.0, min(2.0, intensity))
            
            r = min(255, max(0, int(r * intensity)))
            g = min(255, max(0, int(g * intensity)))
            b = min(255, max(0, int(b * intensity)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            # Return original color if parsing fails
            return color
    
    def destroy(self):
        """Clean up energy field elements."""
        self.canvas.delete("energy_field")

class DataStream:
    """Creates animated data stream effects."""
    
    def __init__(self, canvas: tk.Canvas, x: int, y: int, direction: str = "down"):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = random.uniform(3, 10)
        self.length = random.randint(25, 70)
        self.color = random.choice(["#00ffff", "#ff00ff", "#00ff00", "#ff6600", "#80ffff", "#ff80ff"])
        self.alpha = 1.0
        self.life = random.uniform(4, 10)
        self.max_life = self.life
        
        # Create data stream elements
        self._create_data_stream()
        
        # Start animation
        self._start_animation()
    
    def _create_data_stream(self):
        """Create visual data stream elements."""
        if self.direction == "down":
            self.stream_line = self.canvas.create_line(
                self.x, self.y, self.x, self.y + self.length,
                fill=self.color,
                width=3,
                tags="data_stream"
            )
            
            # Add enhanced data particles
            self.particles = []
            for i in range(6):
                particle = self.canvas.create_oval(
                    self.x - 3, self.y + i * self.length // 6,
                    self.x + 3, self.y + i * self.length // 6 + 6,
                    fill=self.color,
                    outline="",
                    tags="data_stream"
                )
                self.particles.append(particle)
        
        elif self.direction == "right":
            self.stream_line = self.canvas.create_line(
                self.x, self.y, self.x + self.length, self.y,
                fill=self.color,
                width=3,
                tags="data_stream"
            )
            
            # Add enhanced data particles
            self.particles = []
            for i in range(6):
                particle = self.canvas.create_oval(
                    self.x + i * self.length // 6, self.y - 3,
                    self.x + i * self.length // 6 + 6, self.y + 3,
                    fill=self.color,
                    outline="",
                    tags="data_stream"
                )
                self.particles.append(particle)
    
    def _start_animation(self):
        """Start data stream animation."""
        self._animate_data_stream()
    
    def _animate_data_stream(self):
        """Animate data stream effects."""
        if not self.canvas.winfo_exists():
            return
        
        # Update position
        if self.direction == "down":
            self.y += self.speed
            if self.y > self.canvas.winfo_height():
                self.y = -self.length
        else:
            self.x += self.speed
            if self.x > self.canvas.winfo_width():
                self.x = -self.length
        
        # Update life
        self.life -= 0.06
        self.alpha = self.life / self.max_life
        
        # Update visual elements
        if self.direction == "down":
            self.canvas.coords(self.stream_line, self.x, self.y, self.x, self.y + self.length)
            
            for i, particle in enumerate(self.particles):
                py = self.y + i * self.length // 6
                self.canvas.coords(particle, self.x - 3, py, self.x + 3, py + 6)
        else:
            self.canvas.coords(self.stream_line, self.x, self.y, self.x + self.length, self.y)
            
            for i, particle in enumerate(self.particles):
                px = self.x + i * self.length // 6
                self.canvas.coords(particle, px, self.y - 3, px + 6, self.y + 3)
        
        # Update opacity with enhanced effects
        stream_color = self._adjust_color_intensity(self.color, self.alpha)
        self.canvas.itemconfig(self.stream_line, fill=stream_color)
        
        for particle in self.particles:
            self.canvas.itemconfig(particle, fill=stream_color)
        
        # Schedule next animation
        self.canvas.after(25, self._animate_data_stream)
    
    def _adjust_color_intensity(self, color: str, intensity: float) -> str:
        """Adjust color intensity for enhanced glow effects."""
        try:
            color = color.lstrip('#')
            if len(color) != 6:
                return color  # Return original color if invalid format
            
            r = int(color[0:2], 16)
            g = int(color[1:3], 16)
            b = int(color[2:4], 16)
            
            # Ensure intensity is within valid range
            intensity = max(0.0, min(2.0, intensity))
            
            r = min(255, max(0, int(r * intensity)))
            g = min(255, max(0, int(g * intensity)))
            b = min(255, max(0, int(b * intensity)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            # Return original color if parsing fails
            return color
    
    def is_alive(self) -> bool:
        """Check if data stream is still alive."""
        return self.life > 0
    
    def destroy(self):
        """Clean up data stream elements."""
        self.canvas.delete("data_stream")

def create_holographic_reflection(canvas: tk.Canvas, x: int, y: int, width: int, height: int, color: str):
    """Create a holographic reflection effect."""
    return HolographicReflection(canvas, x, y, width, height, color)

def create_energy_field(canvas: tk.Canvas, x: int, y: int, width: int, height: int, color: str):
    """Create an energy field effect."""
    return EnergyField(canvas, x, y, width, height, color)

def create_data_stream(canvas: tk.Canvas, x: int, y: int, direction: str = "down"):
    """Create a data stream effect."""
    return DataStream(canvas, x, y, direction)
