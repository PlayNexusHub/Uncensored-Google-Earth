"""
PlayNexus Satellite Toolkit - Progress Tracking Module
Provides progress indicators, cancellation support, and user feedback for long-running operations.
"""

import time
import threading
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass
class ProgressStep:
    """Represents a single step in a multi-step process."""
    name: str
    description: str
    weight: float = 1.0
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = None


class ProgressTracker:
    """Tracks progress of satellite imagery processing operations."""
    
    def __init__(self, operation_name: str = "Satellite Processing"):
        self.operation_name = operation_name
        self.steps: List[ProgressStep] = []
        self.current_step_index: int = -1
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.cancelled: bool = False
        self._lock = threading.Lock()
        
        # Progress callback
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        
        # Estimated time remaining
        self.eta_seconds: Optional[float] = None
    
    def add_step(self, name: str, description: str, weight: float = 1.0) -> int:
        """Add a processing step."""
        with self._lock:
            step = ProgressStep(name=name, description=description, weight=weight)
            self.steps.append(step)
            return len(self.steps) - 1
    
    def start_operation(self):
        """Start the overall operation."""
        with self._lock:
            self.start_time = time.time()
            self.cancelled = False
            self._notify_progress()
    
    def start_step(self, step_index: int):
        """Start a specific step."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                self.current_step_index = step_index
                self.steps[step_index].status = "running"
                self.steps[step_index].start_time = time.time()
                self._notify_progress()
    
    def complete_step(self, step_index: int, details: Dict[str, Any] = None):
        """Mark a step as completed."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                self.steps[step_index].status = "completed"
                self.steps[step_index].end_time = time.time()
                if details:
                    self.steps[step_index].details = details
                self._notify_progress()
    
    def fail_step(self, step_index: int, error: str):
        """Mark a step as failed."""
        with self._lock:
            if 0 <= step_index < len(self.steps):
                self.steps[step_index].status = "failed"
                self.steps[step_index].end_time = time.time()
                self.steps[step_index].error = error
                self._notify_progress()
    
    def cancel_operation(self):
        """Cancel the operation."""
        with self._lock:
            self.cancelled = True
            if self.current_step_index >= 0:
                self.steps[self.current_step_index].status = "cancelled"
            self._notify_progress()
    
    def complete_operation(self):
        """Mark the operation as completed."""
        with self._lock:
            self.end_time = time.time()
            self._notify_progress()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress information."""
        with self._lock:
            if not self.steps:
                return {"progress": 0.0, "status": "No steps defined"}
            
            total_weight = sum(step.weight for step in self.steps)
            completed_weight = sum(
                step.weight for step in self.steps 
                if step.status == "completed"
            )
            
            progress_percent = (completed_weight / total_weight) * 100 if total_weight > 0 else 0
            
            # Calculate ETA
            if self.start_time and progress_percent > 0:
                elapsed = time.time() - self.start_time
                if progress_percent < 100:
                    self.eta_seconds = (elapsed / progress_percent) * (100 - progress_percent)
                else:
                    self.eta_seconds = 0
            
            current_step = self.steps[self.current_step_index] if self.current_step_index >= 0 else None
            
            return {
                "operation_name": self.operation_name,
                "progress_percent": progress_percent,
                "completed_weight": completed_weight,
                "total_weight": total_weight,
                "current_step": current_step.name if current_step else None,
                "current_step_description": current_step.description if current_step else None,
                "current_step_status": current_step.status if current_step else None,
                "steps": [
                    {
                        "name": step.name,
                        "description": step.description,
                        "status": step.status,
                        "weight": step.weight,
                        "error": step.error
                    }
                    for step in self.steps
                ],
                "start_time": self.start_time,
                "end_time": self.end_time,
                "eta_seconds": self.eta_seconds,
                "cancelled": self.cancelled,
                "elapsed_seconds": time.time() - self.start_time if self.start_time else 0
            }
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def _notify_progress(self):
        """Notify progress callback if set."""
        if self.progress_callback:
            try:
                progress_info = self.get_progress()
                self.progress_callback(progress_info)
            except Exception as e:
                # Don't let callback errors break progress tracking
                print(f"Progress callback error: {e}", file=sys.stderr)


class ConsoleProgressBar:
    """Simple console-based progress bar."""
    
    def __init__(self, width: int = 50):
        self.width = width
        self.last_update = 0
        self.update_interval = 0.1  # Update every 100ms
    
    def update(self, progress_info: Dict[str, Any]):
        """Update the progress bar display."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Clear line
        print("\r", end="", flush=True)
        
        # Progress bar
        progress = progress_info.get("progress_percent", 0)
        bar_width = int((progress / 100) * self.width)
        bar = "█" * bar_width + "░" * (self.width - bar_width)
        
        # Status info
        current_step = progress_info.get("current_step", "Unknown")
        eta = progress_info.get("eta_seconds")
        eta_str = f"ETA: {self._format_time(eta)}" if eta else ""
        
        # Display
        print(f"{current_step} |{bar}| {progress:.1f}% {eta_str}", end="", flush=True)
    
    def complete(self):
        """Mark progress as complete."""
        print()  # New line after progress bar
    
    def _format_time(self, seconds: Optional[float]) -> str:
        """Format time in human-readable format."""
        if seconds is None:
            return ""
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"


class FileProgressLogger:
    """Logs progress to a file for persistent tracking."""
    
    def __init__(self, log_file: Union[str, Path]):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def update(self, progress_info: Dict[str, Any]):
        """Log progress information to file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "progress_percent": progress_info.get("progress_percent", 0),
            "current_step": progress_info.get("current_step", "Unknown"),
            "status": progress_info.get("current_step_status", "Unknown"),
            "eta_seconds": progress_info.get("eta_seconds")
        }
        
        try:
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp} - {log_entry}\n")
        except Exception as e:
            print(f"Failed to log progress: {e}", file=sys.stderr)


def track_progress(operation_name: str = "Satellite Processing"):
    """Decorator for tracking progress of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = ProgressTracker(operation_name)
            
            # Add progress callback if available
            if "progress_callback" in kwargs:
                tracker.set_progress_callback(kwargs.pop("progress_callback"))
            
            try:
                tracker.start_operation()
                result = func(*args, **kwargs)
                tracker.complete_operation()
                return result
            except Exception as e:
                if tracker.current_step_index >= 0:
                    tracker.fail_step(tracker.current_step_index, str(e))
                raise
            finally:
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_progress())
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test progress tracking
    tracker = ProgressTracker("Test Operation")
    
    # Add steps
    tracker.add_step("Download", "Downloading satellite data", weight=2.0)
    tracker.add_step("Process", "Processing imagery", weight=3.0)
    tracker.add_step("Analyze", "Analyzing anomalies", weight=2.0)
    tracker.add_step("Save", "Saving results", weight=1.0)
    
    # Set up console progress bar
    progress_bar = ConsoleProgressBar()
    tracker.set_progress_callback(progress_bar.update)
    
    # Simulate processing
    tracker.start_operation()
    
    for i in range(len(tracker.steps)):
        tracker.start_step(i)
        time.sleep(1)  # Simulate work
        tracker.complete_step(i, {"processed_pixels": 1000 * (i + 1)})
    
    tracker.complete_operation()
    progress_bar.complete()
    
    print("Progress tracking test completed!")
