"""
Performance Optimization Module for PlayNexus Satellite Toolkit
Provides GPU acceleration, parallel processing, and memory management capabilities.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
import queue
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
import logging
from functools import partial, wraps
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import joblib
import numba
from numba import jit, prange, cuda
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except (ImportError, OSError):
    CUPY_AVAILABLE = False
    cp = None

from .error_handling import PlayNexusLogger, ValidationError, ProcessingError
from .config import ConfigManager
from .progress_tracker import ProgressTracker, track_progress

logger = PlayNexusLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization for satellite imagery processing."""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize the performance optimizer."""
        self.config = config or ConfigManager()
        self.logger = PlayNexusLogger(__name__)
        self.progress_tracker = ProgressTracker()
        self._setup_optimization_config()
        self._check_available_acceleration()
    
    def _setup_optimization_config(self):
        """Setup performance optimization configuration."""
        self.max_workers = min(cpu_count(), 8)  # Limit to prevent system overload
        self.chunk_size = 1024  # Default chunk size for parallel processing
        self.memory_limit = '4GB'  # Memory limit for Dask
        self.gpu_memory_fraction = 0.8  # GPU memory usage limit
        self.enable_gpu = True
        self.enable_dask = True
        self.enable_numba = True
    
    def _check_available_acceleration(self):
        """Check what acceleration methods are available."""
        self.acceleration_methods = {
            'gpu_cupy': False,
            'gpu_numba': False,
            'dask': False,
            'numba_cpu': False,
            'multiprocessing': True,  # Always available
            'threading': True  # Always available
        }
        
        # Check GPU availability
        try:
            if self.enable_gpu:
                # Check CuPy (NVIDIA GPU)
                if CUPY_AVAILABLE and cp is not None:
                    try:
                        cp.cuda.Device(0)
                        self.acceleration_methods['gpu_cupy'] = True
                        self.logger.info("CuPy GPU acceleration available")
                    except:
                        self.logger.info("CuPy installed but CUDA not available - GPU acceleration disabled")
                        self.acceleration_methods['gpu_cupy'] = False
                
                # Check Numba CUDA
                try:
                    if cuda.is_available():
                        self.acceleration_methods['gpu_numba'] = True
                        self.logger.info("Numba CUDA acceleration available")
                except:
                    pass
        
        except Exception as e:
            self.logger.warning(f"GPU acceleration check failed: {e}")
        
        # Check Dask
        try:
            if self.enable_dask:
                self.acceleration_methods['dask'] = True
                self.logger.info("Dask parallel processing available")
        except Exception as e:
            self.logger.warning(f"Dask check failed: {e}")
        
        # Check Numba CPU
        try:
            if self.enable_numba:
                self.acceleration_methods['numba_cpu'] = True
                self.logger.info("Numba CPU acceleration available")
        except Exception as e:
            self.logger.warning(f"Numba CPU check failed: {e}")
    
    def get_available_methods(self) -> Dict[str, bool]:
        """Get available acceleration methods."""
        return self.acceleration_methods.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimization decisions."""
        info = {
            'cpu_count': cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if Path('/').exists() else 0
        }
        
        # GPU info if available
        if self.acceleration_methods['gpu_cupy']:
            try:
                gpu_memory = cp.cuda.runtime.memGetInfo()
                info['gpu_memory_total'] = gpu_memory[1]
                info['gpu_memory_free'] = gpu_memory[0]
                info['gpu_memory_percent'] = (gpu_memory[0] / gpu_memory[1]) * 100
            except:
                pass
        
        return info
    
    @track_progress("Parallel processing")
    def parallel_process(
        self,
        data: np.ndarray,
        func: Callable,
        n_workers: int = None,
        method: str = 'multiprocessing',
        chunk_size: int = None,
        **kwargs
    ) -> np.ndarray:
        """Process data in parallel using various methods."""
        n_workers = n_workers or self.max_workers
        chunk_size = chunk_size or self.chunk_size
        
        if method == 'multiprocessing':
            return self._multiprocessing_process(data, func, n_workers, chunk_size, **kwargs)
        elif method == 'threading':
            return self._threading_process(data, func, n_workers, chunk_size, **kwargs)
        elif method == 'dask' and self.acceleration_methods['dask']:
            return self._dask_process(data, func, n_workers, chunk_size, **kwargs)
        else:
            self.logger.warning(f"Method {method} not available, falling back to multiprocessing")
            return self._multiprocessing_process(data, func, n_workers, chunk_size, **kwargs)
    
    def _multiprocessing_process(
        self,
        data: np.ndarray,
        func: Callable,
        n_workers: int,
        chunk_size: int,
        **kwargs
    ) -> np.ndarray:
        """Process data using multiprocessing."""
        # Prepare chunks
        chunks = self._prepare_chunks(data, chunk_size)
        
        # Process chunks in parallel
        with Pool(processes=n_workers) as pool:
            results = pool.map(partial(func, **kwargs), chunks)
        
        # Combine results
        return self._combine_chunks(results, data.shape)
    
    def _threading_process(
        self,
        data: np.ndarray,
        func: Callable,
        n_workers: int,
        chunk_size: int,
        **kwargs
    ) -> np.ndarray:
        """Process data using threading (good for I/O bound tasks)."""
        # Prepare chunks
        chunks = self._prepare_chunks(data, chunk_size)
        
        # Process chunks using threads
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(func, chunk, **kwargs) for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]
        
        # Combine results
        return self._combine_chunks(results, data.shape)
    
    def _dask_process(
        self,
        data: np.ndarray,
        func: Callable,
        n_workers: int,
        chunk_size: int,
        **kwargs
    ) -> np.ndarray:
        """Process data using Dask for out-of-memory processing."""
        # Convert to Dask array
        dask_data = da.from_array(data, chunks=chunk_size)
        
        # Apply function
        result = dask_data.map_blocks(func, **kwargs)
        
        # Compute result
        return result.compute()
    
    def _prepare_chunks(self, data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Prepare data chunks for parallel processing."""
        chunks = []
        
        if data.ndim == 2:
            # 2D data - split into rows
            for i in range(0, data.shape[0], chunk_size):
                end_i = min(i + chunk_size, data.shape[0])
                chunks.append(data[i:end_i, :])
        
        elif data.ndim == 3:
            # 3D data - split into 2D slices
            for i in range(0, data.shape[0], chunk_size):
                end_i = min(i + chunk_size, data.shape[0])
                chunks.append(data[i:end_i, :, :])
        
        else:
            # Higher dimensional data - flatten and split
            flat_data = data.flatten()
            for i in range(0, len(flat_data), chunk_size):
                end_i = min(i + chunk_size, len(flat_data))
                chunks.append(flat_data[i:end_i])
        
        return chunks
    
    def _combine_chunks(self, results: List[np.ndarray], original_shape: Tuple[int, ...]) -> np.ndarray:
        """Combine processed chunks back into original shape."""
        if not results:
            return np.array([])
        
        # Handle different result types
        if isinstance(results[0], np.ndarray):
            if results[0].ndim == 1:
                # 1D results - concatenate
                combined = np.concatenate(results)
                # Try to reshape to original shape
                try:
                    return combined.reshape(original_shape)
                except:
                    return combined
            else:
                # Multi-dimensional results - concatenate along first axis
                return np.concatenate(results, axis=0)
        else:
            # Non-array results - convert to array
            return np.array(results)
    
    @track_progress("GPU acceleration")
    def gpu_accelerate(
        self,
        data: np.ndarray,
        func: Callable,
        method: str = 'cupy',
        **kwargs
    ) -> np.ndarray:
        """Accelerate processing using GPU."""
        if method == 'cupy' and self.acceleration_methods['gpu_cupy']:
            return self._cupy_accelerate(data, func, **kwargs)
        elif method == 'numba_cuda' and self.acceleration_methods['gpu_numba']:
            return self._numba_cuda_accelerate(data, func, **kwargs)
        else:
            self.logger.warning(f"GPU method {method} not available, falling back to CPU")
            return func(data, **kwargs)
    
    def _cupy_accelerate(
        self,
        data: np.ndarray,
        func: Callable,
        **kwargs
        ) -> np.ndarray:
        """Accelerate using CuPy (NVIDIA GPU)."""
        if not CUPY_AVAILABLE or cp is None:
            self.logger.warning("CuPy not available, falling back to CPU")
            return func(data, **kwargs)
            
        try:
            # Transfer data to GPU
            gpu_data = cp.asarray(data)
            
            # Apply function (assuming it's CuPy-compatible)
            if hasattr(func, '__name__') and func.__name__ in dir(cp):
                # Use CuPy equivalent function
                cupy_func = getattr(cp, func.__name__)
                result = cupy_func(gpu_data, **kwargs)
            else:
                # Try to use the function directly
                result = func(gpu_data, **kwargs)
            
            # Transfer result back to CPU
            return cp.asnumpy(result)
            
        except Exception as e:
            self.logger.warning(f"CuPy acceleration failed: {e}, falling back to CPU")
            return func(data, **kwargs)
    
    def _numba_cuda_accelerate(
        self,
        data: np.ndarray,
        func: Callable,
        **kwargs
    ) -> np.ndarray:
        """Accelerate using Numba CUDA."""
        try:
            # This would require the function to be CUDA-compatible
            # For now, return CPU version
            self.logger.warning("Numba CUDA acceleration not fully implemented")
            return func(data, **kwargs)
            
        except Exception as e:
            self.logger.warning(f"Numba CUDA acceleration failed: {e}, falling back to CPU")
            return func(data, **kwargs)
    
    @track_progress("Memory optimization")
    def optimize_memory(
        self,
        data: np.ndarray,
        target_dtype: np.dtype = None,
        compression: bool = False,
        chunking: bool = True
    ) -> np.ndarray:
        """Optimize memory usage of data arrays."""
        original_dtype = data.dtype
        original_size = data.nbytes
        
        # Determine optimal dtype
        if target_dtype is None:
            target_dtype = self._get_optimal_dtype(data)
        
        # Convert dtype if beneficial
        if target_dtype != original_dtype:
            try:
                data = data.astype(target_dtype)
                self.logger.info(f"Converted dtype from {original_dtype} to {target_dtype}")
            except Exception as e:
                self.logger.warning(f"Dtype conversion failed: {e}")
        
        # Apply compression if requested
        if compression:
            data = self._compress_data(data)
        
        # Apply chunking if requested
        if chunking and data.ndim > 1:
            data = self._chunk_data(data)
        
        # Report memory savings
        new_size = data.nbytes
        savings = (original_size - new_size) / original_size * 100
        self.logger.info(f"Memory optimization: {savings:.1f}% reduction ({original_size/1e6:.1f}MB -> {new_size/1e6:.1f}MB)")
        
        return data
    
    def _get_optimal_dtype(self, data: np.ndarray) -> np.dtype:
        """Determine optimal data type for memory efficiency."""
        if data.dtype.kind in ['f', 'c']:  # Float or complex
            if data.dtype == np.float64:
                # Check if float32 precision is sufficient
                if np.allclose(data, data.astype(np.float32), rtol=1e-5):
                    return np.float32
            elif data.dtype == np.float32:
                # Check if float16 precision is sufficient
                if np.allclose(data, data.astype(np.float16), rtol=1e-3):
                    return np.float16
        
        elif data.dtype.kind in ['i', 'u']:  # Integer or unsigned integer
            min_val = np.min(data)
            max_val = np.max(data)
            
            if data.dtype.kind == 'i':  # Signed integer
                if min_val >= -128 and max_val <= 127:
                    return np.int8
                elif min_val >= -32768 and max_val <= 32767:
                    return np.int16
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    return np.int32
            else:  # Unsigned integer
                if max_val <= 255:
                    return np.uint8
                elif max_val <= 65535:
                    return np.uint16
                elif max_val <= 4294967295:
                    return np.uint32
        
        return data.dtype
    
    def _compress_data(self, data: np.ndarray) -> np.ndarray:
        """Apply data compression techniques."""
        # For now, return original data
        # In production, you might implement actual compression
        return data
    
    def _chunk_data(self, data: np.ndarray) -> np.ndarray:
        """Apply data chunking for better memory management."""
        # For now, return original data
        # In production, you might implement actual chunking
        return data
    
    def create_dask_cluster(
        self,
        n_workers: int = None,
        memory_limit: str = None,
        local_directory: str = None
    ) -> Client:
        """Create a Dask cluster for distributed processing."""
        if not self.acceleration_methods['dask']:
            raise ValidationError("Dask is not available")
        
        n_workers = n_workers or self.max_workers
        memory_limit = memory_limit or self.memory_limit
        local_directory = local_directory or str(Path.home() / '.playnexus' / 'dask-worker-space')
        
        # Create local cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            memory_limit=memory_limit,
            local_directory=local_directory,
            processes=True
        )
        
        # Create client
        client = Client(cluster)
        
        self.logger.info(f"Created Dask cluster with {n_workers} workers")
        return client
    
    def monitor_performance(self, func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                self.logger.info(f"Function {func.__name__} executed in {execution_time:.2f}s, "
                               f"memory change: {memory_used/1e6:.1f}MB")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                self.logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s: {e}")
                raise
        
        return wrapper
    
    def batch_process(
        self,
        data_list: List[np.ndarray],
        func: Callable,
        batch_size: int = 10,
        **kwargs
    ) -> List[np.ndarray]:
        """Process data in batches to manage memory."""
        results = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for item in batch:
                try:
                    result = func(item, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to process item {i}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Force garbage collection
            gc.collect()
            
            # Log progress
            progress = min((i + batch_size) / len(data_list) * 100, 100)
            self.logger.info(f"Batch processing progress: {progress:.1f}%")
        
        return results
    
    def optimize_workflow(
        self,
        workflow_steps: List[Callable],
        data: np.ndarray,
        optimization_level: str = 'balanced',
        **kwargs
    ) -> np.ndarray:
        """Optimize a workflow of processing steps."""
        optimization_strategies = {
            'memory': self._memory_optimized_workflow,
            'speed': self._speed_optimized_workflow,
            'balanced': self._balanced_optimized_workflow
        }
        
        strategy = optimization_strategies.get(optimization_level, self._balanced_optimized_workflow)
        return strategy(workflow_steps, data, **kwargs)
    
    def _memory_optimized_workflow(
        self,
        workflow_steps: List[Callable],
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Memory-optimized workflow execution."""
        result = data
        
        for i, step in enumerate(workflow_steps):
            self.logger.info(f"Executing workflow step {i+1}/{len(workflow_steps)}: {step.__name__}")
            
            # Execute step
            result = step(result, **kwargs)
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            memory_usage = psutil.Process().memory_info().rss / 1e6
            self.logger.info(f"Memory usage after step {i+1}: {memory_usage:.1f}MB")
        
        return result
    
    def _speed_optimized_workflow(
        self,
        workflow_steps: List[Callable],
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Speed-optimized workflow execution."""
        # Use parallel processing where possible
        result = data
        
        for i, step in enumerate(workflow_steps):
            self.logger.info(f"Executing workflow step {i+1}/{len(workflow_steps)}: {step.__name__}")
            
            # Check if step can be parallelized
            if hasattr(step, '__name__') and any(keyword in step.__name__.lower() 
                                                for keyword in ['filter', 'transform', 'process']):
                result = self.parallel_process(result, step, method='multiprocessing', **kwargs)
            else:
                result = step(result, **kwargs)
        
        return result
    
    def _balanced_optimized_workflow(
        self,
        workflow_steps: List[Callable],
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Balanced optimization workflow execution."""
        # Combine memory and speed optimizations
        result = data
        
        for i, step in enumerate(workflow_steps):
            self.logger.info(f"Executing workflow step {i+1}/{len(workflow_steps)}: {step.__name__}")
            
            # Execute step with monitoring
            monitored_step = self.monitor_performance(step)
            result = monitored_step(result, **kwargs)
            
            # Optimize memory if needed
            if result.nbytes > 1e9:  # 1GB threshold
                result = self.optimize_memory(result, chunking=True)
            
            # Periodic garbage collection
            if i % 3 == 0:  # Every 3 steps
                gc.collect()
        
        return result

# Convenience functions
def parallel_process_satellite_data(
    data: np.ndarray,
    func: Callable,
    n_workers: int = None,
    method: str = 'multiprocessing',
    **kwargs
) -> np.ndarray:
    """Convenience function for parallel processing."""
    optimizer = PerformanceOptimizer()
    return optimizer.parallel_process(data, func, n_workers, method, **kwargs)

def gpu_accelerate_satellite_processing(
    data: np.ndarray,
    func: Callable,
    method: str = 'cupy',
    **kwargs
) -> np.ndarray:
    """Convenience function for GPU acceleration."""
    optimizer = PerformanceOptimizer()
    return optimizer.gpu_accelerate(data, func, method, **kwargs)

def optimize_satellite_workflow(
    workflow_steps: List[Callable],
    data: np.ndarray,
    optimization_level: str = 'balanced',
    **kwargs
) -> np.ndarray:
    """Convenience function for workflow optimization."""
    optimizer = PerformanceOptimizer()
    return optimizer.optimize_workflow(workflow_steps, data, optimization_level, **kwargs)
