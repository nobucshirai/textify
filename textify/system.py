"""
System monitoring and GPU/hardware detection module.

This module handles:
- System capability initialization (CUDA, GPU, ffprobe, etc.)
- GPU information retrieval
- Resource monitoring (CPU/GPU usage, power consumption)
"""

import logging
import threading
import time
import subprocess
from typing import Dict, Any

# Global state variables
gpu_available = False
pynvml_available = False
ffprobe_available = False
psutil_available = False
cuda_available = False
easyocr_available = False

# Module references
pynvml = None
psutil = None
easyocr = None


def initialize_system_checks(verbose=False):
    """
    Initialize system capability checks. This function checks for CUDA, GPU, ffprobe, etc.
    
    Args:
        verbose (bool): Whether to show detailed information messages
    """
    global gpu_available, pynvml_available, ffprobe_available, psutil_available, cuda_available
    global easyocr_available, easyocr, pynvml, psutil
    
    # Set log level based on verbose flag
    log_level = logging.INFO if verbose else logging.DEBUG
    
    # Check if CUDA is available in PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        # Detect CPU-only PyTorch builds (torch.version.cuda is None)
        if cuda_available and getattr(torch.version, 'cuda', None) is None:
            # Forge a more accurate “no CUDA” on CPU-only builds
            logging.log(log_level, "PyTorch built without CUDA support; CUDA disabled.")
            cuda_available = False
        if not cuda_available:
            logging.log(log_level, "CUDA is not available in PyTorch. Will use CPU for processing.")
    except ImportError:
        logging.log(log_level, "PyTorch not installed or cannot be imported. Will use CPU for processing.")
        cuda_available = False

    # Check if ffprobe is available
    try:
        result = subprocess.run(['ffprobe', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffprobe_available = result.returncode == 0
    except FileNotFoundError:
        ffprobe_available = False
    except Exception as e:
        logging.log(log_level, f"Error checking ffprobe: {str(e)}")
        ffprobe_available = False

    # Try to import psutil
    try:
        import psutil as _psutil
        psutil = _psutil
        psutil_available = True
    except ImportError:
        logging.log(log_level, "psutil not available. CPU monitoring will be disabled.")
        psutil = None
        psutil_available = False

    # Try to initialize GPU monitoring
    try:
        import pynvml as _pynvml
        pynvml = _pynvml
        pynvml.nvmlInit()
        pynvml_available = True
        gpu_available = True
    except (ImportError, Exception) as e:
        logging.log(log_level, f"pynvml not available or no NVIDIA GPU detected: {str(e)}")
        pynvml = None
        pynvml_available = False
        gpu_available = False
    
    # Try to import EasyOCR for document and image processing
    try:
        import easyocr as _easyocr
        easyocr = _easyocr
        easyocr_available = True
        logging.log(log_level, "EasyOCR is available for document and image processing.")
        
        # Also check for PyMuPDF (for PDF processing)
        try:
            import fitz
            logging.log(log_level, "PyMuPDF is available for PDF processing.")
        except ImportError:
            logging.log(log_level, "PyMuPDF not available. PDF processing will be limited.")
            
    except ImportError:
        logging.log(log_level, "EasyOCR not available. Document and image processing will be disabled.")
        easyocr = None
        easyocr_available = False


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed information about the GPU.
    
    Returns:
        Dict[str, Any]: Dictionary containing GPU information or empty dict if unavailable
    """
    gpu_info = {}
    
    if not pynvml_available:
        return gpu_info
    
    try:
        # Get device count
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info['device_count'] = device_count
        
        if device_count > 0:
            # Get information for the first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get name
            try:
                gpu_info['name'] = pynvml.nvmlDeviceGetName(handle)
            except pynvml.NVMLError:
                gpu_info['name'] = "Unknown"
            
            # Get memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_info['memory_total'] = mem_info.total / (1024**2)  # Convert to MB
                gpu_info['memory_used'] = mem_info.used / (1024**2)
                gpu_info['memory_free'] = mem_info.free / (1024**2)
            except pynvml.NVMLError:
                pass
            
            # Get driver version
            try:
                gpu_info['driver_version'] = pynvml.nvmlSystemGetDriverVersion()
            except pynvml.NVMLError:
                pass
            
            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_info['gpu_util'] = util.gpu
                gpu_info['memory_util'] = util.memory
            except pynvml.NVMLError:
                pass
            
            # Get power usage
            try:
                gpu_info['power_usage'] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
            except pynvml.NVMLError:
                pass
            
            # Get temperature
            try:
                gpu_info['temperature'] = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                pass
    
    except pynvml.NVMLError as e:
        logging.warning(f"Error getting GPU information: {e}")
    
    return gpu_info


def monitor_resources(stop_event: threading.Event, interval: int) -> None:
    """
    Monitors CPU and GPU resource usage by sampling at regular intervals.
    Calculates statistics and total energy consumption using trapezoid rule
    and outputs results at the end.

    Args:
        stop_event (threading.Event): Event to signal the monitoring to stop.
        interval (int): Interval in seconds to record the resources.
    """
    # Lists to store resource usage data
    cpu_usage_data = []
    gpu_usage_data = []
    gpu_power_data = []
    timestamps = []
    
    start_time = time.time()
    
    # Function to sample resources and append to data arrays
    def sample_resources():
        current_time = time.time() - start_time
        timestamps.append(current_time)
        
        # Monitor CPU if available
        if psutil_available:
            try:
                cpu_usage = psutil.cpu_percent(interval=None)
                cpu_usage_data.append(cpu_usage)
            except Exception as e:
                logging.warning(f"Error monitoring CPU: {str(e)}")
                # Append last known value or 0 if no data
                if len(cpu_usage_data) > 0:
                    cpu_usage_data.append(cpu_usage_data[-1])
                else:
                    cpu_usage_data.append(0)
        
        # Monitor GPU only if available
        if gpu_available:
            try:
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
                gpu_usage_data.append(gpu_util)
                
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000  # Convert milliwatts to watts
                gpu_power_data.append(gpu_power)
            except Exception as e:
                logging.warning(f"Error monitoring GPU: {str(e)}")
                # If we have previous data, just use the last value
                if len(gpu_usage_data) > 0:
                    gpu_usage_data.append(gpu_usage_data[-1])
                else:
                    gpu_usage_data.append(0)
                
                if len(gpu_power_data) > 0:
                    gpu_power_data.append(gpu_power_data[-1])
                else:
                    gpu_power_data.append(0)
    
    # Take initial sample immediately
    sample_resources()
    logging.debug("Initial resource sample taken")
    
    # Sample resource usage until stop event is set
    while not stop_event.is_set():
        try:
            # Wait for interval or until stop_event is set
            # This is more precise than time.sleep()
            if stop_event.wait(interval):
                # If wait returns True, the event was set
                break
                
            # Take another sample
            sample_resources()
            
        except KeyboardInterrupt:
            logging.info("Monitoring interrupted by user")
            break
    
    # Calculate statistics and total energy consumption
    logging.info("Resource usage statistics:")
    
    # CPU statistics
    if cpu_usage_data:
        cpu_min = min(cpu_usage_data)
        cpu_max = max(cpu_usage_data)
        cpu_avg = sum(cpu_usage_data) / len(cpu_usage_data)
        logging.info(f"CPU Usage: Avg={cpu_avg:.2f}%, Min={cpu_min:.2f}%, Max={cpu_max:.2f}%")
    else:
        logging.info("CPU monitoring disabled")
    
    # GPU statistics
    if gpu_usage_data:
        gpu_min = min(gpu_usage_data)
        gpu_max = max(gpu_usage_data)
        gpu_avg = sum(gpu_usage_data) / len(gpu_usage_data)
        logging.info(f"GPU Usage: Avg={gpu_avg:.2f}%, Min={gpu_min:.2f}%, Max={gpu_max:.2f}%")
        
        if gpu_power_data:
            power_min = min(gpu_power_data)
            power_max = max(gpu_power_data)
            power_avg = sum(gpu_power_data) / len(gpu_power_data)
            logging.info(f"GPU Power: Avg={power_avg:.2f}W, Min={power_min:.2f}W, Max={power_max:.2f}W")
            
            # Calculate total energy consumption using trapezoid rule
            if len(gpu_power_data) > 1 and len(timestamps) == len(gpu_power_data):
                # Energy in watt-seconds (joules)
                energy_joules = 0
                for i in range(1, len(timestamps)):
                    dt = timestamps[i] - timestamps[i-1]
                    avg_power = (gpu_power_data[i] + gpu_power_data[i-1]) / 2
                    energy_joules += avg_power * dt
                
                # Convert to watt-hours
                energy_wh = energy_joules / 3600
                logging.info(f"Total GPU Energy Consumption: {energy_wh:.4f} Wh")
    else:
        logging.info("GPU monitoring disabled")
