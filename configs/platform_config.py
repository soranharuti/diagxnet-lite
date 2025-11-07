"""
Cross-platform configuration for DiagXNet-Lite
Handles OS-specific differences between Mac, Windows, and Linux
"""

import platform
import torch
from pathlib import Path

def get_os_type():
    """Get the operating system type"""
    system = platform.system()
    return system  # Returns 'Darwin' for Mac, 'Windows' for Windows, 'Linux' for Linux

def get_optimal_num_workers():
    """
    Get optimal number of DataLoader workers based on OS
    
    Windows has issues with multiprocessing in PyTorch DataLoader,
    so we use fewer workers or 0 for Windows.
    
    Returns:
        int: Recommended number of workers
    """
    system = platform.system()
    
    if system == 'Windows':
        # Windows has multiprocessing issues, use 0 or 1
        return 0
    elif system == 'Darwin':  # Mac
        # Mac works well with multiple workers
        return 4
    else:  # Linux
        # Linux typically works best with multiple workers
        return 4

def get_device():
    """
    Get the best available device for training
    Works across Mac (MPS), Windows (CUDA), and Linux (CUDA)
    
    Returns:
        torch.device: Best available device
    """
    # Check for CUDA (NVIDIA GPU) - Windows/Linux
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[OK] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    # Check for MPS (Apple Silicon) - Mac
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[OK] Using Apple Silicon MPS")
        return device
    
    # Fallback to CPU
    device = torch.device("cpu")
    print(f"[OK] Using CPU (no GPU acceleration available)")
    return device

def configure_multiprocessing():
    """
    Configure multiprocessing for Windows compatibility
    Must be called at the start of training scripts
    """
    import multiprocessing
    system = platform.system()
    
    if system == 'Windows':
        # Windows requires 'spawn' method for PyTorch multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Disable OpenMP on Windows to avoid conflicts
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
    
    return system

def get_path_separator():
    """Get the correct path separator for the OS"""
    return '\\' if platform.system() == 'Windows' else '/'

def print_system_info():
    """Print system and hardware information"""
    system = platform.system()
    print("="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Operating System: {system}")
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Device info
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS (Apple Silicon): Yes")
    else:
        print(f"GPU Acceleration: Not available (using CPU)")
    
    print(f"Optimal DataLoader Workers: {get_optimal_num_workers()}")
    print("="*70)
    print()

# Configuration constants
OS_TYPE = get_os_type()
OPTIMAL_NUM_WORKERS = get_optimal_num_workers()
DEVICE = get_device()

# Platform-specific settings
PLATFORM_SETTINGS = {
    'num_workers': OPTIMAL_NUM_WORKERS,
    'pin_memory': torch.cuda.is_available(),  # Only useful with CUDA
    'persistent_workers': OPTIMAL_NUM_WORKERS > 0,  # Only if using workers
}

if __name__ == "__main__":
    print_system_info()
    print(f"\nPlatform Settings:")
    for key, value in PLATFORM_SETTINGS.items():
        print(f"  {key}: {value}")

