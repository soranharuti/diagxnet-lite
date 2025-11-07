"""
Platform utilities for cross-platform compatibility
"""

import platform
import torch
from torch.utils.data import DataLoader

def get_dataloader_kwargs():
    """
    Get platform-optimized DataLoader kwargs
    
    Returns:
        dict: Keyword arguments for DataLoader
    """
    system = platform.system()
    
    kwargs = {
        'pin_memory': torch.cuda.is_available(),
    }
    
    if system == 'Windows':
        # Windows: Use 0 workers to avoid multiprocessing issues
        kwargs['num_workers'] = 0
        kwargs['persistent_workers'] = False
    else:
        # Mac/Linux: Can use multiple workers
        kwargs['num_workers'] = 4
        kwargs['persistent_workers'] = True
    
    return kwargs

def create_dataloader(dataset, batch_size, shuffle=False, drop_last=False, **override_kwargs):
    """
    Create a DataLoader with platform-optimized settings
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        **override_kwargs: Override default settings
    
    Returns:
        DataLoader: Configured DataLoader
    """
    # Get platform-specific defaults
    kwargs = get_dataloader_kwargs()
    
    # Add common arguments
    kwargs.update({
        'batch_size': batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last,
    })
    
    # Apply any overrides
    kwargs.update(override_kwargs)
    
    return DataLoader(dataset, **kwargs)

def setup_windows_compatibility():
    """
    Setup Windows-specific compatibility fixes
    Call this at the start of training scripts
    """
    import multiprocessing
    import os
    
    system = platform.system()
    
    if system == 'Windows':
        # Set spawn method for Windows
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Limit OpenMP threads
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('MKL_NUM_THREADS', '1')
        
        print("[OK] Windows compatibility mode enabled")
        print("  - Using spawn for multiprocessing")
        print("  - DataLoader workers set to 0")
    
    return system

if __name__ == "__main__":
    print("Platform Detection Test")
    print("="*50)
    print(f"Operating System: {platform.system()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"\nDataLoader Settings:")
    kwargs = get_dataloader_kwargs()
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

