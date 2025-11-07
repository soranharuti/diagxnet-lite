# Cross-Platform Setup Guide

This guide ensures DiagXNet-Lite works seamlessly on **Windows**, **Mac**, and **Linux**.

---

## 🖥️ Platform Compatibility

### Supported Platforms
- ✅ **Windows 10/11** (CPU or NVIDIA GPU)
- ✅ **macOS** (Intel or Apple Silicon with MPS)
- ✅ **Linux** (CPU or NVIDIA GPU)

### Hardware Acceleration
- **Windows/Linux**: CUDA (NVIDIA GPU)
- **macOS**: MPS (Apple Silicon M1/M2/M3)
- **All**: CPU fallback

---

## 📦 Installation

### Step 1: Clone Repository

```bash
# All platforms
git clone https://github.com/yourusername/diagxnet-lite.git
cd diagxnet-lite
```

### Step 2: Create Virtual Environment

#### Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
```

#### Windows (Command Prompt):
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

#### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

#### All Platforms:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows with NVIDIA GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Mac with Apple Silicon:
```bash
# PyTorch with MPS support (included in default install)
pip install torch torchvision torchaudio
```

---

## ✅ Verify Installation

Run the cross-platform compatibility test:

```bash
python test_cross_platform.py
```

This will check:
- ✓ All dependencies installed
- ✓ PyTorch configured correctly
- ✓ GPU/MPS acceleration available
- ✓ DataLoader working properly
- ✓ Platform-specific settings correct

---

## 🔧 Platform-Specific Fixes

### Windows-Specific Issues

#### Issue 1: DataLoader Multiprocessing
**Problem**: `RuntimeError: DataLoader worker exited unexpectedly`

**Solution**: Already fixed! The project automatically uses `num_workers=0` on Windows.

#### Issue 2: Path Separators
**Problem**: Hardcoded `/` paths don't work

**Solution**: Already fixed! Using `pathlib.Path` throughout the project.

#### Issue 3: Spawn Method
**Problem**: Multiprocessing errors during training

**Solution**: Automatically configured! The project sets `spawn` method on Windows.

### Mac-Specific Issues

#### Issue 1: MPS Not Available
**Problem**: Running on older Intel Mac without MPS

**Solution**: Automatically falls back to CPU.

#### Issue 2: OpenMP Conflicts
**Problem**: `OMP: Error #15` or similar

**Solution**: Limit OpenMP threads:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### Linux-Specific Issues

#### Issue 1: CUDA Out of Memory
**Problem**: GPU memory errors

**Solution**: Reduce batch size:
```bash
python scripts/train_densenet_vit_full.py --batch-size 8
```

---

## 🚀 Running Training

### All Platforms (Recommended):

```bash
# Activate virtual environment first!
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Run training
python scripts/train_densenet_vit_full.py
```

### With Custom Settings:

```bash
# Adjust epochs and batch size
python scripts/train_densenet_vit_full.py \
  --epochs-densenet 5 \
  --epochs-vit 5 \
  --batch-size 16
```

### Windows PowerShell (Multi-line):
```powershell
python scripts/train_densenet_vit_full.py `
  --epochs-densenet 5 `
  --epochs-vit 5 `
  --batch-size 16
```

---

## 📊 Running Evaluation

```bash
# Evaluate trained models
python scripts/evaluate_densenet_vit_ensemble.py
```

---

## 🐛 Troubleshooting

### Problem: ImportError or ModuleNotFoundError

**Windows:**
```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Reinstall requirements
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: CUDA/GPU Not Detected (Windows/Linux)

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Problem: MPS Not Working (Mac)

```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# Update to latest PyTorch
pip install --upgrade torch torchvision torchaudio
```

### Problem: Training Very Slow

**Possible causes:**
1. Running on CPU (no GPU acceleration)
2. Too many DataLoader workers
3. Batch size too small

**Solutions:**
```bash
# Check device being used
python test_cross_platform.py

# Increase batch size if you have enough GPU memory
python scripts/train_densenet_vit_full.py --batch-size 32
```

---

## 📁 Data Setup (All Platforms)

1. Download CheXpert dataset from: https://stanfordmlgroup.github.io/competitions/chexpert/

2. Extract to `data/chexpert_small/`

3. Verify structure:
```
data/
└── chexpert_small/
    └── CheXpert-v1.0-small/
        ├── train/
        │   └── train.csv
        └── valid/
            └── valid.csv
```

---

## 🔄 Switching Between Mac and Windows

If you developed on Mac and want to continue on Windows (or vice versa):

### Step 1: Transfer Files
Copy the entire project folder to the new machine.

### Step 2: Recreate Virtual Environment
**DO NOT** copy the `venv` folder! Create a new one:

**On Windows:**
```powershell
# Delete old venv if it exists
Remove-Item -Recurse -Force venv

# Create new venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**On Mac:**
```bash
# Delete old venv
rm -rf venv

# Create new venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Test Compatibility
```bash
python test_cross_platform.py
```

### Step 4: Resume Training
Your trained model checkpoints (`.pth` files) are platform-independent! You can continue training or evaluation on any platform.

---

## ✨ Key Features of Cross-Platform Support

### 1. Automatic Device Detection
```python
# Automatically selects best device:
# - CUDA on Windows/Linux with NVIDIA GPU
# - MPS on Mac with Apple Silicon
# - CPU as fallback
device = get_device()
```

### 2. Platform-Optimized DataLoader
```python
# Windows: num_workers=0 (avoids multiprocessing issues)
# Mac/Linux: num_workers=4 (faster data loading)
from src.utils.platform_utils import create_dataloader
```

### 3. Path Handling
```python
# Works on all platforms using pathlib
from pathlib import Path
data_path = PROJECT_ROOT / "data" / "chexpert_small"
```

---

## 📝 Summary Checklist

Before starting training:
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Compatibility test passed (`python test_cross_platform.py`)
- [ ] Dataset downloaded and extracted
- [ ] GPU/MPS detected (or understand it will use CPU)

---

## 💡 Pro Tips

1. **Use the same Python version** across platforms (3.9+)
2. **Don't commit virtual environment** to git
3. **Model checkpoints are portable** between platforms
4. **Results may vary slightly** due to different hardware
5. **Windows is slower** for data loading (num_workers=0)

---

## 🆘 Getting Help

If you encounter issues:
1. Run `python test_cross_platform.py` to diagnose
2. Check the error message carefully
3. Verify virtual environment is activated
4. Try with smaller batch size (`--batch-size 8`)
5. Ensure you have enough disk space (50GB+)

---

**You're all set! The project now works seamlessly on Windows, Mac, and Linux! 🎉**

