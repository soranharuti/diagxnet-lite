# Training Vision Transformer + DenseNet-121 Ensemble

## Overview

I've set up everything you need to train a **Vision Transformer (ViT-B/16) + DenseNet-121 stacking ensemble**. This is your second ensemble architecture that will use a different approach than the DenseNet-121 + Inception-ResNet-V2 ensemble.

## What's Been Created

### 1. Vision Transformer Architecture Added
- **File**: `src/models/architectures.py`
- **Added**: `VisionTransformerModel` class
- **Supports**: ViT-B/16, ViT-B/32, ViT-L/16
- **Features**:
  - Pre-trained on ImageNet
  - Modified for grayscale (1-channel) input
  - Dropout for regularization
  - Transformer-based architecture with self-attention

### 2. Training Script
- **File**: `scripts/train_densenet_vit_ensemble.py`
- **What it does**:
  1. Loads your existing DenseNet-121 model (no retraining needed!)
  2. Trains a new Vision Transformer from scratch
  3. Creates a stacking ensemble with neural network meta-learner
  4. Saves models to `models/densenet_vit_stacking/`

### 3. Evaluation Script
- **File**: `scripts/evaluate_densenet_vit_ensemble.py`
- **What it does**:
  - Evaluates DenseNet-121, ViT, and Ensemble separately
  - Compares all three models
  - Generates comparison plots
  - Saves results to `evaluation_results/densenet_vit_evaluation/`

### 4. Documentation
- **File**: `models/densenet_vit_stacking/README.md`
- Complete guide for this ensemble approach

## Organized Model Structure

Your models are now organized like this:

```
models/
├── densenet121_inception_stacking/      # First ensemble (already trained)
│   ├── base_models/
│   │   ├── densenet121_best.pth
│   │   └── inception_resnet_v2_best.pth
│   └── ensemble/
│       └── ensemble_best.pth
│
└── densenet_vit_stacking/               # Second ensemble (ready to train)
    ├── base_models/
    │   └── vit_b_16_best.pth           (will be created during training)
    └── ensemble/
        └── ensemble_best.pth           (will be created during training)
```

## How to Train

### Basic Training (Recommended)
```bash
cd /Users/soranharuti/Desktop/diagxnet-lite
python scripts/train_densenet_vit_ensemble.py
```

This will train with default settings:
- Vision Transformer: 5 epochs (~4-5 hours)
- Meta-learner: 3 epochs (~30 minutes)
- Batch size: 16
- Learning rate (ViT): 1e-4
- Learning rate (Meta): 1e-5

### Custom Training Options

```bash
# More epochs for better performance
python scripts/train_densenet_vit_ensemble.py --epochs-vit 10 --epochs-meta 5

# Larger batch size (needs more GPU memory)
python scripts/train_densenet_vit_ensemble.py --batch-size 32

# Custom learning rates
python scripts/train_densenet_vit_ensemble.py --lr-vit 5e-5 --lr-meta 1e-5

# Use a different DenseNet checkpoint
python scripts/train_densenet_vit_ensemble.py \
  --densenet-path models/densenet121_inception_stacking/base_models/densenet121_best.pth
```

## What Makes This Different?

### Vision Transformer vs Inception-ResNet-V2

| Feature | Inception-ResNet-V2 | Vision Transformer |
|---------|---------------------|-------------------|
| Architecture | CNN | Transformer |
| Local features | ✓ Strong | ○ Moderate |
| Global context | ○ Limited | ✓ Strong |
| Attention | ✗ No | ✓ Yes |
| Parameters | ~55M | ~86M |
| Training time | 3-4 hours | 4-5 hours |
| Memory usage | Moderate | Higher |

### Why Vision Transformer?

1. **Global Understanding**: ViT processes the entire image from the start, unlike CNNs that build up from local features
2. **Attention Mechanism**: Can learn which parts of X-rays are important for each disease
3. **Complementary to DenseNet**: DenseNet excels at local features, ViT excels at global context
4. **State-of-the-Art**: ViT has shown excellent results on medical imaging tasks

## Training Features

### For Vision Transformer
- **Optimizer**: AdamW (better for transformers)
- **Learning Rate Schedule**: OneCycleLR with warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Data Augmentation**: Same as previous training

### For Meta-Learner
- **Architecture**: 2-layer MLP (64 → 32 neurons)
- **Input**: Concatenated predictions from DenseNet + ViT
- **Optimizer**: Adam with weight decay

## After Training

### Evaluate Your Models
```bash
python scripts/evaluate_densenet_vit_ensemble.py
```

This generates:
- Individual model metrics (AUROC, AUPRC per disease)
- Comparison plots
- Improvement heatmap showing where ensemble helps most
- CSV files with detailed metrics

### Compare Both Ensembles

You can then compare:
1. **Ensemble 1**: DenseNet-121 + Inception-ResNet-V2
2. **Ensemble 2**: DenseNet-121 + Vision Transformer

To see which combination works better!

## Expected Results

Based on similar studies:
- **DenseNet-121 alone**: AUROC ~0.82-0.85
- **Vision Transformer alone**: AUROC ~0.83-0.86  
- **Ensemble**: AUROC ~0.85-0.88

The ensemble should show the most improvement on:
- **Cardiomegaly**: Benefits from ViT's global view
- **Consolidation**: Benefits from DenseNet's local features
- **Pleural Effusion**: Benefits from both approaches

## GPU Requirements

- **Minimum**: 10 GB VRAM (batch size 16)
- **Recommended**: 16 GB VRAM (batch size 24-32)
- **Estimated Training Time**: ~5 hours total

## Next Steps After Training

1. **Evaluate both ensembles** and compare performance
2. **Analyze where each ensemble excels**
3. **Consider a 3-model ensemble** (DenseNet + Inception + ViT)
4. **Try different ViT variants** (ViT-B/32 for speed, ViT-L/16 for performance)
5. **Experiment with meta-learner architectures**

## Quick Start Commands

```bash
# 1. Train the Vision Transformer ensemble
python scripts/train_densenet_vit_ensemble.py

# 2. Evaluate it
python scripts/evaluate_densenet_vit_ensemble.py

# 3. Compare with previous ensemble
ls -lh evaluation_results/
```

## Files Updated

✅ `src/models/architectures.py` - Added ViT support  
✅ `scripts/train_densenet_vit_ensemble.py` - Training script created  
✅ `scripts/evaluate_densenet_vit_ensemble.py` - Evaluation script created  
✅ `models/densenet_vit_stacking/README.md` - Documentation created  

---

**Ready to train!** Just run the training script and it will handle everything automatically. 🚀
