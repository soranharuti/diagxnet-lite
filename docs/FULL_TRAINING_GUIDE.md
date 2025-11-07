# Full Training Guide: DenseNet-121 + Vision Transformer Ensemble

## Quick Start

Train both models from scratch with all checkpoints saved:

```bash
cd /Users/soranharuti/Desktop/diagxnet-lite
python scripts/train_densenet_vit_full.py
```

## What This Does

✅ **Trains DenseNet-121 from scratch**
- 10 epochs (default)
- Saves checkpoint for every epoch
- Saves best model based on validation loss

✅ **Trains Vision Transformer from scratch**
- 10 epochs (default)
- Saves checkpoint for every epoch
- Saves best model based on validation loss

✅ **Trains Meta-Learner**
- 5 epochs (default)
- Saves checkpoint for every epoch
- Saves best ensemble

## Saved Checkpoints

After training completes, you'll have:

```
models/densenet_vit_stacking/
├── base_models/
│   ├── densenet121_best.pth              # Best DenseNet checkpoint
│   ├── densenet121_checkpoints/
│   │   ├── densenet121_epoch_01.pth
│   │   ├── densenet121_epoch_02.pth
│   │   ├── ...
│   │   ├── densenet121_epoch_10.pth
│   │   └── training_history.json        # Training metrics
│   │
│   ├── vit_b_16_best.pth                # Best ViT checkpoint
│   └── vit_checkpoints/
│       ├── vit_b_16_epoch_01.pth
│       ├── vit_b_16_epoch_02.pth
│       ├── ...
│       ├── vit_b_16_epoch_10.pth
│       └── training_history.json        # Training metrics
│
└── ensemble/
    ├── ensemble_best.pth                 # Best ensemble
    └── checkpoints/
        ├── ensemble_epoch_01.pth
        ├── ensemble_epoch_02.pth
        ├── ...
        ├── ensemble_epoch_05.pth
        └── training_history.json         # Training metrics
```

## Custom Training Options

### Train with more epochs
```bash
python scripts/train_densenet_vit_full.py \
  --epochs-densenet 15 \
  --epochs-vit 15 \
  --epochs-meta 10
```

### Custom learning rates
```bash
python scripts/train_densenet_vit_full.py \
  --lr-densenet 5e-5 \
  --lr-vit 5e-5 \
  --lr-meta 1e-6
```

### Larger batch size (needs more GPU memory)
```bash
python scripts/train_densenet_vit_full.py --batch-size 32
```

### Full custom configuration
```bash
python scripts/train_densenet_vit_full.py \
  --epochs-densenet 12 \
  --epochs-vit 12 \
  --epochs-meta 8 \
  --batch-size 24 \
  --lr-densenet 1e-4 \
  --lr-vit 8e-5 \
  --lr-meta 5e-6
```

## Monitoring Training

### Check training progress
The script prints detailed progress including:
- Current batch and loss
- Epoch summary (train/val loss)
- Learning rate
- Time per epoch
- Best model saves

### Review training history
After training, check the JSON files:

```bash
# DenseNet-121 history
cat models/densenet_vit_stacking/base_models/densenet121_checkpoints/training_history.json

# ViT history
cat models/densenet_vit_stacking/base_models/vit_checkpoints/training_history.json

# Ensemble history
cat models/densenet_vit_stacking/ensemble/checkpoints/training_history.json
```

## Choosing the Best Checkpoint

### Option 1: Use the automatically selected best model (Recommended)
The script automatically saves the model with the lowest validation loss:
- `densenet121_best.pth`
- `vit_b_16_best.pth`
- `ensemble_best.pth`

### Option 2: Manually select from checkpoints

1. **Review training history**:
```python
import json
with open('models/densenet_vit_stacking/base_models/densenet121_checkpoints/training_history.json') as f:
    history = json.load(f)
    for entry in history:
        print(f"Epoch {entry['epoch']}: Val Loss = {entry['val_loss']:.4f}")
```

2. **Copy desired checkpoint**:
```bash
# Example: Use epoch 8 instead of automatically selected best
cp models/densenet_vit_stacking/base_models/densenet121_checkpoints/densenet121_epoch_08.pth \
   models/densenet_vit_stacking/base_models/densenet121_best.pth
```

## Time Estimates

### With default settings (10+10+5 epochs, batch_size=16):
- **DenseNet-121**: ~2-3 hours (10 epochs)
- **Vision Transformer**: ~4-5 hours (10 epochs)
- **Meta-Learner**: ~30-40 minutes (5 epochs)
- **Total**: ~7-9 hours

### GPU Memory Requirements:
- Batch size 16: ~10-12 GB VRAM
- Batch size 24: ~14-16 GB VRAM
- Batch size 32: ~18-20 GB VRAM

## After Training

### Evaluate the ensemble:
```bash
python scripts/evaluate_densenet_vit_ensemble.py
```

### Compare different checkpoints:
You can modify the evaluation script to load different epoch checkpoints and compare their performance.

## Training Features

### DenseNet-121
- **Optimizer**: Adam with weight decay
- **Scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Pre-training**: ImageNet weights
- **Dropout**: 0.2

### Vision Transformer
- **Optimizer**: AdamW (better for transformers)
- **Scheduler**: OneCycleLR with 10% warmup
- **Gradient Clipping**: Max norm 1.0 (prevents exploding gradients)
- **Pre-training**: ImageNet weights
- **Dropout**: 0.2

### Meta-Learner
- **Optimizer**: Adam with weight decay
- **Architecture**: 2-layer MLP (64→32 neurons)
- **Base models**: Frozen during meta-learner training

## Troubleshooting

### Out of memory error
```bash
# Reduce batch size
python scripts/train_densenet_vit_full.py --batch-size 8
```

### Training too slow
```bash
# Reduce epochs
python scripts/train_densenet_vit_full.py --epochs-densenet 5 --epochs-vit 5 --epochs-meta 3
```

### Want to resume from checkpoint
Currently not supported in this script. To resume, you would need to:
1. Load the checkpoint
2. Extract optimizer and scheduler states
3. Continue training from that epoch

---

**Ready to train!** 🚀

The script will handle everything automatically and save all checkpoints so you can choose the best performing models later.
