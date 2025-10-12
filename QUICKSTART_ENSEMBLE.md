# 🚀 Quick Start: Pair-wise Stacking Ensemble

## 📋 Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) For Inception-ResNet-V2
pip install timm
```

## ✅ Test Setup

```bash
# Verify everything is working
python test_ensemble_setup.py
```

## 🎯 Choose Your Ensemble

### Option 1: DenseNet-121 + EfficientNet-B3 (Recommended)
**Fast, efficient, good performance**

```bash
python train_stacking_ensemble.py \
    --model1 densenet121 \
    --model2 efficientnet_b3 \
    --meta-learner neural_network \
    --epochs-base 5 \
    --epochs-meta 3
```

### Option 2: DenseNet-121 + Inception-ResNet-V2
**Maximum performance, requires more resources**

```bash
python train_stacking_ensemble.py \
    --model1 densenet121 \
    --model2 inception_resnet_v2 \
    --meta-learner neural_network \
    --epochs-base 5 \
    --epochs-meta 3
```

## 📊 Compare Options

```bash
python compare_ensemble_options.py
```

## 🎓 Training Process

```
Stage 1: Train DenseNet-121      (~2 hours, 5 epochs)
Stage 2: Train EfficientNet-B3   (~2 hours, 5 epochs)
Stage 3: Train Meta-Learner      (~30 mins, 3 epochs)
────────────────────────────────────────────────────
Total Time: ~5 hours
```

## 📁 Output Files

After training, you'll find:
- `models/stacking_densenet121_efficientnet_b3_*_model1_best.pth`
- `models/stacking_densenet121_efficientnet_b3_*_model2_best.pth`
- `models/stacking_densenet121_efficientnet_b3_*_ensemble_best.pth`
- `results/stacking_densenet121_efficientnet_b3_*_summary.json`

## 🔧 Advanced Options

```bash
# Use logistic meta-learner (faster)
--meta-learner logistic

# More epochs for base models
--epochs-base 10

# More epochs for meta-learner
--epochs-meta 5

# Smaller batch size (for limited GPU)
--batch-size 8

# Custom learning rate
--lr 5e-5
```

## 📚 Documentation

- **Full Guide**: `ENSEMBLE_GUIDE.md`
- **Implementation Details**: `ENSEMBLE_IMPLEMENTATION.md`
- **Compare Options**: `python compare_ensemble_options.py`

## 🆘 Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
--batch-size 8
```

### Missing timm library (for Inception-ResNet-V2)
```bash
pip install timm
```

### Slow training
```bash
# Use EfficientNet-B3 instead of Inception-ResNet-V2
--model2 efficientnet_b3
```

## 📈 Expected Results

| Model | AUROC | Improvement |
|-------|-------|-------------|
| DenseNet-121 | 0.80-0.82 | Baseline |
| + EfficientNet-B3 | 0.82-0.85 | +2-3% |
| + Inception-ResNet-V2 | 0.83-0.86 | +3-4% |

## ✨ Next Steps After Training

1. Evaluate ensemble performance
2. Compare with individual models
3. Generate Grad-CAM visualizations
4. Analyze per-condition improvements
5. Test on hold-out test set

---

**Ready to start?** Run `python test_ensemble_setup.py` first!
