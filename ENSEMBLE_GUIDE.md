# Pair-wise Stacking Ensemble for DiagXNet-Lite

This module implements **pair-wise stacking ensemble learning** by combining predictions from two base models using a meta-learner.

## 🎯 Overview

**Stacking** is an ensemble learning technique that:
1. Trains multiple base models independently
2. Uses base model predictions as features for a meta-learner
3. Meta-learner learns the optimal combination of predictions

## 🏗️ Architecture

### Base Models (Choose One Pair)
1. **DenseNet-121** + **EfficientNet-B3**
2. **DenseNet-121** + **Inception-ResNet-V2**

### Meta-Learner Options
- **Neural Network**: 2-layer feedforward network (recommended)
- **Logistic Regression**: Simple linear combination

## 📊 Training Pipeline

```
┌─────────────────────────────────────────┐
│  Stage 1: Train Base Models            │
│  ────────────────────────                │
│  Model 1 (DenseNet-121)                 │
│  Model 2 (EfficientNet-B3 / Inception)  │
│  • Trained independently on 60% data    │
│  • Validated on 20% data                │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Stage 2: Extract Predictions           │
│  ─────────────────────────               │
│  • Get predictions from both models     │
│  • On separate 20% meta-training set    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Stage 3: Train Meta-Learner            │
│  ────────────────────────                │
│  • Input: Concatenated predictions      │
│  • Learn optimal combination             │
│  • Base models frozen (default)         │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Additional Dependencies

```bash
pip install timm  # For Inception-ResNet-V2
```

### 2. Train DenseNet-121 + EfficientNet-B3

```bash
python train_stacking_ensemble.py \
    --model1 densenet121 \
    --model2 efficientnet_b3 \
    --meta-learner neural_network \
    --epochs-base 5 \
    --epochs-meta 3 \
    --batch-size 16 \
    --lr 1e-4
```

### 3. Train DenseNet-121 + Inception-ResNet-V2

```bash
python train_stacking_ensemble.py \
    --model1 densenet121 \
    --model2 inception_resnet_v2 \
    --meta-learner neural_network \
    --epochs-base 5 \
    --epochs-meta 3
```

## 📁 File Structure

```
diagxnet-lite/
├── src/models/
│   ├── architectures.py      # Base model architectures
│   │   ├── DenseNet-121
│   │   ├── EfficientNet-B3
│   │   └── Inception-ResNet-V2
│   └── ensemble.py            # Ensemble models
│       ├── StackingEnsemble   # Stacking with meta-learner
│       ├── MetaLearner        # Neural network / logistic
│       └── WeightedAverage    # Simple weighted average (alternative)
└── train_stacking_ensemble.py # Training script
```

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--model1` | `densenet121` | Any architecture | First base model |
| `--model2` | `efficientnet_b3` | `efficientnet_b3`, `inception_resnet_v2` | Second base model |
| `--meta-learner` | `neural_network` | `neural_network`, `logistic` | Meta-learner type |
| `--epochs-base` | `5` | Any int | Epochs for base models |
| `--epochs-meta` | `3` | Any int | Epochs for meta-learner |
| `--batch-size` | `16` | Any int | Batch size |
| `--lr` | `1e-4` | Any float | Learning rate |

## 🎓 How It Works

### 1. Base Model Training

Each base model is trained independently:

```python
# Train Model 1
model1 = create_model("densenet121", num_classes=14)
train(model1, train_data)  # 60% of data

# Train Model 2  
model2 = create_model("efficientnet_b3", num_classes=14)
train(model2, train_data)  # 60% of data
```

### 2. Meta-Feature Generation

Extract predictions as features:

```python
# On meta-training set (20% of data)
pred1 = model1(X_meta)  # Shape: (N, 14)
pred2 = model2(X_meta)  # Shape: (N, 14)

# Concatenate
meta_features = torch.cat([pred1, pred2], dim=1)  # Shape: (N, 28)
```

### 3. Meta-Learner Training

Train meta-learner on base predictions:

```python
meta_learner = MetaLearner(
    num_base_models=2,
    num_classes=14,
    meta_learner_type="neural_network"
)

# Learn optimal combination
output = meta_learner([pred1, pred2])  # Shape: (N, 14)
```

## 📈 Expected Improvements

Ensemble learning typically provides:
- **+2-5% AUROC** improvement over single models
- **Better calibration** (more reliable confidence scores)
- **Reduced variance** (more stable predictions)
- **Complementary errors** (models make different mistakes)

## 🔍 Evaluation

After training, evaluate the ensemble:

```python
from src.models.ensemble import create_ensemble

# Load trained models
ensemble = create_ensemble(model1, model2, ensemble_type="stacking")
ensemble.load_state_dict(torch.load("ensemble_best.pth"))

# Evaluate
ensemble.eval()
with torch.no_grad():
    output, base_outputs = ensemble(X_test)
    
# Compare with individual models
print(f"Model 1 AUROC: {auroc(y_test, base_outputs['model1'])}")
print(f"Model 2 AUROC: {auroc(y_test, base_outputs['model2'])}")
print(f"Ensemble AUROC: {auroc(y_test, output)}")
```

## 💡 Tips

1. **Choose Diverse Models**: EfficientNet and DenseNet have different architectures
2. **Meta-Learner Type**: Neural network usually works better but takes longer
3. **Freeze Base Models**: Keep base models frozen during meta-training (default)
4. **Data Split**: 60/20/20 split ensures meta-learner sees unseen predictions

## 📚 References

- **Stacking**: Wolpert, D. H. (1992). "Stacked generalization"
- **DenseNet**: Huang et al. (2017). "Densely Connected Convolutional Networks"
- **EfficientNet**: Tan & Le (2019). "EfficientNet: Rethinking Model Scaling"
- **Inception-ResNet**: Szegedy et al. (2017). "Inception-v4, Inception-ResNet"

## 🎯 Next Steps

After training:
1. ✅ Evaluate ensemble vs individual models
2. ✅ Calculate AUROC, AUPRC, F1 for all 14 conditions
3. ✅ Analyze which conditions benefit most from ensemble
4. ✅ Generate Grad-CAM for ensemble predictions
5. ✅ Compare calibration curves

---

**Questions?** Check the main README or open an issue!
