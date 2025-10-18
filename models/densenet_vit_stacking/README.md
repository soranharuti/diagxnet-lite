# DenseNet-121 + Vision Transformer Ensemble

This folder will contain the trained models for the DenseNet-121 + Vision Transformer stacking ensemble.

## Architecture

### Base Models
1. **DenseNet-121** (Reused from previous training)
   - Convolutional Neural Network
   - Pre-trained on ImageNet
   - Parameters: ~8M
   - Input: 224x224 grayscale X-rays

2. **Vision Transformer (ViT-B/16)** (Newly trained)
   - Transformer-based architecture
   - Pre-trained on ImageNet
   - Parameters: ~86M
   - Patch size: 16x16
   - Input: 224x224 grayscale X-rays

### Meta-Learner
- Type: Neural Network (2-layer MLP)
- Input: Concatenated predictions from both models (28 features)
- Hidden layers: 64 → 32 neurons
- Output: 14 disease classifications

## Training

To train this ensemble:

```bash
python scripts/train_densenet_vit_ensemble.py
```

### Custom Training Options

```bash
# Train with custom epochs
python scripts/train_densenet_vit_ensemble.py --epochs-vit 10 --epochs-meta 5

# Train with custom learning rates
python scripts/train_densenet_vit_ensemble.py --lr-vit 5e-5 --lr-meta 1e-5

# Train with larger batch size (if you have enough GPU memory)
python scripts/train_densenet_vit_ensemble.py --batch-size 32
```

### Training Time Estimate
- **Vision Transformer**: ~4-5 hours (5 epochs)
- **Meta-Learner**: ~30 minutes (3 epochs)
- **Total**: ~5 hours

### GPU Memory Requirements
- Batch size 16: ~10-12 GB VRAM
- Batch size 32: ~18-20 GB VRAM

## Evaluation

To evaluate the trained ensemble:

```bash
python scripts/evaluate_densenet_vit_ensemble.py
```

This will:
- Evaluate DenseNet-121 individually
- Evaluate Vision Transformer individually  
- Evaluate the stacked ensemble
- Generate comparison plots and metrics
- Save results to `evaluation_results/densenet_vit_evaluation/`

## Why Vision Transformer?

Vision Transformers (ViT) offer several advantages for medical imaging:

1. **Global Context**: Unlike CNNs, ViT captures long-range dependencies from the start
2. **Attention Mechanism**: Can focus on relevant regions across the entire image
3. **Complementary to CNNs**: Combines well with CNNs like DenseNet for ensemble methods
4. **State-of-the-Art**: ViT achieves excellent performance on medical imaging tasks

## Expected Performance

Based on similar studies, we expect:
- **DenseNet-121 alone**: AUROC ~0.82-0.85
- **ViT alone**: AUROC ~0.83-0.86
- **Ensemble**: AUROC ~0.85-0.88 (2-3% improvement)

The ensemble should show improvements especially on:
- Cardiomegaly (benefits from global view)
- Consolidation (benefits from local features)
- Pleural Effusion (benefits from both)

## Model Files

After training, you'll have:

```
models/densenet_vit_stacking/
├── base_models/
│   └── vit_b_16_best.pth         (~340 MB)
└── ensemble/
    └── ensemble_best.pth         (~435 MB - includes both models + meta-learner)
```

Note: DenseNet-121 is loaded from the previous ensemble folder.

## Next Steps

After evaluating this ensemble, you can:

1. **Compare with previous ensemble**: 
   - DenseNet-121 + Inception-ResNet-V2 vs DenseNet-121 + ViT
   
2. **Try different ViT variants**:
   - ViT-B/32 (faster, less memory)
   - ViT-L/16 (larger, potentially better performance)

3. **Create a 3-model ensemble**:
   - DenseNet-121 + Inception-ResNet-V2 + ViT

4. **Experiment with meta-learner architectures**:
   - Deeper networks
   - Attention-based meta-learners
