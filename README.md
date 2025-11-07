# DiagXNet-Lite

**Deep Learning Ensemble for Multi-Label Chest X-Ray Classification**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

DiagXNet-Lite is an advanced AI system that automatically detects 14 thoracic pathologies from chest X-ray images using the CheXpert dataset. The system implements a **two-stage stacking ensemble** combining DenseNet-121 and Vision Transformer (ViT-B/16) with a meta-learner for optimal predictions.

**Key Features:**
- 🎯 Multi-label classification for 14 chest X-ray pathologies
- 🚀 State-of-the-art ensemble learning (AUROC: 0.8465)
- ⚡ GPU-accelerated training with mixed precision
- 🔧 Focal Loss + Balanced Sampling for class imbalance
- 📊 Comprehensive evaluation and visualization tools
- 🌐 Cross-platform support (Windows, Linux, macOS)

---

## 🏆 Performance Highlights

| Model | Mean AUROC | Mean AUPRC | Training Time |
|-------|------------|------------|---------------|
| **DenseNet-121** | 0.8409 | 0.5121 | 163 min |
| **Vision Transformer** | 0.7443 | 0.4100 | 314 min |
| **Ensemble (V2)** | **0.8465** | **0.5149** | 184 min |

**Top Pathology Performance:**
- Pleural Effusion: **0.9203 AUROC** ✅
- Consolidation: **0.9160 AUROC** ✅
- Enlarged Cardiomediastinum: **0.8997 AUROC** ✅
- No Finding: **0.8984 AUROC** ✅ (127% improvement over V1)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diagxnet-lite.git
cd diagxnet-lite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the CheXpert dataset:
```bash
# Visit: https://stanfordmlgroup.github.io/competitions/chexpert/
# Download CheXpert-v1.0-small
# Extract to: data/chexpert_small/
```

### 3. Training

**Train V2 Model (Recommended):**
```bash
python scripts/train_densenet_vit_v2_improved.py
```

**Train V1 Model:**
```bash
python scripts/train_densenet_vit_full_optimized.py
```

### 4. Evaluation

```bash
# Evaluate V2 ensemble
python scripts/evaluate_densenet_vit_v2_ensemble.py

# Evaluate V1 ensemble
python scripts/evaluate_densenet_vit_ensemble.py
```

---

## 📁 Project Structure

```
diagxnet-lite/
├── configs/                      # Configuration files
│   ├── config.py                 # Main configuration
│   └── platform_config.py        # Platform-specific settings
├── data/                         # Dataset directory
│   └── chexpert_small/           # CheXpert dataset
├── docs/                         # Documentation
│   ├── FULL_TRAINING_GUIDE.md
│   ├── OPTIMIZATION_GUIDE.md
│   └── CROSS_PLATFORM_SETUP.md
├── evaluation_results/           # Evaluation outputs
│   ├── densenet_vit_evaluation/  # V1 results
│   └── densenet_vit_v2_evaluation/ # V2 results
├── models/                       # Trained model checkpoints
│   ├── densenet_vit_stacking/    # V1 models
│   └── densenet_vit_stacking_v2/ # V2 models
├── notebooks/                    # Jupyter notebooks
│   └── 01_exploratory_data_analysis.ipynb
├── project_comparison/           # V1 vs V2 comparison
│   ├── V1_VS_V2_COMPARISON_REPORT.md
│   └── figures/
├── project_report/               # Comprehensive documentation
│   ├── COMPREHENSIVE_PROJECT_REPORT.md
│   ├── EXECUTIVE_SUMMARY.pdf.md
│   ├── evaluation_tables_and_figures/
│   └── figures/
├── scripts/                      # Training and evaluation scripts
│   ├── train_densenet_vit_v2_improved.py  # V2 training
│   ├── evaluate_densenet_vit_v2_ensemble.py # V2 evaluation
│   └── ...
├── src/                          # Source code
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training utilities
│   ├── evaluation/               # Evaluation metrics
│   └── utils/                    # Utility functions
├── .gitignore
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🔬 Methodology

### Two-Stage Stacking Ensemble

**Stage 1: Base Model Training**
- DenseNet-121 (CNN): Captures local patterns and features
- Vision Transformer (ViT-B/16): Captures global context with self-attention
- Both trained independently for 10 epochs

**Stage 2: Meta-Learner Training**
- Freeze base models (preserve learned representations)
- Extract 28 predictions (14 from each model)
- Train 3-layer MLP meta-learner (12 epochs)
- Meta-learner learns optimal combination strategy

### V2 Improvements (Focal Loss + Balanced Sampling)

**Problem:** V1 had severe class imbalance issues
- Most common: Lung Opacity (47.3%)
- Least common: Pleural Other (1.6%)
- **Imbalance ratio:** 32:1

**Solution:**
1. **Focal Loss** (α=0.25, γ=2.0)
   - Down-weights easy examples
   - Focuses on hard-to-classify cases
   - Addresses false positive bias

2. **Balanced Batch Sampler**
   - 2x oversampling for rare classes (<10% prevalence)
   - Ensures exposure to underrepresented pathologies

**Results:**
- ✅ Mean AUROC: +6.0% improvement
- ✅ "No Finding" detection: +127.9% improvement (0.39 → 0.90)
- ✅ Atelectasis: +16.8% improvement (0.74 → 0.86)

---

## 📊 Dataset

**CheXpert Dataset**
- Training: 223,414 frontal chest X-rays
- Validation: 234 images
- Patients: ~65,000
- Labels: 14 thoracic observations

**Pathologies:**
1. No Finding
2. Enlarged Cardiomediastinum
3. Cardiomegaly
4. Lung Opacity
5. Lung Lesion
6. Edema
7. Consolidation
8. Pneumonia
9. Atelectasis
10. Pneumothorax
11. Pleural Effusion
12. Pleural Other
13. Fracture (excluded - insufficient data)
14. Support Devices (excluded in V2 - not a disease)

---

## 🛠️ Technical Details

### Model Specifications

| Component | Details |
|-----------|---------|
| **DenseNet-121** | 7.98M parameters, ImageNet pretrained |
| **Vision Transformer** | 86.57M parameters, ImageNet-21k pretrained |
| **Meta-Learner** | 41K trainable parameters |
| **Total Parameters** | 94.59M (only 41K trained in Stage 2) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-4 |
| **Batch Size** | 32 |
| **Loss Function** | Focal Loss (V2) / BCE (V1) |
| **Epochs** | 10 (base), 12 (ensemble) |
| **GPU Memory** | ~10 GB |
| **Mixed Precision** | Enabled (AMP) |

### Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (RTX 3060+)
- RAM: 16GB
- Storage: 50GB

**Recommended:**
- GPU: 16GB VRAM (RTX 4060 Ti+)
- RAM: 32GB
- Storage: 100GB

---

## 📈 Results

### V1 vs V2 Comparison

| Pathology | V1 AUROC | V2 AUROC | Δ | Status |
|-----------|----------|----------|---|--------|
| **No Finding** | 0.3941 | **0.8984** | **+0.5042** | 🚀 Critical Fix |
| **Atelectasis** | 0.7362 | **0.8597** | **+0.1235** | 🔥 Major Improvement |
| **Enlarged Cardiomediastinum** | 0.8495 | 0.8997 | +0.0502 | ✅ Improvement |
| **Lung Opacity** | 0.8688 | 0.8918 | +0.0229 | ✅ Improvement |
| **Pleural Effusion** | 0.9037 | 0.9203 | +0.0166 | ✅ Improvement |
| **Consolidation** | 0.9180 | 0.9160 | -0.0020 | ✅ Maintained |
| **Pneumonia** | 0.9143 | 0.8191 | -0.0951 | ⚠️ Decrease |

**Overall:** 67% of pathologies improved, 25% maintained, 8% decreased.

---

## 📚 Documentation

### Comprehensive Reports
- [📄 Comprehensive Project Report](project_report/COMPREHENSIVE_PROJECT_REPORT.md) - Full technical documentation
- [📊 Executive Summary](project_report/EXECUTIVE_SUMMARY.pdf.md) - High-level overview
- [📈 V1 vs V2 Comparison](project_comparison/V1_VS_V2_COMPARISON_REPORT.md) - Detailed analysis
- [📑 Evaluation Tables & Figures](project_report/evaluation_tables_and_figures/) - Publication-ready materials

### Guides
- [🚀 Full Training Guide](docs/FULL_TRAINING_GUIDE.md)
- [⚡ Optimization Guide](docs/OPTIMIZATION_GUIDE.md)
- [🌐 Cross-Platform Setup](docs/CROSS_PLATFORM_SETUP.md)

---

## 🔧 Advanced Usage

### Custom Training

```python
from src.models.architectures import create_model
from src.training.focal_loss import FocalLoss

# Create model
model = create_model('densenet121', num_classes=13, pretrained=True)

# Use Focal Loss
criterion = FocalLoss(alpha=0.25, gamma=2.0)

# Train with balanced sampling
from src.data.balanced_sampler import BalancedBatchSampler
sampler = BalancedBatchSampler(dataset, oversample_factor=2.0)
```

### Custom Evaluation

```python
from src.evaluation.metrics import calculate_metrics

# Evaluate model
metrics = calculate_metrics(predictions, labels, masks, pathology_names)
print(f"Mean AUROC: {metrics['mean_auroc']:.4f}")
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{diagxnet_lite_2025,
  title={DiagXNet-Lite: DenseNet-ViT Stacking Ensemble for CheXpert Classification},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/diagxnet-lite}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Stanford ML Group** for the CheXpert dataset
- **PyTorch Team** for the deep learning framework
- **Ross Wightman** for the timm library
- **Open Source Community** for various contributions

---

## ⚠️ Disclaimer

This is a research project. The models should **not be used for clinical diagnosis** without proper validation and regulatory approval. Always consult qualified healthcare professionals for medical decisions.

---

## 📞 Contact

For questions or collaborations:
- Email: your.email@example.com
- GitHub Issues: [Report a bug](https://github.com/yourusername/diagxnet-lite/issues)

---

<p align="center">
  <b>Built with ❤️ for advancing medical AI research</b>
</p>
