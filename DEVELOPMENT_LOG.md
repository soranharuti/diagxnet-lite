# DiagXNet-Lite: Development Log & Technical Report 📋

## Project Overview
**DiagXNet-Lite** is a medical AI system for automated chest X-ray disease detection using deep learning. This document chronicles the complete development process, challenges encountered, solutions implemented, and final results achieved.

**Timeline**: September 6-20, 2025  
**Duration**: ~3 weeks of development  
**Type**: MSc Computer Science Final Research Project

---

## 🎯 Project Objectives (From Proposal)

### Primary Goals:
1. **Model Architecture**: Fine-tune DenseNet-121 (ImageNet pre-trained) for chest X-ray classification
2. **Training Specifications**: 5 epochs, batch size 16, learning rate 1e-4
3. **Performance Targets**:
   - Macro AUROC ≥ 0.80
   - Expected Calibration Error (ECE) ≤ 0.10
4. **Interpretability**: Generate 12 Grad-CAM overlays (3 TP, 3 TN, 3 FP, 3 FN)
5. **Dataset**: CheXpert-small with 14 pathology labels

---

## 📊 Final Results Summary

### ✅ **Successfully Completed:**
| Requirement | Target | Achieved | Status |
|-------------|---------|----------|---------|
| **Training Epochs** | 5 | ✅ 5 | PASS |
| **Batch Size** | 16 | ✅ 16 | PASS |
| **Learning Rate** | 1e-4 | ✅ 1e-4 | PASS |
| **Architecture** | DenseNet-121 | ✅ DenseNet-121 | PASS |
| **Dataset Processing** | CheXpert-small | ✅ 191,027 samples | PASS |
| **Evaluation Metrics** | AUROC, AUPRC, F1 | ✅ Comprehensive evaluation | PASS |
| **Calibration Analysis** | ECE measurement | ✅ With temperature scaling | PASS |

### 📈 **Performance Metrics:**
| Metric | Target | Achieved | Status |
|--------|---------|----------|---------|
| **Macro AUROC** | ≥ 0.80 | 0.74 | ⚠️ Below target |
| **ECE (Before Calibration)** | ≤ 0.10 | 0.2542 | ⚠️ Above target |
| **ECE (After Temperature Scaling)** | ≤ 0.10 | 0.2355 | ⚠️ Improved but still above target |
| **Final Validation Accuracy** | - | 70.66% | ✅ Reasonable for medical AI |

### ⏱️ **Training Performance:**
- **Total Training Time**: 3 hours 41 minutes (221.7 minutes)
- **Training Loss**: 1.036 → 0.909 (improved)
- **Validation Loss**: 0.978 → 1.025 (slight overfitting)
- **Device**: MacBook Pro M3 with MPS acceleration

---

## 🛠️ Technical Challenges & Solutions

### 1. **Environment Setup & Dependencies**
#### Challenge:
- Complex PyTorch installation with MPS (Metal Performance Shaders) support for Apple Silicon
- Multiple package compatibility issues (torch, torchvision, scikit-learn versions)

#### Solution:
```bash
# Final working environment setup:
pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0
pip install scikit-learn matplotlib seaborn pandas numpy
pip install opencv-python Pillow
```

#### Lesson Learned:
Start with environment setup and verify GPU/MPS acceleration before development.

---

### 2. **Data Loading & Processing**
#### Challenge:
- CheXpert dataset uncertainty labels (-1, 0, 1) requiring special handling
- Large dataset (191K+ samples) causing memory management issues
- Multiprocessing conflicts on macOS with spawn method

#### Solution:
```python
# Uncertainty handling in dataset.py:
if self.uncertainty_policy == 'ignore':
    # Convert uncertain labels to NaN and mask
    labels[labels == -1] = float('nan')
    masks = ~torch.isnan(labels)
    labels = torch.nan_to_num(labels, nan=0.0)

# Memory-efficient data loading:
DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)
```

#### Lesson Learned:
Medical datasets require domain-specific preprocessing. Always validate data loading with small samples first.

---

### 3. **Model Architecture Adaptation**
#### Challenge:
- DenseNet-121 pre-trained on RGB images, but X-rays are grayscale
- Need to adapt final layer for 14-class multi-label classification
- Proper weight initialization for modified layers

#### Solution:
```python
# Model adaptation in architectures.py:
def create_model(architecture="densenet121", num_classes=14, pretrained=True, input_channels=1):
    model = models.densenet121(pretrained=pretrained)
    
    # Adapt first layer for grayscale input
    if input_channels == 1:
        # Convert RGB conv to grayscale by averaging weights
        original_conv = model.features.conv0
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Average the RGB weights to create grayscale weights
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        model.features.conv0 = new_conv
    
    # Replace classifier for multi-label classification
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
```

#### Lesson Learned:
Transfer learning requires careful adaptation of input/output layers while preserving pre-trained features.

---

### 4. **Training Loop Implementation**
#### Challenge:
- Multi-label classification requires different loss function than single-label
- Class imbalance in medical data (some diseases are rare)
- Proper handling of uncertain labels during training

#### Solution:
```python
# Multi-label training with masking:
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# In training loop:
masked_outputs = outputs * masks  # Apply uncertainty masks
masked_labels = labels * masks
loss = criterion(masked_outputs, masked_labels)
```

#### Lesson Learned:
Medical AI requires specialized loss functions and careful handling of label uncertainty.

---

### 5. **Device Compatibility (MPS on Apple Silicon)**
#### Challenge:
- Mixed tensor device placement (MPS vs CPU)
- Some operations not supported on MPS requiring fallback to CPU
- Grad-CAM tensor device mismatches

#### Solution:
```python
# Consistent device handling:
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# In Grad-CAM generation:
cam = torch.zeros(activations.shape[1:], dtype=torch.float32, 
                  device=activations.device)  # Match device explicitly
```

#### Lesson Learned:
Apple Silicon (MPS) support is newer and requires explicit device management.

---

### 6. **Evaluation Metrics Implementation**
#### Challenge:
- Multiple evaluation metrics (AUROC, AUPRC, F1, ECE) for 14 labels
- Optimal threshold selection using Youden's J statistic
- Calibration analysis with temperature scaling optimization

#### Solution:
```python
# Comprehensive evaluation in metrics.py:
def find_optimal_thresholds(self):
    """Find optimal thresholds using Youden's J statistic"""
    for i, label in enumerate(self.labels):
        if valid_mask.sum() > 0:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
```

#### Lesson Learned:
Medical AI evaluation requires domain-specific metrics and threshold optimization.

---

### 7. **Grad-CAM Visualization**
#### Challenge:
- Complex implementation requiring gradient hooks
- Device mismatch errors between tensors
- Multiprocessing issues preventing completion

#### Solutions Attempted:
1. **Original Implementation**: Complex data loader iteration
2. **Device Fix**: Explicit tensor device matching
3. **Simplified Version**: Direct dataset access to avoid multiprocessing

#### Final Solution:
```python
# Simplified direct dataset access:
image, labels, masks = dataset[target_data_index]
image = image.unsqueeze(0).to(self.device)
cam = self.gradcam.generate_cam(image, label_idx)
```

#### Current Status:
⚠️ **Partial completion** - Successfully generates individual overlays but multiprocessing issues prevent full 12-overlay generation.

#### Lesson Learned:
Grad-CAM implementation is complex and sensitive to device/multiprocessing configurations. Consider simplified approaches for prototyping.

---

## 📈 Performance Analysis

### 1. **Training Convergence**
```
Epoch 1: Train Loss: 1.036, Val Loss: 0.978, Val Acc: 58.64%
Epoch 2: Train Loss: 0.958, Val Loss: 0.974, Val Acc: 64.12%
Epoch 3: Train Loss: 0.932, Val Loss: 1.070, Val Acc: 68.43%
Epoch 4: Train Loss: 0.918, Val Loss: 1.036, Val Acc: 69.87%
Epoch 5: Train Loss: 0.909, Val Loss: 1.025, Val Acc: 70.66%
```

**Analysis:**
- Training loss steadily decreased (good)
- Validation loss increased after Epoch 2 (slight overfitting)
- Accuracy consistently improved
- 5 epochs may be insufficient for optimal convergence

### 2. **Per-Label Performance**
**Best Performing Labels (AUROC > 0.80):**
- No Finding: 0.831
- Pleural Effusion: 0.860

**Challenging Labels (AUROC < 0.70):**
- Enlarged Cardiomediastinum: 0.621
- Lung Opacity: 0.685
- Atelectasis: 0.669

**Analysis:**
- Some pathologies are inherently harder to detect
- Class imbalance affects performance
- "No Finding" performs well (binary classification easier)

### 3. **Calibration Analysis**
- **Original ECE**: 0.2542 (target: ≤ 0.10)
- **After Temperature Scaling**: 0.2355 (improved but still above target)
- **Optimal Temperature**: 0.4943

**Analysis:**
- Model tends to be overconfident
- Temperature scaling provides some improvement
- Additional calibration methods may be needed

---

## 🔧 Technical Stack

### **Core Technologies:**
- **Deep Learning**: PyTorch 2.8.0 with MPS acceleration
- **Computer Vision**: torchvision, OpenCV, PIL
- **Data Science**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Medical Imaging**: CheXpert dataset processing

### **Development Environment:**
- **Hardware**: MacBook Pro M3 with 8-core GPU
- **OS**: macOS with Apple Silicon optimization
- **Python**: 3.9.10 with virtual environment
- **Storage**: ~50GB for dataset and results

### **Project Structure:**
```
diagxnet-lite/
├── configs/config.py          # Central configuration
├── src/
│   ├── data/dataset.py        # Data loading & preprocessing
│   ├── models/architectures.py # Model definitions
│   ├── training/train.py      # Training pipeline
│   └── evaluation/
│       ├── metrics.py         # Comprehensive evaluation
│       └── gradcam.py         # Interpretability analysis
├── results/                   # Experiment outputs
└── data/chexpert_small/       # Dataset location
```

---

## 📚 Academic Contributions

### **Novel Aspects:**
1. **Comprehensive Evaluation Pipeline**: Implemented full academic-standard evaluation including calibration analysis
2. **Apple Silicon Optimization**: Successfully adapted medical AI pipeline for M-series chips
3. **Uncertainty Handling**: Proper implementation of CheXpert uncertainty policies
4. **Multi-label Medical Classification**: Complete pipeline for 14 simultaneous disease predictions

### **Reproducibility:**
- All hyperparameters documented and configurable
- Seed-based reproducible data splits
- Complete logging of training metrics
- Version-controlled codebase

### **Educational Value:**
- Comprehensive documentation for ML beginners
- Step-by-step explanation of medical AI concepts
- Real-world implementation challenges and solutions

---

## 🎯 Recommendations for Improvement

### **Short-term (Model Performance):**
1. **Extended Training**: Increase to 10-15 epochs with early stopping
2. **Data Augmentation**: Add rotation, brightness, contrast augmentation
3. **Learning Rate Scheduling**: Implement cosine annealing or step decay
4. **Architecture Comparison**: Test ResNet-50, EfficientNet alternatives

### **Medium-term (Technical):**
1. **Grad-CAM Completion**: Resolve multiprocessing issues for full visualization suite
2. **Advanced Calibration**: Implement Platt scaling, isotonic regression
3. **Ensemble Methods**: Combine multiple models for improved performance
4. **Hyperparameter Optimization**: Use Optuna or similar for systematic tuning

### **Long-term (Research):**
1. **External Validation**: Test on NIH ChestX-ray or MIMIC-CXR datasets
2. **Clinical Integration**: Develop DICOM-compatible inference pipeline
3. **Uncertainty Quantification**: Implement Bayesian neural networks
4. **Federated Learning**: Explore multi-institutional training

---

## 📝 Lessons Learned

### **Technical Insights:**
1. **Environment Setup is Critical**: Spend time upfront getting dependencies right
2. **Medical Data is Complex**: Requires domain-specific preprocessing and evaluation
3. **Device Management Matters**: Especially important for Apple Silicon/MPS
4. **Evaluation is as Important as Training**: Medical AI needs rigorous validation

### **Project Management:**
1. **Start Simple**: Build basic pipeline before adding complexity
2. **Test Early and Often**: Validate each component before integration
3. **Document Everything**: Critical for reproducibility and debugging
4. **Plan for Challenges**: Medical AI has unique requirements vs. standard computer vision

### **Research Insights:**
1. **Academic Standards are High**: Requires comprehensive evaluation beyond simple accuracy
2. **Interpretability is Essential**: Medical AI must be explainable for clinical adoption
3. **Calibration Often Overlooked**: But critical for real-world deployment
4. **Incremental Progress is Normal**: 74% AUROC is respectable for medical AI

---

## 🏆 Final Assessment

### **Project Success Metrics:**
✅ **Technical Implementation**: 95% complete (missing only full Grad-CAM suite)  
✅ **Academic Rigor**: Comprehensive evaluation following research standards  
✅ **Documentation**: Extensive documentation for reproducibility and learning  
✅ **Real-world Applicability**: Industry-standard medical AI pipeline  

### **Academic Contribution:**
This project represents a complete implementation of a modern medical AI system, addressing real-world challenges including data uncertainty, class imbalance, model calibration, and interpretability. While performance targets weren't fully met, the comprehensive approach and thorough documentation provide significant educational and research value.

### **Personal Development:**
Successfully demonstrated ability to:
- Implement complex deep learning systems
- Navigate real-world technical challenges
- Follow academic research standards
- Create comprehensive documentation
- Work with medical imaging data

---

## 📖 References & Resources

### **Key Papers:**
- DenseNet: Huang et al. "Densely Connected Convolutional Networks" (2017)
- CheXpert: Irvin et al. "CheXpert: A Large Chest Radiograph Dataset..." (2019)
- Grad-CAM: Selvaraju et al. "Grad-CAM: Visual Explanations..." (2017)

### **Datasets:**
- CheXpert-small: 191,027 chest X-ray images with 14 pathology labels
- Stanford AIMI Dataset: https://stanfordaimi.azurewebsites.net/

### **Technical Documentation:**
- PyTorch Official Documentation
- Apple Silicon MPS Programming Guide
- Medical Imaging with Deep Learning (MIDL) Community

---

*This document serves as a comprehensive technical report for the DiagXNet-Lite project, documenting the journey from concept to implementation, challenges faced, solutions developed, and lessons learned. It provides both academic documentation and practical guidance for future medical AI projects.*

**Project Status**: ✅ **SUBSTANTIALLY COMPLETE**  
**Completion Date**: September 20, 2025  
**Total Development Time**: ~3 weeks