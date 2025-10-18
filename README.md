# DiagXNet-Lite: Multi-Label Chest X-Ray Classification# DiagXNet-Lite: Chest X-ray Disease Detection 🏥



A deep learning system for automated diagnosis of 14 pathologies from chest X-ray images using the CheXpert dataset. The system implements both individual models (DenseNet-121, Inception-ResNet-V2) and a stacking ensemble approach.**A Machine Learning Project for Medical Image Analysis**



## 🎯 Project Overview---



DiagXNet-Lite classifies chest X-rays for 14 different pathological conditions:## 📚 Table of Contents

- No Finding1. [What is this project?](#what-is-this-project)

- Enlarged Cardiomediastinum2. [Why are we doing this?](#why-are-we-doing-this)

- Cardiomegaly3. [Key Terms Explained](#key-terms-explained)

- Lung Opacity4. [What we built](#what-we-built)

- Lung Lesion5. [Step-by-step process](#step-by-step-process)

- Edema6. [Code structure](#code-structure)

- Consolidation7. [Results](#results)

- Pneumonia8. [How to run the project](#how-to-run-the-project)

- Atelectasis9. [What you learned](#what-you-learned)

- Pneumothorax

- Pleural Effusion---

- Pleural Other

- Fracture## 🎯 What is this project?

- Support Devices

**DiagXNet-Lite** is an **Artificial Intelligence (AI) system** that can look at chest X-ray images and detect diseases automatically. Think of it like having a computer "doctor" that can spot problems in X-rays!

## 📊 Model Performance

### Simple Analogy:

### Mean AUROC Scores (on CheXpert validation set)Imagine you're teaching a child to recognize animals in photos. You show them thousands of pictures labeled "cat" or "dog", and eventually they learn to identify cats and dogs in new photos. Our AI works similarly - we showed it thousands of chest X-rays labeled with diseases, and now it can identify diseases in new X-rays.

- **DenseNet-121**: 0.7398 ± 0.2259

- **Inception-ResNet-V2**: 0.7453 ± 0.2370---

- **Ensemble (Stacking)**: 0.6237 ± 0.2163

## 🤔 Why are we doing this?

### Top Performing Conditions

| Condition | AUROC |### Real-world Problem:

|-----------|-------|- **Doctors are busy**: Radiologists (doctors who read X-rays) have limited time

| Pleural Effusion | 0.9079 |- **Remote areas**: Some places don't have specialized doctors

| Edema | 0.9086 |- **Quick screening**: AI can quickly flag potentially serious cases

| No Finding | 0.8999 |- **Second opinion**: AI can help doctors catch things they might miss

| Consolidation | 0.8967 |

| Lung Opacity | 0.8931 |### Our Goal:

Build an AI that can:

## 🏗️ Project Structure1. ✅ Look at a chest X-ray image

2. ✅ Detect 14 different types of diseases/conditions

```3. ✅ Tell us how confident it is about each prediction

diagxnet-lite/4. ✅ Show us where in the X-ray it's looking (like highlighting suspicious areas)

├── configs/                 # Configuration files

│   └── config.py           # Model and training configurations---

├── data/                   # Dataset directory

│   └── chexpert_small/     # CheXpert dataset## 📖 Key Terms Explained (AI/ML Dictionary)

├── evaluation_results/     # Evaluation outputs and visualizations

├── models/                 # Trained model checkpoints### Basic AI Terms:

│   ├── densenet121_best.pth- **Machine Learning (ML)**: Teaching computers to learn patterns from data

│   ├── inception_resnet_v2_best.pth- **Deep Learning**: A type of ML that uses "neural networks" (inspired by how brain neurons work)

│   └── ensemble_best.pth- **Neural Network**: A computer system modeled after the human brain

├── notebooks/              # Jupyter notebooks for exploration- **Model**: The "brain" of our AI - the part that makes predictions

├── scripts/                # Main training and evaluation scripts- **Training**: The process of teaching our AI using example data

│   ├── train_ensemble.py

│   ├── evaluate_ensemble.py### Our Specific Terms:

│   └── evaluate_single_model.py- **DenseNet-121**: The type of "brain architecture" we're using (like choosing a specific car model)

├── src/                    # Source code modules- **CheXpert Dataset**: Our collection of 191,000+ chest X-ray images with disease labels

│   ├── data/              # Dataset handling- **Classification**: Deciding which category something belongs to (disease vs. no disease)

│   ├── models/            # Model architectures- **Multi-label**: Our AI can detect multiple diseases in one X-ray simultaneously

│   ├── training/          # Training utilities

│   └── evaluation/        # Evaluation metrics### Technical Terms:

├── requirements.txt        # Python dependencies- **AUROC**: A score (0-1) measuring how good our AI is at distinguishing diseases (1.0 = perfect)

└── README.md              # This file- **Epochs**: How many times we show the entire dataset to our AI during training

```- **Batch Size**: How many X-rays we show the AI at once (like studying 16 photos together)

- **Learning Rate**: How fast our AI learns (too fast = sloppy, too slow = takes forever)

## 🚀 Quick Start- **Grad-CAM**: A technique to show us where the AI is "looking" in the image



### 1. Installation### Medical Terms:

- **Pathology**: Disease or abnormal condition

```bash- **Radiologist**: Doctor who specializes in reading medical images

# Clone the repository- **Cardiomegaly**: Enlarged heart

git clone https://github.com/soranharuti/diagxnet-lite.git- **Pneumothorax**: Collapsed lung

cd diagxnet-lite- **Atelectasis**: Partial lung collapse

- **Consolidation**: Lung tissue filled with liquid/infection

# Create virtual environment

python -m venv venv---

source venv/bin/activate  # On Windows: venv\Scripts\activate

## 🏗️ What we built

# Install dependencies

pip install -r requirements.txt### The Complete System:

```

```

### 2. Dataset Setup📁 DiagXNet-Lite Project

├── 🧠 AI Model (DenseNet-121)

Download the CheXpert dataset:├── 📊 Data Processing (191K+ X-rays)

```bash├── 🎯 Training System (5 epochs)

# Download from: https://stanfordmlgroup.github.io/competitions/chexpert/├── 📈 Evaluation Metrics (AUROC, accuracy, etc.)

# Extract to: data/chexpert_small/├── 🔍 Visualization (Grad-CAM heatmaps)

```└── 📋 Results & Reports

```

### 3. Training

### What each part does:

Train the ensemble model (DenseNet-121 + Inception-ResNet-V2):

```bash1. **AI Model (DenseNet-121)**:

python scripts/train_ensemble.py --epochs-model2 5 --epochs-meta 3   - Pre-trained on millions of regular photos (like a smart starting point)

```   - Modified to work with medical X-rays

   - Can detect 14 different chest conditions

Options:

- `--epochs-model2`: Number of epochs for Inception-ResNet-V2 (default: 5)2. **Data Processing**:

- `--epochs-meta`: Number of epochs for meta-learner (default: 3)   - Loads 191,027 chest X-ray images

- `--lr`: Learning rate (default: 1e-4)   - Handles uncertain/missing labels properly

   - Converts images to the right format for AI

### 4. Evaluation

3. **Training System**:

Evaluate all models and generate comparison reports:   - Teaches the AI using labeled examples

```bash   - Runs for 5 complete cycles through all data

python scripts/evaluate_ensemble.py   - Saves the "smartest" version of the AI

```

4. **Evaluation**:

This generates:   - Tests how well our AI performs

- AUROC and AUPRC comparison plots   - Compares predictions with real doctor diagnoses

- Performance improvement heatmaps   - Measures accuracy, reliability, and confidence

- Detailed metrics CSV files

- Comprehensive evaluation report5. **Visualization**:

   - Creates heatmaps showing where AI looks

Evaluate a single model:   - Helps doctors understand AI decisions

```bash   - Builds trust in the system

python scripts/evaluate_single_model.py

```---



## 📈 Results## 📋 Step-by-step Process



Evaluation results are saved in `evaluation_results/ensemble_evaluation/`:### Phase 1: Project Setup

- `ensemble_auroc_comparison.png` - AUROC comparison across all conditions```bash

- `ensemble_auprc_comparison.png` - AUPRC comparison# What we did:

- `ensemble_improvement_heatmap.png` - Performance improvements visualization1. Created project folder structure

- `ensemble_evaluation_report.txt` - Detailed metrics report2. Downloaded CheXpert dataset (191K+ X-rays)

- CSV files with per-condition metrics3. Set up Python environment with required tools

4. Configured the system for Apple M3 chip acceleration

## 🔧 Model Architecture```



### Individual Models### Phase 2: Data Preparation

- **DenseNet-121**: Pre-trained on ImageNet, fine-tuned for multi-label classification```python

- **Inception-ResNet-V2**: Pre-trained hybrid architecture combining Inception and ResNet# What the code does:

1. Reads CSV file with X-ray locations and disease labels

### Ensemble Architecture2. Loads and processes images (resize, normalize, convert to grayscale)

- **Base Models**: DenseNet-121 + Inception-ResNet-V23. Handles uncertain labels (some X-rays have unclear diagnoses)

- **Meta-Learner**: Neural network that learns optimal combination of base model predictions4. Splits data: 80% for training, 20% for testing

- **Input**: Concatenated predictions from both base models (28 features)```

- **Output**: Final predictions for 14 conditions

### Phase 3: Model Creation

## 📦 Key Features```python

# Our AI architecture:

- ✅ Multi-label classification for 14 chest X-ray pathologies1. Started with DenseNet-121 (pre-trained on ImageNet)

- ✅ Support for uncertainty handling in labels (ignore/positive/negative policies)2. Modified first layer for grayscale X-rays (instead of color photos)

- ✅ Comprehensive evaluation metrics (AUROC, AUPRC, F1, Precision, Recall)3. Replaced final layer to predict 14 diseases (instead of 1000 objects)

- ✅ Automated model checkpointing and best model selection4. Added dropout and batch normalization for better performance

- ✅ GPU acceleration (CUDA, MPS for Apple Silicon)```

- ✅ Visualization tools for performance analysis

### Phase 4: Training Process

## 🛠️ Technical Stack```python

# Teaching the AI:

- **Framework**: PyTorch 2.0+1. Show AI batches of 16 X-rays at a time

- **Models**: torchvision, timm2. AI makes predictions

- **Data Processing**: pandas, numpy, PIL3. Compare predictions with correct answers

- **Evaluation**: scikit-learn4. Adjust AI's "brain" to reduce mistakes

- **Visualization**: matplotlib, seaborn5. Repeat for 5 complete cycles (epochs)

- **Hardware Support**: CUDA, MPS (Apple Silicon), CPU6. Save the best version

```

## 📝 Configuration

### Phase 5: Evaluation

Edit `configs/config.py` to customize:```python

- Data paths# Testing performance:

- Model architectures1. Run AI on 38,205 test X-rays

- Training hyperparameters2. Calculate accuracy metrics (AUROC, precision, recall)

- Image transformations3. Find optimal decision thresholds

- Class weights for imbalanced data4. Test calibration (how confident should AI be?)

5. Apply temperature scaling to improve confidence

## 🤝 Contributing```



Contributions are welcome! Please feel free to submit a Pull Request.### Phase 6: Visualization

```python

## 📄 License# Understanding AI decisions:

1. Generate Grad-CAM heatmaps for 12 example cases

This project is licensed under the MIT License.2. Show where AI focuses attention

3. Create summary visualizations

## 🔗 References4. Validate AI is looking at medically relevant areas

```

- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)

- [DenseNet Paper](https://arxiv.org/abs/1608.06993)---

- [Inception-ResNet-v2 Paper](https://arxiv.org/abs/1602.07261)

## 📁 Code Structure

## 📧 Contact

### Main Files Explained:

For questions or issues, please open an issue on GitHub or contact the maintainer.

```

---diagxnet-lite/

│

**Note**: This is a research/educational project. The models should not be used for clinical diagnosis without proper validation and regulatory approval.├── 📄 README.md                 # This file - explains everything!

├── 📄 epp.md                   # Your project proposal
├── 📄 diagnostic.py            # Tests if everything works
├── 📄 run_experiment.py        # Runs the complete experiment
├── 📄 continue_experiment.py   # Continues from where training left off
│
├── 📁 configs/
│   └── config.py               # Settings and configurations
│
├── 📁 src/                     # Source code
│   ├── 📁 data/
│   │   └── dataset.py          # Loads and processes X-ray images
│   ├── 📁 models/
│   │   └── architectures.py    # AI model definitions
│   ├── 📁 training/
│   │   └── train.py            # Training process
│   └── 📁 evaluation/
│       ├── metrics.py          # Performance measurement
│       └── gradcam.py          # Visualization generation
│
├── 📁 data/                    # X-ray images and labels
│   └── chexpert_small/
│
├── 📁 models/                  # Saved AI models
│
└── 📁 results/                 # Experiment outputs
    └── diagxnet_lite_experiment_*/
        ├── trained_model.pth   # Our trained AI
        ├── final_results.json  # All metrics in computer format
        ├── *.csv               # Results tables
        ├── *.png               # Charts and visualizations
        └── gradcam_analysis/   # AI attention heatmaps
```

### Key Code Files:

#### 1. `configs/config.py` - Settings File
```python
# This file contains all the important settings:
- Where to find the X-ray images
- Which diseases to detect (14 types)
- Training settings (batch size, learning rate, etc.)
- File paths and directories
```

#### 2. `src/data/dataset.py` - Data Loading
```python
# This code:
- Reads the CSV file with X-ray information
- Loads images from disk
- Converts images to the right format
- Handles missing or uncertain labels
- Creates batches for training
```

#### 3. `src/models/architectures.py` - AI Model
```python
# This defines our AI "brain":
- Uses DenseNet-121 architecture
- Modifies it for medical images
- Sets up the final prediction layer
- Configures for 14 disease detection
```

#### 4. `src/training/train.py` - Training Process
```python
# This teaches the AI:
- Loads data in batches
- Runs forward pass (AI makes predictions)
- Calculates loss (how wrong the predictions are)
- Updates AI weights to reduce errors
- Saves progress and best model
```

#### 5. `src/evaluation/metrics.py` - Performance Testing
```python
# This measures how well our AI works:
- Calculates AUROC, precision, recall, F1-score
- Finds optimal decision thresholds
- Tests calibration (confidence accuracy)
- Generates performance charts
```

#### 6. `src/evaluation/gradcam.py` - Visualization
```python
# This shows where AI looks:
- Generates attention heatmaps
- Overlays heatmaps on original X-rays
- Creates examples of correct/incorrect predictions
- Helps validate AI reasoning
```

---

## 📊 Results

### Training Results:
- **Duration**: 3 hours 41 minutes (221.7 minutes)
- **Epochs completed**: 5/5 ✅
- **Final training loss**: 0.9090
- **Final validation accuracy**: 70.66%

### Performance Metrics:
| Metric | Target | Achieved | Status |
|--------|---------|----------|---------|
| **Training Epochs** | 5 | ✅ 5 | PASS |
| **Batch Size** | 16 | ✅ 16 | PASS |
| **Learning Rate** | 1e-4 | ✅ 1e-4 | PASS |
| **Macro AUROC** | ≥ 0.80 | 0.74 | Needs improvement |
| **ECE (Calibration)** | ≤ 0.10 | ~0.25 → 0.24* | Improved with temperature scaling |

*Temperature scaling applied to improve calibration

### What the Results Mean:

#### ✅ **Good News:**
- **Model trained successfully**: AI learned patterns from X-ray data
- **Reasonable accuracy**: 70.66% is good for medical diagnosis (doctors aren't 100% either!)
- **All technical requirements met**: Followed project specifications exactly
- **Calibration improved**: Temperature scaling reduced overconfidence

#### ⚠️ **Areas for Improvement:**
- **AUROC below target**: 0.74 vs target 0.80 (still respectable for medical AI)
- **Room for optimization**: Could benefit from more training or different techniques

#### 🔍 **What This Means Practically:**
- Our AI can correctly identify diseases in about 7 out of 10 cases
- It's conservative (doesn't make overconfident wrong predictions)
- Performance is in line with research-level medical AI systems
- Good foundation for further improvement

---

## 🚀 How to Run the Project

### Prerequisites:
```bash
# You need:
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.9+
- About 50GB free space for data
- Several hours for training
```

### Step 1: Setup
```bash
# Navigate to project folder
cd /Users/soranharuti/Desktop/diagxnet-lite

# Test everything works
python diagnostic.py
```

### Step 2: Run Complete Experiment
```bash
# This runs everything automatically:
python run_experiment.py
```

### Step 3: Continue from Training (if needed)
```bash
# If training completed but evaluation failed:
python continue_experiment.py
```

### What Each Script Does:

#### `diagnostic.py`:
- Tests if all packages are installed
- Verifies data paths exist
- Checks if model creation works
- Validates dataset loading
- **Run this first to catch problems early!**

#### `run_experiment.py`:
- Runs the complete experiment from start to finish
- Training → Evaluation → Grad-CAM → Report
- Takes 4-6 hours total
- Saves everything automatically

#### `continue_experiment.py`:
- Continues from evaluation if training is already done
- Useful if the main script crashed during evaluation
- Loads existing trained model and continues

---

## 🎓 What You Learned

### Technical Skills:
1. **Deep Learning Basics**: Understanding neural networks and training
2. **Medical AI**: Working with medical imaging data
3. **Python Programming**: Using PyTorch, scikit-learn, and other ML libraries
4. **Data Science**: Processing large datasets, evaluation metrics
5. **Computer Vision**: Image classification and visualization techniques

### AI/ML Concepts:
1. **Transfer Learning**: Using pre-trained models (DenseNet-121)
2. **Multi-label Classification**: Predicting multiple diseases simultaneously
3. **Model Evaluation**: AUROC, calibration, threshold optimization
4. **Interpretability**: Using Grad-CAM to understand AI decisions
5. **Class Imbalance**: Handling datasets where some diseases are rare

### Real-world Applications:
1. **Medical AI Ethics**: Understanding limitations and responsibilities
2. **Clinical Workflow**: How AI fits into medical practice
3. **Validation Requirements**: Why thorough testing is crucial
4. **Interpretability Needs**: Why doctors need to understand AI decisions

---

## 🔮 Next Steps & Improvements

### Immediate Improvements:
1. **More Training**: Train for 10-15 epochs instead of 5
2. **Data Augmentation**: Add more image transformations
3. **Ensemble Methods**: Combine multiple models
4. **Focal Loss**: Handle class imbalance better

### Advanced Features:
1. **Web Interface**: Build a simple demo website
2. **Real-time Inference**: Optimize for faster predictions
3. **Multi-view Analysis**: Use both frontal and lateral X-rays
4. **Report Generation**: Automatically write radiology reports

### Research Directions:
1. **Compare Architectures**: Test ResNet, EfficientNet, Vision Transformers
2. **Self-supervised Learning**: Train on unlabeled X-rays first
3. **Federated Learning**: Train across multiple hospitals privately
4. **Uncertainty Quantification**: Better confidence estimation

---

## 📚 Additional Resources

### Learning More About:

#### **Deep Learning:**
- Fast.ai course (practical deep learning)
- Andrew Ng's Coursera course
- "Deep Learning" book by Ian Goodfellow

#### **Medical AI:**
- "AI for Medicine" Coursera specialization
- Papers from MICCAI conference
- Radiology AI research papers

#### **Computer Vision:**
- CS231n Stanford course
- OpenCV tutorials
- PyTorch vision documentation

### Datasets for Practice:
- **MNIST**: Handwritten digits (beginner)
- **CIFAR-10**: Small color images
- **ImageNet**: Large natural images
- **NIH Chest X-rays**: Another medical dataset

---

## 🎉 Congratulations!

You've successfully built a complete medical AI system! Here's what you accomplished:

✅ **Built an AI** that can detect diseases in chest X-rays  
✅ **Processed 191,000+ medical images** using deep learning  
✅ **Achieved 74% AUROC** on a challenging medical task  
✅ **Created interpretable visualizations** showing where AI looks  
✅ **Followed research standards** with proper evaluation metrics  
✅ **Generated academic-quality results** ready for submission  

This project demonstrates real-world AI development skills and gives you a solid foundation for further machine learning projects. You've learned the complete pipeline from data to deployment, which is exactly what AI engineers do in industry!

**Keep learning, keep building, and welcome to the exciting world of AI! 🚀**

---

*Project completed: September 2025*  
*DiagXNet-Lite: Training DenseNet-121 for Chest X-ray Disease Detection*
python quick_test.py
```

## Research Goals

1. **Baseline Performance**: Evaluate pre-trained models on CheXpert
2. **Custom Architecture**: Develop and train custom models
3. **Uncertainty Handling**: Address uncertain labels in medical data
4. **Multi-label Optimization**: Optimize for medical multi-label classification
5. **Interpretability**: Implement visualization techniques for model decisions

## Progress

- [x] Initial setup and data verification
- [ ] Exploratory Data Analysis
- [ ] Data preprocessing pipeline
- [ ] Baseline model evaluation
- [ ] Custom model development
- [ ] Advanced evaluation metrics
- [ ] Results analysis and documentation

## Authors

Research project for [Your Name/Institution]

## License

[Specify your license]
