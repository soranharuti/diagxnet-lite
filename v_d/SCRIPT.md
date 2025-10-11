# IDV Narration Script - DiagXNet-Lite

**Total Time: ~9 minutes (allowing 1-minute buffer)**

---

## 🎬 **OPENING** (30 seconds)
*"Welcome to the interim demonstration of DiagXNet-Lite, an AI-powered system for automated chest X-ray disease detection. I'll show you the practical work completed, including technical challenges overcome and clinical results achieved."*

**ACTIONS:**
- Show project folder in Finder
- Open VS Code with project

---

## 📁 **SEGMENT 1: System Architecture** (1m 30s)

### Script:
*"DiagXNet-Lite uses DenseNet-121 to analyze chest X-rays and detect 14 pathological conditions simultaneously. The system has modular design with data processing, training, evaluation, and interpretability components."*

**ACTIONS:**
- Navigate folder structure in VS Code
- Open `README.md` briefly
- Show `configs/config.py` 
- Run `python diagnostic.py` in terminal

*"System diagnostic shows all components operational with Apple Silicon MPS acceleration providing 3-4x speed improvement."*

---

## 🔬 **SEGMENT 2: Data Pipeline** (2m 00s)

### Script:
*"Medical imaging presents unique challenges. Our preprocessing handles CheXpert's tri-state labeling where radiologists mark findings as positive, negative, or uncertain."*

**ACTIONS:**
- Open `src/data/dataset.py`
- Highlight uncertainty handling code
- Show data loading output with class distributions

*"The 'ignore' policy masks uncertain labels during training. A critical finding is severe class imbalance - Atelectasis has 96.7% positive samples, significantly impacting learning."*

---

## 📊 **SEGMENT 3: Performance Results** (2m 30s)

### Script:
*"Training results demonstrate clinically relevant performance. Mean AUROC of 0.740 across all conditions meets medical AI standards."*

**ACTIONS:**
- Open `interim_report_evidence/classification_metrics.csv`
- Show `roc_curves.png`
- Navigate to key metrics in Excel/Numbers

*"Pleural Effusion achieves 0.860 AUROC - near radiologist-level. Critical conditions like pneumothorax and pneumonia show good screening capability. Temperature scaling improved calibration from 0.254 to 0.236."*

**ACTIONS:**
- Show `calibration_curves.png`
- Highlight top performing conditions

---

## 🏥 **SEGMENT 4: Clinical Analysis** (2m 00s)

### Script:
*"Enhanced evaluation goes beyond standard metrics. Each condition's error patterns are analyzed individually with clinical urgency scoring that prioritizes life-threatening conditions."*

**ACTIONS:**
- Open `enhanced_analysis/confusion_matrices.png`
- Show `clinical_significance_analysis.png`
- Navigate `comprehensive_analysis.csv`

*"This systematic approach identified specific failure patterns - enlarged cardiomediastinum has 28.7% false positive rate due to class imbalance."*

---

## 🔧 **SEGMENT 5: Technical Challenges** (1m 30s)

### Script:
*"Significant technical challenges were overcome. Apple Silicon compatibility required custom MPS integration. Medical label uncertainty needed specialized masking during training."*

**ACTIONS:**
- Open `DEVELOPMENT_LOG.md`
- Scroll through challenge sections
- Show uncertainty handling implementation

*"Class imbalance quantification revealed the root cause of performance variation. These systematic solutions are documented throughout development."*

---

## 🎯 **CLOSING** (30s)

### Script:
*"DiagXNet-Lite demonstrates substantial progress with 94.5% of objectives completed. The system shows clinically relevant performance with clear improvement pathways through focal loss and ensemble methods. All evidence supports the technical feasibility and clinical potential of AI-assisted chest X-ray diagnosis."*

**ACTIONS:**
- Show summary statistics
- Close with project folder overview

---

## 📋 **Quick Reference During Recording**

### Key Numbers to Mention:
- **191,027** training samples processed
- **14** pathological conditions detected  
- **0.740** mean AUROC achieved
- **11/14** conditions above clinical threshold (0.70)
- **94.5%** project completion rate
- **3-4x** speed improvement with MPS

### Files to Have Ready:
1. VS Code with project open
2. `diagnostic.py` ready to run
3. `classification_metrics.csv` 
4. `roc_curves.png`
5. `confusion_matrices.png`
6. `DEVELOPMENT_LOG.md`

### Mouse Movement Tips:
- Point to specific numbers in CSV files
- Highlight code sections being discussed
- Circle important areas in visualizations
- Use mouse to guide viewer attention

### Backup Plans:
- If file doesn't open quickly, mention what it contains
- If terminal command fails, explain expected output
- Keep speaking while navigating between files
- Have screenshot backup if live demo fails

---

**Remember: Be confident, speak clearly, and let the evidence demonstrate your excellent work!**