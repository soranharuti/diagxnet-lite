# IDV Recording Day Checklist - DiagXNet-Lite

## 🕐 **1 Hour Before Recording**

### **Environment Setup**
- [ ] **Clean physical space** - organized desk, good lighting
- [ ] **Charge laptop** or connect power cable  
- [ ] **Close all unnecessary applications** 
- [ ] **Turn off notifications** (Do Not Disturb mode)
- [ ] **Clear desktop** of personal/irrelevant files
- [ ] **Test internet connection** (if uploading afterwards)

### **Technical Preparation** 
- [ ] **Launch OBS Studio** and verify settings
- [ ] **Test webcam** - positioning, lighting, focus
- [ ] **Test microphone** - clear audio, appropriate levels
- [ ] **Test screen capture** - full screen visible, sharp quality
- [ ] **Record 30-second test** - verify audio/video sync

### **Content Preparation**
- [ ] **Open VS Code** with DiagXNet-Lite project
- [ ] **Navigate to evidence folder** - have it ready in Finder
- [ ] **Pre-open key files** in appropriate apps:
  - `classification_metrics.csv` (Numbers/Excel)
  - `roc_curves.png` (Preview)
  - `confusion_matrices.png` (Preview) 
  - `DEVELOPMENT_LOG.md` (VS Code or TextEdit)
- [ ] **Test terminal** - ensure `python diagnostic.py` runs smoothly
- [ ] **Review script** - final read-through of narration

---

## 🎬 **During Recording (10 minutes)**

### **Opening Setup (First 30 seconds)**
- [ ] **Start recording** in OBS
- [ ] **Take a breath** - speak clearly and confidently
- [ ] **Begin with introduction** - look at camera, then screen
- [ ] **Show project folder** - establish context immediately

### **Segment-by-Segment Execution**

#### **Segment 1: Architecture (1m 30s)**
- [ ] Navigate folder structure smoothly
- [ ] Point with mouse to key directories
- [ ] Open `README.md` briefly to show overview
- [ ] Run `python diagnostic.py` - narrate while it executes
- [ ] Highlight successful system check results

#### **Segment 2: Data Pipeline (2m)**  
- [ ] Open `src/data/dataset.py` - scroll to key functions
- [ ] Explain uncertainty handling while showing code
- [ ] Display data loading output with class distributions
- [ ] Use mouse to point to specific imbalance statistics

#### **Segment 3: Performance (2m 30s)**
- [ ] Open `classification_metrics.csv` cleanly
- [ ] Point to key numbers (0.740 AUROC, top performers)
- [ ] Switch to `roc_curves.png` - explain curves briefly
- [ ] Show `calibration_curves.png` - mention calibration improvement

#### **Segment 4: Clinical Analysis (2m)**
- [ ] Navigate to enhanced analysis folder
- [ ] Open `confusion_matrices.png` - explain medical relevance
- [ ] Show `clinical_significance_analysis.png`
- [ ] Briefly review comprehensive CSV data

#### **Segment 5: Challenges (1m 30s)**
- [ ] Open `DEVELOPMENT_LOG.md` - scroll through systematically  
- [ ] Highlight specific challenge sections
- [ ] Explain solutions while showing relevant code
- [ ] Connect challenges to learning and improvement

### **Closing (30s)**
- [ ] **Summarize achievements** with confidence
- [ ] **State completion percentage** (94.5%)
- [ ] **End positively** - technical feasibility and clinical potential
- [ ] **Stop recording** smoothly

---

## ✅ **Post-Recording Tasks**

### **Immediate Review**
- [ ] **Watch entire video** - check for technical issues
- [ ] **Verify audio quality** - clear speech throughout
- [ ] **Check webcam visibility** - face visible, not blocking content
- [ ] **Confirm timing** - under 10 minutes
- [ ] **Review content coverage** - all key points addressed

### **Quality Assessment**
- [ ] **Technical demonstration clear** - all files opened successfully
- [ ] **Performance evidence visible** - numbers readable, charts clear
- [ ] **Problem-solving shown** - challenges and solutions explained
- [ ] **Logical flow maintained** - smooth transitions between segments
- [ ] **Professional presentation** - confident delivery, good pacing

### **File Management**
- [ ] **Locate recorded file** - note file size and location
- [ ] **Test file playback** - ensure video plays correctly
- [ ] **Backup video file** - save to cloud storage if large
- [ ] **Prepare for upload** - check submission requirements
- [ ] **Document video details** - length, file size, format for submission

---

## 🎯 **Success Indicators**

### **Technical Quality**
✅ **Video:** Sharp, readable text and clear visualizations  
✅ **Audio:** Clear speech with consistent volume  
✅ **Webcam:** Professional appearance, good lighting  
✅ **Timing:** 8-10 minutes (allowing buffer under limit)

### **Content Quality**  
✅ **Practical Work:** Comprehensive system demonstration  
✅ **Evidence:** Clear visibility of performance metrics and results  
✅ **Problem-Solving:** Technical challenges and solutions explained  
✅ **Organization:** Logical flow from overview to technical details  

### **Assessment Alignment**
✅ **Amount of Work (10%):** Full pipeline demonstrated  
✅ **Quality of Work (10%):** Clinical-grade results shown  
✅ **Problem Resolution (10%):** Systematic solutions explained  
✅ **Logical Presentation (10%):** Clear structure maintained  
✅ **Clear Evidence (10%):** All metrics and visualizations visible  

---

## 🚨 **Emergency Backup Plans**

### **If Technology Fails**
- **OBS crashes:** Have QuickTime ready as backup screen recorder
- **File won't open:** Explain what it contains while trying to fix
- **Python script fails:** Describe expected output and continue
- **Webcam stops:** Continue recording, address in post-production

### **If You Make Mistakes**
- **Wrong information:** Correct immediately and continue
- **Stumble over words:** Take brief pause and continue naturally
- **Miss timing:** Prioritize most important content, skip less critical details
- **Lose place:** Refer to quick reference notes briefly

### **Technical Issues During Recording**
- **Keep talking** while resolving issues
- **Stay calm** - technical problems show real-world demonstration
- **Explain briefly** what you're doing to fix issues
- **Move forward** - don't spend too long on any one problem

---

## 📋 **Final Confidence Boosters**

### **Remember Your Achievements**
- ✅ **94.5% project completion** - substantial progress demonstrated
- ✅ **Clinical-grade performance** - 0.740 AUROC exceeds standards  
- ✅ **Technical innovation** - Apple Silicon optimization, uncertainty handling
- ✅ **Systematic methodology** - professional medical AI development

### **You're Well Prepared**
- ✅ **Complete evidence folder** with 26+ supporting files
- ✅ **Comprehensive analysis** beyond standard computer vision metrics
- ✅ **Clear problem-solving** documented throughout development
- ✅ **Professional documentation** supporting every claim

**🎬 You're ready to create an excellent IDV that showcases your outstanding work on DiagXNet-Lite!**