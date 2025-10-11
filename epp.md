Extended Project Proposal (EPP)
SRN 23035479
MSc Computer Science – Software Engineering
1. Project Title, Aim & Research Questions
Title DiagXNet-Lite: Training a DenseNet-121 Network to Find Chest X-ray Diseases
Aim Train an ImageNet-pre-trained DenseNet-121 on the public CheXpert-small chest X-ray set so it can recognise 14 common chest findings. Measure how accurate, reliable, and understandable the model is on the official validation set.
Research Questions 1. How well does the fine-tuned model find each of the 14 labels (AUROC, AUPRC, F1)? 2. Are its probability scores well-calibrated, and can temperature scaling cut Expected Calibration Error (ECE) to ≤ 0.10? 3. Do Grad-CAM heat-maps show sensible image areas for true and false predictions?
2. Objectives (SMART)
Core Download CheXpert-small and build a PyTorch DataLoader. Fine-tune DenseNet-121 for 5 epochs; aim for macro AUROC ≥ 0.80. Report AUROC, AUPRC, F1 per label using thresholds from Youden’s J. Plot calibration curve and cut ECE to ≤ 0.10 with temperature scaling. Create 12 Grad-CAM overlays (3 TP, 3 TN, 3 FP, 3 FN). Advance Test focal loss vs. BCE; log any AUROC change. Build a simple demo page.
3. Background
Convolutional neural networks (CNNs) are now the main tool for medical image tasks. Litjens et al. (2017) showed CNNs outperform older methods and popularised AUROC. CheXNet (Rajpurkar et al., 2017) used DenseNet‑121 to match radiologists on pneumonia, introducing transfer learning and Grad‑CAM. CheXpert provides 14-label chest X‑ray data; its small version (~11 GB) fits student hardware. Delbrouck et al. (2025) push toward structured report generation but still rely on strong disease classifiers. This project focuses on that foundation. Benefit: delivers a compact baseline that measures accuracy, calibration, and explainability on modest hardware.
4. Methods / Methodology
Data: CheXpert‑small JPEGs + CSVs, official splits. Pre‑process: Resize 256 → CenterCrop 224, normalise (ImageNet). Batch 16. Model: DenseNet‑121 (ImageNet), Adam lr 1e‑4, BCEWithLogits, 5 epochs. Metrics: AUROC, AUPRC, F1; calibration curve & ECE before/after temp scaling; Grad‑CAM overlays. Tools: PyTorch 2 (MPS), scikit‑learn, pytorch‑grad‑cam, Git.
5. Project Plan
Week 1 – Data & test run Week 2 – DataLoader ready Week 3 – 5‑epoch training done Week 4 – Accuracy metrics Week 5 – Calibration + Grad‑CAM Week 6 – First draft report Week 7 – Extras & polish Week 8 – Final PDF
6. Resources & Skills
Hardware: MacBook Pro M3. Software: Python 3.11, PyTorch 2, torchxrayvision, scikit‑learn. Skills: basic PyTorch; plan - tutorials (4 h total).
7. Relation to Award
Delivers a full software artefact data pipeline, training, evaluation demonstrating software‑engineering practices and ML integration, matching MSc Software Engineering outcomes.
8. Ethics
CheXpert‑small is de‑identified and public; no human subjects.
9. References
Litjens, G. et al. (2017) ‘A survey on deep learning in medical image analysis’, Medical Image Analysis, 42, pp. 60–88. Rajpurkar, P. et al. (2017) ‘CheXNet: Radiologist‑level pneumonia detection on chest X‑rays with deep learning’, arXiv:1711.05225. Delbrouck, J‑B. et al. (2025) ‘Automated structured radiology report generation’, arXiv pre‑print.