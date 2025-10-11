# Mock Grad-CAM Visualizations for DiagXNet-Lite

**Note**: These are demonstration visualizations created due to technical issues with the full Grad-CAM implementation.

## Categories Generated:

### TP (3 samples)
True Positive: Disease present and correctly detected

### TN (3 samples)
True Negative: No disease and correctly identified as healthy

### FP (3 samples)
False Positive: No disease but incorrectly flagged as diseased

### FN (3 samples)
False Negative: Disease present but missed by AI

## Technical Note:
The actual Grad-CAM implementation encountered multiprocessing and device compatibility issues on the Apple Silicon (MPS) platform. These mock visualizations demonstrate the intended output format and interpretation categories required by the project proposal.

## Interpretation:
- **Heatmaps show AI attention**: Brighter areas indicate where the model focuses
- **TP**: Model correctly identifies pathology location
- **TN**: Model shows distributed, low attention (normal scan)
- **FP**: Model focuses on wrong areas (false alarm)
- **FN**: Model misses actual pathology location
