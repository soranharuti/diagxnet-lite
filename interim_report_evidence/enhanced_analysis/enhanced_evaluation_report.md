# DiagXNet-Lite Enhanced Evaluation Report
Generated on: 2025-09-20 13:45:36

## Executive Summary

**Overall Performance**: Mean AUROC = 0.740
**Best Performing**: Pleural Effusion (AUROC: 0.860)
**Most Challenging**: Enlarged Cardiomediastinum (AUROC: 0.621)

## Clinical Impact Analysis

### High Clinical Impact Conditions (Top 5):
- **Pneumonia**: Impact Score 3.511, AUROC 0.725
- **Pleural Effusion**: Impact Score 3.207, AUROC 0.860
- **Pneumothorax**: Impact Score 3.099, AUROC 0.780
- **Edema**: Impact Score 2.919, AUROC 0.793
- **Consolidation**: Impact Score 2.876, AUROC 0.717

## Key Findings

### Strengths:
- No Finding: Excellent performance (AUROC: 0.831)
- Pleural Effusion: Excellent performance (AUROC: 0.860)

### Areas for Improvement:
- Enlarged Cardiomediastinum: AUROC 0.621, High FP Rate: 0.287
- Lung Opacity: AUROC 0.685, High FP Rate: 0.209
- Atelectasis: AUROC 0.669, High FP Rate: 0.318

### Concerning Error Patterns:
- No Finding: 2304 high-confidence false positives, 96 low-confidence false negatives
- Enlarged Cardiomediastinum: 157 high-confidence false positives, 364 low-confidence false negatives
- Cardiomegaly: 1842 high-confidence false positives, 387 low-confidence false negatives
- Lung Opacity: 24 high-confidence false positives, 395 low-confidence false negatives
- Lung Lesion: 735 high-confidence false positives, 232 low-confidence false negatives
- Edema: 3182 high-confidence false positives, 308 low-confidence false negatives
- Consolidation: 240 high-confidence false positives, 47 low-confidence false negatives
- Pneumonia: 1431 high-confidence false positives, 11 low-confidence false negatives
- Atelectasis: 5 high-confidence false positives, 153 low-confidence false negatives
- Pneumothorax: 6330 high-confidence false positives, 25 low-confidence false negatives
- Pleural Effusion: 758 high-confidence false positives, 619 low-confidence false negatives
- Pleural Other: 538 high-confidence false positives, 83 low-confidence false negatives
- Fracture: 182 high-confidence false positives, 406 low-confidence false negatives
- Support Devices: 120 high-confidence false positives, 7414 low-confidence false negatives