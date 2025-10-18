"""
Evaluate DenseNet-121 Model Performance
Calculate AUROC, AUPRC, and other metrics for all 14 conditions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, roc_curve
)
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append('src')
from src.data.dataset import CheXpertDataset, get_val_transforms
from src.models.architectures import create_model
from configs.config import *


def evaluate_densenet121(model_path, output_dir='evaluation_results'):
    """
    Comprehensive evaluation of DenseNet-121 model
    """
    
    print("\n" + "="*70)
    print("DenseNet-121 Model Evaluation")
    print("="*70)
    
    device = get_device()
    print(f"Device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"\n📥 Loading model from: {model_path}")
    model = create_model("densenet121", num_classes=14, pretrained=False, dropout_rate=0.2)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("✓ Model loaded and set to evaluation mode")
    
    # Prepare data
    print("\n📊 Preparing test data...")
    train_csv = CHEXPERT_ROOT / "train" / "train.csv"
    val_transform = get_val_transforms()
    
    full_dataset = CheXpertDataset(
        csv_path=train_csv,
        data_root=DATA_ROOT,
        transform=val_transform,
        uncertainty_policy="ignore",
        frontal_only=True
    )
    
    # Use validation split (last 20% of data)
    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    test_indices = list(range(train_size, total_size))
    
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Test samples: {len(test_dataset):,}")
    
    # Collect predictions
    print("\n🔄 Running inference...")
    all_outputs = []
    all_labels = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, masks) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
            
            if batch_idx % 50 == 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches...")
    
    # Concatenate all predictions
    all_outputs = np.concatenate(all_outputs, axis=0)  # (N, 14)
    all_labels = np.concatenate(all_labels, axis=0)    # (N, 14)
    all_masks = np.concatenate(all_masks, axis=0)      # (N, 14)
    
    print(f"✓ Inference complete. Shape: {all_outputs.shape}")
    
    # Convert logits to probabilities
    all_probs = 1 / (1 + np.exp(-all_outputs))  # Sigmoid
    
    # Calculate metrics per condition
    print("\n📈 Calculating metrics...")
    
    condition_names = CHEXPERT_LABELS
    results = []
    
    for idx, condition in enumerate(condition_names):
        # Get valid samples (where mask == 1)
        valid_mask = all_masks[:, idx] == 1
        
        if valid_mask.sum() == 0:
            print(f"  ⚠️  {condition}: No valid samples")
            continue
        
        y_true = all_labels[valid_mask, idx]
        y_prob = all_probs[valid_mask, idx]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Calculate metrics
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except:
            auroc = np.nan
        
        try:
            auprc = average_precision_score(y_true, y_prob)
        except:
            auprc = np.nan
        
        # Other metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Optimal threshold (Youden's J statistic)
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
        except:
            optimal_threshold = 0.5
        
        results.append({
            'Condition': condition,
            'AUROC': auroc,
            'AUPRC': auprc,
            'F1': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Optimal_Threshold': optimal_threshold,
            'Valid_Samples': valid_mask.sum(),
            'Positive_Samples': y_true.sum(),
            'Negative_Samples': (1 - y_true).sum()
        })
        
        print(f"  ✓ {condition:30s} - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate macro averages
    macro_auroc = results_df['AUROC'].mean()
    macro_auprc = results_df['AUPRC'].mean()
    macro_f1 = results_df['F1'].mean()
    
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE")
    print("="*70)
    print(f"Macro-Average AUROC: {macro_auroc:.4f}")
    print(f"Macro-Average AUPRC: {macro_auprc:.4f}")
    print(f"Macro-Average F1:    {macro_f1:.4f}")
    print("="*70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save CSV
    csv_path = output_dir / f"densenet121_metrics_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n💾 Saved metrics to: {csv_path}")
    
    # Save summary JSON
    summary = {
        'model': 'DenseNet-121',
        'model_path': str(model_path),
        'timestamp': timestamp,
        'test_samples': len(test_dataset),
        'macro_metrics': {
            'auroc': float(macro_auroc),
            'auprc': float(macro_auprc),
            'f1': float(macro_f1),
            'accuracy': float(results_df['Accuracy'].mean()),
            'precision': float(results_df['Precision'].mean()),
            'recall': float(results_df['Recall'].mean())
        },
        'per_condition_metrics': results_df.to_dict('records')
    }
    
    json_path = output_dir / f"densenet121_summary_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Saved summary to: {json_path}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # 1. AUROC bar chart
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Sort by AUROC
    results_sorted = results_df.sort_values('AUROC', ascending=True)
    
    colors = ['green' if x >= 0.80 else 'orange' if x >= 0.70 else 'red' 
              for x in results_sorted['AUROC']]
    
    axes[0].barh(results_sorted['Condition'], results_sorted['AUROC'], color=colors, alpha=0.7)
    axes[0].axvline(x=0.80, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.80)')
    axes[0].axvline(x=0.70, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.70)')
    axes[0].axvline(x=macro_auroc, color='blue', linestyle='-', linewidth=2, label=f'Mean ({macro_auroc:.3f})')
    axes[0].set_xlabel('AUROC Score', fontsize=12, fontweight='bold')
    axes[0].set_title('DenseNet-121: AUROC per Condition', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Add values
    for i, (idx, row) in enumerate(results_sorted.iterrows()):
        axes[0].text(row['AUROC'] + 0.01, i, f"{row['AUROC']:.3f}", 
                    va='center', fontsize=9)
    
    # 2. AUPRC bar chart
    results_sorted_auprc = results_df.sort_values('AUPRC', ascending=True)
    
    axes[1].barh(results_sorted_auprc['Condition'], results_sorted_auprc['AUPRC'], 
                color='steelblue', alpha=0.7)
    axes[1].axvline(x=macro_auprc, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean ({macro_auprc:.3f})')
    axes[1].set_xlabel('AUPRC Score', fontsize=12, fontweight='bold')
    axes[1].set_title('DenseNet-121: AUPRC per Condition', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Add values
    for i, (idx, row) in enumerate(results_sorted_auprc.iterrows()):
        axes[1].text(row['AUPRC'] + 0.01, i, f"{row['AUPRC']:.3f}", 
                    va='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = output_dir / f"densenet121_performance_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved plot to: {plot_path}")
    
    plt.close()
    
    # 3. Summary statistics table
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Condition', 'AUROC', 'AUPRC', 'F1', 'Precision', 'Recall', 'Samples'])
    
    for _, row in results_df.iterrows():
        table_data.append([
            row['Condition'],
            f"{row['AUROC']:.4f}",
            f"{row['AUPRC']:.4f}",
            f"{row['F1']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{int(row['Valid_Samples']):,}"
        ])
    
    # Add macro average
    table_data.append(['', '', '', '', '', '', ''])
    table_data.append([
        'MACRO AVERAGE',
        f"{macro_auroc:.4f}",
        f"{macro_auprc:.4f}",
        f"{macro_f1:.4f}",
        f"{results_df['Precision'].mean():.4f}",
        f"{results_df['Recall'].mean():.4f}",
        f"{results_df['Valid_Samples'].sum():,}"
    ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.12, 0.12, 0.12, 0.12, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style macro average row
    for i in range(7):
        table[(len(table_data)-1, i)].set_facecolor('#2196F3')
        table[(len(table_data)-1, i)].set_text_props(weight='bold', color='white')
    
    plt.title('DenseNet-121: Complete Performance Metrics', 
             fontsize=14, fontweight='bold', pad=20)
    
    table_path = output_dir / f"densenet121_table_{timestamp}.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved table to: {table_path}")
    
    plt.close()
    
    print("\n✅ Evaluation complete!")
    print(f"\n📁 All results saved in: {output_dir}")
    
    return results_df, summary


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DenseNet-121 model')
    parser.add_argument('--model-path', type=str,
                       default='models/densenet121_inception_stacking/base_models/densenet121_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("\nAvailable models:")
        for p in Path('models').glob('*.pth'):
            print(f"  - {p}")
        return
    
    print("\n" + "="*70)
    print("DenseNet-121 Model Evaluation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Output: {args.output_dir}")
    
    results_df, summary = evaluate_densenet121(model_path, args.output_dir)
    
    print("\n🎉 Evaluation completed successfully!")
    print("\nKey Results:")
    print(f"  • Macro AUROC: {summary['macro_metrics']['auroc']:.4f}")
    print(f"  • Macro AUPRC: {summary['macro_metrics']['auprc']:.4f}")
    print(f"  • Macro F1:    {summary['macro_metrics']['f1']:.4f}")
    print(f"\n📊 View detailed results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
