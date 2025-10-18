"""
Comprehensive Ensemble Model Evaluation
Compares DenseNet-121, Inception-ResNet-V2, and Stacking Ensemble performance
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
from src.models.ensemble import create_ensemble
from configs.config import *

# Define conditions
CONDITIONS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def evaluate_model(model, dataloader, device, model_name="Model"):
    """
    Evaluate a single model and return predictions and metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_masks = []
    
    print(f"\n🔍 Evaluating {model_name}...")
    
    with torch.no_grad():
        for batch_idx, (images, labels, masks) in enumerate(dataloader):
            images = images.to(device)
            
            # Handle ensemble vs single model
            if isinstance(model, torch.nn.Module) and hasattr(model, 'meta_learner'):
                outputs, _ = model(images)
            else:
                outputs = model(images)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
            all_masks.append(masks.numpy())
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(dataloader)} batches")
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    return all_preds, all_labels, all_masks


def calculate_metrics(preds, labels, masks, conditions):
    """
    Calculate comprehensive metrics for all conditions
    """
    results = []
    
    for idx, condition in enumerate(conditions):
        # Get valid samples (where mask is 1)
        valid_mask = masks[:, idx] == 1
        
        if valid_mask.sum() == 0:
            print(f"  ⚠️  {condition}: No valid samples")
            continue
        
        y_true = labels[valid_mask, idx]
        y_pred = preds[valid_mask, idx]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            print(f"  ⚠️  {condition}: Only one class present")
            continue
        
        # Calculate metrics
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        
        # Binary predictions (threshold = 0.5)
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Prevalence
        prevalence = y_true.mean()
        n_samples = len(y_true)
        
        results.append({
            'Condition': condition,
            'AUROC': auroc,
            'AUPRC': auprc,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Prevalence': prevalence,
            'N_Samples': n_samples
        })
        
        print(f"  ✓ {condition:25s} AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}  F1: {f1:.4f}")
    
    return pd.DataFrame(results)


def plot_comparison(results_dict, output_dir):
    """
    Create comparison plots for all models
    """
    print("\n📊 Creating comparison plots...")
    
    # Combine all results
    all_results = []
    for model_name, df in results_dict.items():
        df_copy = df.copy()
        df_copy['Model'] = model_name
        all_results.append(df_copy)
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # 1. AUROC Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    conditions = combined['Condition'].unique()
    x = np.arange(len(conditions))
    width = 0.25
    
    colors = {'DenseNet-121': '#1f77b4', 'Inception-ResNet-V2': '#ff7f0e', 'Ensemble': '#2ca02c'}
    
    for i, (model_name, df) in enumerate(results_dict.items()):
        offset = width * (i - 1)
        aurocs = [df[df['Condition'] == c]['AUROC'].values[0] if c in df['Condition'].values else 0 
                  for c in conditions]
        ax.bar(x + offset, aurocs, width, label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('AUROC Comparison: DenseNet-121 vs Inception-ResNet-V2 vs Ensemble', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_auroc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: ensemble_auroc_comparison.png")
    plt.close()
    
    # 2. AUPRC Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (model_name, df) in enumerate(results_dict.items()):
        offset = width * (i - 1)
        auprcs = [df[df['Condition'] == c]['AUPRC'].values[0] if c in df['Condition'].values else 0 
                  for c in conditions]
        ax.bar(x + offset, auprcs, width, label=model_name, color=colors.get(model_name, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
    ax.set_title('AUPRC Comparison: DenseNet-121 vs Inception-ResNet-V2 vs Ensemble', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_auprc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: ensemble_auprc_comparison.png")
    plt.close()
    
    # 3. Performance Improvement Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate improvement over DenseNet-121
    densenet_results = results_dict['DenseNet-121']
    
    improvement_data = []
    for condition in conditions:
        row = {'Condition': condition}
        densenet_auroc = densenet_results[densenet_results['Condition'] == condition]['AUROC'].values[0]
        
        for model_name in ['Inception-ResNet-V2', 'Ensemble']:
            if model_name in results_dict:
                df = results_dict[model_name]
                if condition in df['Condition'].values:
                    model_auroc = df[df['Condition'] == condition]['AUROC'].values[0]
                    improvement = ((model_auroc - densenet_auroc) / densenet_auroc) * 100
                    row[model_name] = improvement
                else:
                    row[model_name] = 0
        
        improvement_data.append(row)
    
    improvement_df = pd.DataFrame(improvement_data)
    improvement_df = improvement_df.set_index('Condition')
    
    sns.heatmap(improvement_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'AUROC Improvement (%)'}, ax=ax)
    ax.set_title('AUROC Improvement over DenseNet-121 (%)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('')
    ax.set_ylabel('Condition', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: ensemble_improvement_heatmap.png")
    plt.close()


def create_summary_report(results_dict, output_dir):
    """
    Create a comprehensive summary report
    """
    print("\n📝 Creating summary report...")
    
    report = []
    report.append("="*80)
    report.append("ENSEMBLE MODEL EVALUATION REPORT")
    report.append("="*80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nModels Evaluated:")
    for model_name in results_dict.keys():
        report.append(f"  • {model_name}")
    
    # Overall performance summary
    report.append("\n" + "="*80)
    report.append("OVERALL PERFORMANCE (Mean AUROC)")
    report.append("="*80)
    
    for model_name, df in results_dict.items():
        mean_auroc = df['AUROC'].mean()
        std_auroc = df['AUROC'].std()
        report.append(f"\n{model_name:25s} {mean_auroc:.4f} ± {std_auroc:.4f}")
    
    # Best improvements
    report.append("\n" + "="*80)
    report.append("TOP 5 CONDITIONS WITH HIGHEST ENSEMBLE IMPROVEMENT")
    report.append("="*80)
    
    if 'Ensemble' in results_dict and 'DenseNet-121' in results_dict:
        ensemble_df = results_dict['Ensemble']
        densenet_df = results_dict['DenseNet-121']
        
        improvements = []
        for _, row in ensemble_df.iterrows():
            condition = row['Condition']
            ensemble_auroc = row['AUROC']
            densenet_row = densenet_df[densenet_df['Condition'] == condition]
            if len(densenet_row) > 0:
                densenet_auroc = densenet_row['AUROC'].values[0]
                improvement = ensemble_auroc - densenet_auroc
                improvements.append({
                    'Condition': condition,
                    'DenseNet AUROC': densenet_auroc,
                    'Ensemble AUROC': ensemble_auroc,
                    'Improvement': improvement
                })
        
        improvements_df = pd.DataFrame(improvements)
        improvements_df = improvements_df.sort_values('Improvement', ascending=False).head(5)
        
        report.append(f"\n{'Condition':<25} {'DenseNet':<12} {'Ensemble':<12} {'Improvement':<12}")
        report.append("-"*80)
        for _, row in improvements_df.iterrows():
            report.append(f"{row['Condition']:<25} {row['DenseNet AUROC']:<12.4f} "
                         f"{row['Ensemble AUROC']:<12.4f} {row['Improvement']:>11.4f}")
    
    # Detailed metrics per condition
    report.append("\n" + "="*80)
    report.append("DETAILED METRICS BY CONDITION")
    report.append("="*80)
    
    for model_name, df in results_dict.items():
        report.append(f"\n{model_name}")
        report.append("-"*80)
        report.append(f"{'Condition':<25} {'AUROC':<10} {'AUPRC':<10} {'F1':<10} {'Samples':<10}")
        report.append("-"*80)
        
        for _, row in df.iterrows():
            report.append(f"{row['Condition']:<25} {row['AUROC']:<10.4f} {row['AUPRC']:<10.4f} "
                         f"{row['F1-Score']:<10.4f} {row['N_Samples']:<10.0f}")
    
    # Save report
    report_text = "\n".join(report)
    report_path = output_dir / 'ensemble_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: ensemble_evaluation_report.txt")
    
    # Also print to console
    print("\n" + report_text)


def main():
    """
    Main evaluation function
    """
    print("\n" + "="*80)
    print("ENSEMBLE MODEL COMPREHENSIVE EVALUATION")
    print("="*80)
    
    device = get_device()
    print(f"\nDevice: {device}")
    
    output_dir = Path('evaluation_results/ensemble_evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup data
    print("\n" + "="*80)
    print("LOADING TEST DATA")
    print("="*80)
    
    val_transform = get_val_transforms()
    val_dataset = CheXpertDataset(
        csv_path=DATA_ROOT / 'CheXpert-v1.0-small/valid/valid.csv',
        data_root=DATA_ROOT,
        transform=val_transform,
        uncertainty_policy='ignore'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Use 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    
    # Load models
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    models_to_evaluate = {}
    
    # 1. DenseNet-121
    print("\n📥 Loading DenseNet-121...")
    densenet_path = 'models/densenet121_inception_stacking/base_models/densenet121_best.pth'
    densenet = create_model("densenet121", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(densenet_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        densenet.load_state_dict(checkpoint['model_state_dict'])
    else:
        densenet.load_state_dict(checkpoint)
    densenet = densenet.to(device)
    densenet.eval()
    models_to_evaluate['DenseNet-121'] = densenet
    print("  ✓ DenseNet-121 loaded")
    
    # 2. Inception-ResNet-V2
    print("\n📥 Loading Inception-ResNet-V2...")
    inception_path = 'models/densenet121_inception_stacking/base_models/inception_resnet_v2_best.pth'
    inception = create_model("inception_resnet_v2", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(inception_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        inception.load_state_dict(checkpoint['model_state_dict'])
    else:
        inception.load_state_dict(checkpoint)
    inception = inception.to(device)
    inception.eval()
    models_to_evaluate['Inception-ResNet-V2'] = inception
    print("  ✓ Inception-ResNet-V2 loaded")
    
    # 3. Ensemble
    print("\n📥 Loading Ensemble...")
    ensemble_path = 'models/densenet121_inception_stacking/ensemble/ensemble_best.pth'
    ensemble = create_ensemble(
        model1=densenet,
        model2=inception,
        num_classes=14,
        hidden_dim=128
    )
    checkpoint = torch.load(ensemble_path, map_location=device)
    if 'ensemble_state_dict' in checkpoint:
        ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
    elif 'model_state_dict' in checkpoint:
        ensemble.load_state_dict(checkpoint['model_state_dict'])
    else:
        ensemble.load_state_dict(checkpoint)
    ensemble = ensemble.to(device)
    ensemble.eval()
    models_to_evaluate['Ensemble'] = ensemble
    print("  ✓ Ensemble loaded")
    
    # Evaluate all models
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)
    
    results_dict = {}
    predictions_dict = {}
    
    for model_name, model in models_to_evaluate.items():
        preds, labels, masks = evaluate_model(model, val_loader, device, model_name)
        metrics_df = calculate_metrics(preds, labels, masks, CONDITIONS)
        
        results_dict[model_name] = metrics_df
        predictions_dict[model_name] = (preds, labels, masks)
        
        # Save individual results
        metrics_df.to_csv(output_dir / f'{model_name.lower().replace(" ", "_").replace("-", "_")}_metrics.csv', 
                         index=False)
        print(f"  ✓ Saved metrics for {model_name}")
    
    # Create comparison plots
    plot_comparison(results_dict, output_dir)
    
    # Create summary report
    create_summary_report(results_dict, output_dir)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  • ensemble_auroc_comparison.png")
    print("  • ensemble_auprc_comparison.png")
    print("  • ensemble_improvement_heatmap.png")
    print("  • ensemble_evaluation_report.txt")
    print("  • Individual model CSV files")
    

if __name__ == '__main__':
    main()
