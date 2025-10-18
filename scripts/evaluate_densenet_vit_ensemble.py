"""
Evaluate DenseNet-121 + Vision Transformer Ensemble
Compares individual model performance against ensemble
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append('src')
from src.data.dataset import CheXpertDataset, get_val_transforms
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble
from configs.config import *


def evaluate_model(model, dataloader, device, model_name="Model"):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    all_labels = []
    all_predictions = []
    all_masks = []
    
    with torch.no_grad():
        for batch_idx, (images, labels, masks) in enumerate(dataloader):
            images = images.to(device)
            
            if "ensemble" in model_name.lower():
                outputs, _ = model(images)
            else:
                outputs = model(images)
            
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            all_labels.append(labels.numpy())
            all_predictions.append(predictions)
            all_masks.append(masks.numpy())
            
            if batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Calculate metrics per class
    auroc_scores = []
    auprc_scores = []
    
    for i, label in enumerate(CHEXPERT_LABELS):
        # Get valid samples for this class
        valid_mask = all_masks[:, i] == 1
        
        if valid_mask.sum() == 0:
            print(f"  ⚠️  {label}: No valid samples")
            auroc_scores.append(np.nan)
            auprc_scores.append(np.nan)
            continue
        
        y_true = all_labels[valid_mask, i]
        y_pred = all_predictions[valid_mask, i]
        
        # Check if we have both classes
        if len(np.unique(y_true)) < 2:
            print(f"  ⚠️  {label}: Only one class present")
            auroc_scores.append(np.nan)
            auprc_scores.append(np.nan)
            continue
        
        try:
            auroc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            auroc_scores.append(auroc)
            auprc_scores.append(auprc)
            print(f"  ✓ {label:20s} AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}")
        except Exception as e:
            print(f"  ✗ {label}: Error - {e}")
            auroc_scores.append(np.nan)
            auprc_scores.append(np.nan)
    
    # Calculate mean scores (ignoring NaN)
    mean_auroc = np.nanmean(auroc_scores)
    mean_auprc = np.nanmean(auprc_scores)
    
    print(f"\n{'─'*60}")
    print(f"Mean AUROC: {mean_auroc:.4f}")
    print(f"Mean AUPRC: {mean_auprc:.4f}")
    print(f"{'─'*60}")
    
    return {
        'model_name': model_name,
        'auroc_scores': auroc_scores,
        'auprc_scores': auprc_scores,
        'mean_auroc': mean_auroc,
        'mean_auprc': mean_auprc,
        'labels': CHEXPERT_LABELS
    }


def plot_comparison(results_list, output_dir):
    """Plot comparison of models"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    df_data = []
    for results in results_list:
        for label, auroc, auprc in zip(results['labels'], 
                                       results['auroc_scores'], 
                                       results['auprc_scores']):
            if not np.isnan(auroc) and not np.isnan(auprc):
                df_data.append({
                    'Model': results['model_name'],
                    'Pathology': label,
                    'AUROC': auroc,
                    'AUPRC': auprc
                })
    
    df = pd.DataFrame(df_data)
    
    # Plot AUROC comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    df_auroc = df.pivot(index='Pathology', columns='Model', values='AUROC')
    df_auroc.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_title('AUROC Comparison: DenseNet-121 vs ViT vs Ensemble', fontsize=14, fontweight='bold')
    ax.legend(title='Model', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'auroc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot AUPRC comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    df_auprc = df.pivot(index='Pathology', columns='Model', values='AUPRC')
    df_auprc.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('AUPRC', fontsize=12)
    ax.set_xlabel('Pathology', fontsize=12)
    ax.set_title('AUPRC Comparison: DenseNet-121 vs ViT vs Ensemble', fontsize=14, fontweight='bold')
    ax.legend(title='Model', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0.0, 1.0])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'auprc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot improvement heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate improvement over individual models
    ensemble_results = [r for r in results_list if 'Ensemble' in r['model_name']][0]
    densenet_results = [r for r in results_list if 'DenseNet' in r['model_name']][0]
    vit_results = [r for r in results_list if ('ViT' in r['model_name'] or 'Vision Transformer' in r['model_name']) and 'Ensemble' not in r['model_name']][0]
    
    improvements = []
    for label, ens_auroc, dn_auroc, vit_auroc in zip(
        ensemble_results['labels'],
        ensemble_results['auroc_scores'],
        densenet_results['auroc_scores'],
        vit_results['auroc_scores']
    ):
        if not (np.isnan(ens_auroc) or np.isnan(dn_auroc) or np.isnan(vit_auroc)):
            improvements.append({
                'Pathology': label,
                'vs DenseNet': ens_auroc - dn_auroc,
                'vs ViT': ens_auroc - vit_auroc
            })
    
    df_improvement = pd.DataFrame(improvements)
    df_improvement = df_improvement.set_index('Pathology')
    
    sns.heatmap(df_improvement.T, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'AUROC Improvement'}, ax=ax)
    ax.set_title('Ensemble Improvement Over Individual Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Comparison', fontsize=12)
    ax.set_xlabel('Pathology', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to {output_dir}")


def save_results(results_list, output_dir):
    """Save results to CSV and text file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual model results
    for results in results_list:
        model_name = results['model_name'].lower().replace(' ', '_').replace('-', '_')
        df = pd.DataFrame({
            'Pathology': results['labels'],
            'AUROC': results['auroc_scores'],
            'AUPRC': results['auprc_scores']
        })
        df.to_csv(output_dir / f'{model_name}_metrics.csv', index=False)
    
    # Save summary report
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DenseNet-121 + Vision Transformer Ensemble Evaluation Report\n")
        f.write("="*80 + "\n\n")
        
        for results in results_list:
            f.write(f"\n{results['model_name']}\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean AUROC: {results['mean_auroc']:.4f}\n")
            f.write(f"Mean AUPRC: {results['mean_auprc']:.4f}\n")
            f.write("\nPer-class metrics:\n")
            for label, auroc, auprc in zip(results['labels'], 
                                          results['auroc_scores'], 
                                          results['auprc_scores']):
                if not np.isnan(auroc):
                    f.write(f"  {label:20s} AUROC: {auroc:.4f}  AUPRC: {auprc:.4f}\n")
            f.write("\n")
    
    print(f"✓ Results saved to {output_dir}")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║          DENSENET-121 + VISION TRANSFORMER ENSEMBLE EVALUATION            ║
╚═══════════════════════════════════════════════════════════════════════════╝
    """)
    
    device = get_device()
    print(f"Device: {device}\n")
    
    # Create validation dataset
    print("="*80)
    print("LOADING VALIDATION DATA")
    print("="*80)
    
    valid_csv = CHEXPERT_ROOT / "valid" / "valid.csv"
    val_transform = get_val_transforms()
    val_dataset = CheXpertDataset(valid_csv, DATA_ROOT, val_transform, "ignore", False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"✓ Validation samples: {len(val_dataset):,}")
    
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
    
    # 2. Vision Transformer
    print("\n📥 Loading Vision Transformer...")
    vit_path = 'models/densenet_vit_stacking/base_models/vit_b_16_best.pth'
    vit = create_model("vit_b_16", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(vit_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        vit.load_state_dict(checkpoint['model_state_dict'])
    else:
        vit.load_state_dict(checkpoint)
    vit = vit.to(device)
    vit.eval()
    models_to_evaluate['Vision Transformer'] = vit
    print("  ✓ Vision Transformer loaded")
    
    # 3. Ensemble
    print("\n📥 Loading Ensemble...")
    ensemble_path = 'models/densenet_vit_stacking/ensemble/ensemble_best.pth'
    ensemble = create_ensemble(
        model1=densenet,
        model2=vit,
        ensemble_type="stacking",
        num_classes=14,
        hidden_dim=64  # Must match training configuration
    )
    checkpoint = torch.load(ensemble_path, map_location=device)
    ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
    ensemble = ensemble.to(device)
    ensemble.eval()
    models_to_evaluate['Ensemble'] = ensemble
    print("  ✓ Ensemble loaded")
    
    # Evaluate all models
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    results_list = []
    for model_name, model in models_to_evaluate.items():
        results = evaluate_model(model, val_loader, device, model_name)
        results_list.append(results)
    
    # Save and visualize results
    output_dir = 'evaluation_results/densenet_vit_evaluation'
    save_results(results_list, output_dir)
    plot_comparison(results_list, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for results in results_list:
        print(f"\n{results['model_name']:25s} Mean AUROC: {results['mean_auroc']:.4f}  Mean AUPRC: {results['mean_auprc']:.4f}")
    
    print("\n" + "="*80)
    print("🎉 EVALUATION COMPLETED!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
