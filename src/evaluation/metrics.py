"""
Evaluation metrics for DiagXNet-Lite following project proposal specifications:
- AUROC, AUPRC, F1 per label using Youden's J thresholds
- Calibration analysis with Expected Calibration Error (ECE)
- Temperature scaling to achieve ECE ≤ 0.10
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import CHEXPERT_LABELS, RESULTS_DIR


class ModelEvaluator:
    """
    Comprehensive evaluator for DiagXNet-Lite following project proposal
    """
    
    def __init__(self, model, device, results_dir=None):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            device: torch device
            results_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.results_dir = results_dir or RESULTS_DIR
        self.labels = CHEXPERT_LABELS
        
        # Results storage
        self.predictions = None
        self.targets = None
        self.probabilities = None
        self.masks = None
        
    def predict(self, data_loader, return_raw=False):
        """
        Generate predictions for a dataset
        
        Args:
            data_loader: DataLoader for the dataset
            return_raw: If True, return raw logits instead of probabilities
            
        Returns:
            predictions, targets, probabilities, masks
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        all_masks = []
        
        print("Generating predictions...")
        with torch.no_grad():
            for batch_idx, (images, labels, masks) in enumerate(data_loader):
                if batch_idx % 50 == 0:
                    print(f"Processing batch {batch_idx}/{len(data_loader)}")
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                all_masks.append(masks.cpu())
        
        # Concatenate all batches
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Convert to probabilities unless raw requested
        if return_raw:
            probabilities = all_outputs
        else:
            probabilities = torch.sigmoid(all_outputs)
        
        # Store results
        self.probabilities = probabilities.numpy()
        self.targets = all_labels.numpy()
        self.masks = all_masks.numpy()
        
        print(f"Predictions generated for {len(all_outputs)} samples")
        return self.probabilities, self.targets, self.masks
    
    def find_optimal_thresholds(self, probabilities=None, targets=None, masks=None):
        """
        Find optimal thresholds using Youden's J statistic as specified in proposal
        
        Args:
            probabilities: Predicted probabilities [n_samples, n_classes]
            targets: True labels [n_samples, n_classes] 
            masks: Valid label masks [n_samples, n_classes]
            
        Returns:
            Dictionary of optimal thresholds per class
        """
        if probabilities is None:
            probabilities = self.probabilities
        if targets is None:
            targets = self.targets
        if masks is None:
            masks = self.masks
            
        optimal_thresholds = {}
        
        print("Finding optimal thresholds using Youden's J statistic...")
        
        for i, label in enumerate(self.labels):
            # Get valid samples for this label
            valid_mask = masks[:, i] == 1
            
            if valid_mask.sum() == 0:
                optimal_thresholds[label] = 0.5
                continue
                
            y_true = targets[valid_mask, i]
            y_prob = probabilities[valid_mask, i]
            
            # Skip if all samples are one class
            if len(np.unique(y_true)) < 2:
                optimal_thresholds[label] = 0.5
                continue
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            # Youden's J statistic = TPR - FPR
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            optimal_thresholds[label] = optimal_threshold
            
        print(f"Optimal thresholds calculated for {len(optimal_thresholds)} labels")
        return optimal_thresholds
    
    def calculate_classification_metrics(self, thresholds=None):
        """
        Calculate AUROC, AUPRC, F1 per label as specified in proposal
        
        Args:
            thresholds: Dictionary of thresholds per class (if None, uses Youden's J)
            
        Returns:
            DataFrame with metrics per label
        """
        if self.probabilities is None:
            raise ValueError("Must call predict() first")
            
        if thresholds is None:
            thresholds = self.find_optimal_thresholds()
            
        metrics_data = []
        
        print("Calculating classification metrics...")
        
        for i, label in enumerate(self.labels):
            # Get valid samples
            valid_mask = self.masks[:, i] == 1
            
            if valid_mask.sum() == 0:
                # No valid samples for this label
                metrics_data.append({
                    'Label': label,
                    'AUROC': np.nan,
                    'AUPRC': np.nan, 
                    'F1': np.nan,
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'Threshold': np.nan,
                    'Valid_Samples': 0
                })
                continue
                
            y_true = self.targets[valid_mask, i]
            y_prob = self.probabilities[valid_mask, i]
            threshold = thresholds.get(label, 0.5)
            y_pred = (y_prob >= threshold).astype(int)
            
            # Skip if all samples are one class
            if len(np.unique(y_true)) < 2:
                auroc = np.nan
                auprc = np.nan
            else:
                # AUROC
                auroc = roc_auc_score(y_true, y_prob)
                
                # AUPRC (Average Precision)
                auprc = average_precision_score(y_true, y_prob)
            
            # F1, Precision, Recall, Accuracy
            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            
            metrics_data.append({
                'Label': label,
                'AUROC': auroc,
                'AUPRC': auprc,
                'F1': f1,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'Threshold': threshold,
                'Valid_Samples': valid_mask.sum()
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Calculate macro averages (proposal target: macro AUROC ≥ 0.80)
        macro_auroc = metrics_df['AUROC'].mean()
        macro_auprc = metrics_df['AUPRC'].mean()
        macro_f1 = metrics_df['F1'].mean()
        
        print(f"\n📊 Classification Metrics Summary:")
        print(f"   Macro AUROC: {macro_auroc:.4f} (Target: ≥ 0.80)")
        print(f"   Macro AUPRC: {macro_auprc:.4f}")
        print(f"   Macro F1: {macro_f1:.4f}")
        
        # Add summary row
        summary_row = {
            'Label': 'MACRO_AVERAGE',
            'AUROC': macro_auroc,
            'AUPRC': macro_auprc,
            'F1': macro_f1,
            'Accuracy': metrics_df['Accuracy'].mean(),
            'Precision': metrics_df['Precision'].mean(),
            'Recall': metrics_df['Recall'].mean(),
            'Threshold': np.nan,
            'Valid_Samples': metrics_df['Valid_Samples'].sum()
        }
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        return metrics_df
    
    def calculate_calibration_metrics(self, n_bins=10):
        """
        Calculate calibration metrics including Expected Calibration Error (ECE)
        Target from proposal: ECE ≤ 0.10
        
        Args:
            n_bins: Number of bins for calibration
            
        Returns:
            Dictionary with calibration metrics per label
        """
        if self.probabilities is None:
            raise ValueError("Must call predict() first")
            
        calibration_data = []
        
        print("Calculating calibration metrics...")
        
        for i, label in enumerate(self.labels):
            # Get valid samples
            valid_mask = self.masks[:, i] == 1
            
            if valid_mask.sum() == 0:
                calibration_data.append({
                    'Label': label,
                    'ECE': np.nan,
                    'MCE': np.nan,
                    'Brier_Score': np.nan,
                    'Valid_Samples': 0
                })
                continue
                
            y_true = self.targets[valid_mask, i]
            y_prob = self.probabilities[valid_mask, i]
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(y_true, y_prob, n_bins)
            
            # Maximum Calibration Error (MCE)
            mce = self._calculate_mce(y_true, y_prob, n_bins)
            
            # Brier Score
            brier_score = np.mean((y_prob - y_true) ** 2)
            
            calibration_data.append({
                'Label': label,
                'ECE': ece,
                'MCE': mce,
                'Brier_Score': brier_score,
                'Valid_Samples': valid_mask.sum()
            })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        # Calculate averages
        avg_ece = calibration_df['ECE'].mean()
        avg_mce = calibration_df['MCE'].mean()
        avg_brier = calibration_df['Brier_Score'].mean()
        
        print(f"\n🎯 Calibration Metrics Summary:")
        print(f"   Average ECE: {avg_ece:.4f} (Target: ≤ 0.10)")
        print(f"   Average MCE: {avg_mce:.4f}")
        print(f"   Average Brier Score: {avg_brier:.4f}")
        
        return calibration_df
    
    def _calculate_ece(self, y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample is in bin m (between bin_lower and bin_upper)
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            # Convert to torch tensor if it's numpy array
            if isinstance(in_bin, np.ndarray):
                in_bin = torch.from_numpy(in_bin)
            if isinstance(y_true, np.ndarray):
                y_true_torch = torch.from_numpy(y_true)
            else:
                y_true_torch = y_true
            if isinstance(y_prob, np.ndarray):
                y_prob_torch = torch.from_numpy(y_prob)
            else:
                y_prob_torch = y_prob
                
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = y_true_torch[in_bin].float().mean()
                avg_confidence_in_bin = y_prob_torch[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def _calculate_mce(self, y_true, y_prob, n_bins=10):
        """Calculate Maximum Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            
            # Convert to torch tensor if it's numpy array
            if isinstance(in_bin, np.ndarray):
                in_bin = torch.from_numpy(in_bin)
            if isinstance(y_true, np.ndarray):
                y_true_torch = torch.from_numpy(y_true)
            else:
                y_true_torch = y_true
            if isinstance(y_prob, np.ndarray):
                y_prob_torch = torch.from_numpy(y_prob)
            else:
                y_prob_torch = y_prob
                
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = y_true_torch[in_bin].float().mean()
                avg_confidence_in_bin = y_prob_torch[in_bin].mean()
                mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin).item())
        
        return mce
    
    def temperature_scaling(self, val_loader, target_ece=0.10):
        """
        Apply temperature scaling to achieve ECE ≤ 0.10 as specified in proposal
        
        Args:
            val_loader: Validation data loader for optimization
            target_ece: Target ECE (default 0.10 from proposal)
            
        Returns:
            Optimal temperature, calibrated probabilities
        """
        print("Applying temperature scaling for calibration...")
        
        # Get raw logits
        raw_logits, targets, masks = self.predict(val_loader, return_raw=True)
        
        def temperature_scale_loss(temperature):
            """Negative log likelihood loss with temperature scaling"""
            scaled_logits = raw_logits / temperature
            scaled_probs = torch.sigmoid(torch.tensor(scaled_logits))
            
            # Calculate average ECE across all labels
            total_ece = 0
            valid_labels = 0
            
            for i in range(len(self.labels)):
                valid_mask = masks[:, i] == 1
                if valid_mask.sum() == 0:
                    continue
                    
                y_true = targets[valid_mask, i]
                y_prob = scaled_probs[valid_mask, i]
                
                ece = self._calculate_ece(y_true, y_prob)
                total_ece += ece
                valid_labels += 1
            
            avg_ece = total_ece / max(valid_labels, 1)
            return avg_ece
        
        # Optimize temperature
        result = minimize_scalar(
            temperature_scale_loss,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_temperature = result.x
        final_ece = result.fun
        
        print(f"   Optimal temperature: {optimal_temperature:.4f}")
        print(f"   Final ECE: {final_ece:.4f}")
        
        # Apply temperature scaling to get calibrated probabilities
        calibrated_logits = raw_logits / optimal_temperature
        calibrated_probs = torch.sigmoid(torch.tensor(calibrated_logits)).numpy()
        
        # Update stored probabilities
        self.calibrated_probabilities = calibrated_probs
        
        return optimal_temperature, calibrated_probs
    
    def plot_roc_curves(self, save_path=None):
        """Plot ROC curves for all labels"""
        if self.probabilities is None:
            raise ValueError("Must call predict() first")
            
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, label in enumerate(self.labels):
            valid_mask = self.masks[:, i] == 1
            
            if valid_mask.sum() == 0 or len(np.unique(self.targets[valid_mask, i])) < 2:
                axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[i].set_title(f'{label}\n(No data)')
                continue
                
            y_true = self.targets[valid_mask, i]
            y_prob = self.probabilities[valid_mask, i]
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = auc(fpr, tpr)
            
            axes[i].plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{label}\nAUROC = {auroc:.3f}')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved: {save_path}")
        
        plt.show()
    
    def plot_calibration_curves(self, save_path=None):
        """Plot calibration curves for all labels"""
        if self.probabilities is None:
            raise ValueError("Must call predict() first")
            
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()
        
        for i, label in enumerate(self.labels):
            valid_mask = self.masks[:, i] == 1
            
            if valid_mask.sum() == 0:
                axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[i].set_title(f'{label}\n(No data)')
                continue
                
            y_true = self.targets[valid_mask, i]
            y_prob = self.probabilities[valid_mask, i]
            
            # Ensure probabilities are in [0, 1] range
            y_prob = np.clip(y_prob, 0.0, 1.0)
            
            if len(np.unique(y_true)) < 2:
                axes[i].text(0.5, 0.5, 'Single class', ha='center', va='center')
                axes[i].set_title(f'{label}\n(Single class)')
                continue
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=10
            )
            
            axes[i].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
            axes[i].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            axes[i].set_xlabel('Mean Predicted Probability')
            axes[i].set_ylabel('Fraction of Positives')
            axes[i].set_title(f'{label}')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration curves saved: {save_path}")
        
        plt.show()
    
    def generate_full_evaluation_report(self, val_loader, save_dir=None):
        """
        Generate complete evaluation report following proposal specifications
        
        Args:
            val_loader: Validation data loader
            save_dir: Directory to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        if save_dir is None:
            save_dir = self.results_dir
            
        print("\n" + "="*60)
        print("DiagXNet-Lite: Full Evaluation Report")
        print("="*60)
        
        # 1. Generate predictions
        probabilities, targets, masks = self.predict(val_loader)
        
        # 2. Classification metrics with Youden's J thresholds
        thresholds = self.find_optimal_thresholds()
        classification_metrics = self.calculate_classification_metrics(thresholds)
        
        # 3. Calibration analysis
        calibration_metrics = self.calculate_calibration_metrics()
        
        # 4. Temperature scaling
        optimal_temp, calibrated_probs = self.temperature_scaling(val_loader)
        
        # 5. Plots
        self.plot_roc_curves(save_dir / "roc_curves.png")
        self.plot_calibration_curves(save_dir / "calibration_curves.png")
        
        # 6. Save results
        classification_metrics.to_csv(save_dir / "classification_metrics.csv", index=False)
        calibration_metrics.to_csv(save_dir / "calibration_metrics.csv", index=False)
        
        # Save thresholds
        thresholds_df = pd.DataFrame(list(thresholds.items()), columns=['Label', 'Threshold'])
        thresholds_df.to_csv(save_dir / "optimal_thresholds.csv", index=False)
        
        # Summary report
        results = {
            'classification_metrics': classification_metrics,
            'calibration_metrics': calibration_metrics,
            'optimal_thresholds': thresholds,
            'temperature_scaling': {
                'optimal_temperature': optimal_temp,
                'calibrated_probabilities': calibrated_probs
            },
            'proposal_targets': {
                'macro_auroc_target': 0.80,
                'macro_auroc_achieved': classification_metrics[classification_metrics['Label'] == 'MACRO_AVERAGE']['AUROC'].iloc[0],
                'ece_target': 0.10,
                'ece_achieved': calibration_metrics['ECE'].mean()
            }
        }
        
        # Check if targets met
        macro_auroc = results['proposal_targets']['macro_auroc_achieved']
        avg_ece = results['proposal_targets']['ece_achieved']
        
        print(f"\n🎯 Proposal Targets Assessment:")
        print(f"   Macro AUROC ≥ 0.80: {'✅ ACHIEVED' if macro_auroc >= 0.80 else '❌ NOT MET'} ({macro_auroc:.4f})")
        print(f"   ECE ≤ 0.10: {'✅ ACHIEVED' if avg_ece <= 0.10 else '❌ NOT MET'} ({avg_ece:.4f})")
        
        print(f"\n💾 Results saved in: {save_dir}")
        print("="*60)
        
        return results


def main():
    """Test the evaluator"""
    print("DiagXNet-Lite Evaluation Module")
    print("This module provides comprehensive evaluation following the project proposal")


if __name__ == "__main__":
    main()
