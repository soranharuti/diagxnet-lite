"""
Enhanced Evaluation Script for DiagXNet-Lite
Creates comprehensive analysis including confusion matrices, clinical insights, and failure analysis
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append('/Users/soranharuti/Desktop/diagxnet-lite')

from configs.config import CHEXPERT_LABELS, get_device, RESULTS_DIR
from src.data.dataset import CheXpertDataset, get_val_transforms, create_data_loaders
from src.models.architectures import create_model
from src.evaluation.metrics import ModelEvaluator

class EnhancedEvaluator:
    """Comprehensive model evaluation with clinical insights"""
    
    def __init__(self, model_path, results_dir):
        self.device = get_device()
        self.results_dir = Path(results_dir)
        self.enhanced_dir = self.results_dir / "enhanced_analysis"
        self.enhanced_dir.mkdir(exist_ok=True)
        
        # Load model
        print("🔧 Loading trained model...")
        self.model = create_model('densenet121', num_classes=14, pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Setup data
        print("📊 Setting up validation data...")
        train_csv = "/Users/soranharuti/Desktop/diagxnet-lite/data/chexpert_small/CheXpert-v1.0-small/train/train.csv"
        _, self.val_loader = create_data_loaders(
            train_csv=train_csv,
            batch_size=32,
            num_workers=4,
            uncertainty_policy="ignore",
            val_split=0.2
        )
        
        self.labels = CHEXPERT_LABELS
        
    def get_predictions_and_targets(self):
        """Get model predictions and true labels"""
        print("🔍 Generating predictions...")
        
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch_idx, (images, targets, masks) in enumerate(self.val_loader):
                if batch_idx % 50 == 0:
                    print(f"   Processing batch {batch_idx}/{len(self.val_loader)}")
                
                images = images.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs)
                
                all_predictions.append(probabilities.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_masks.append(masks.cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        masks = np.vstack(all_masks)
        
        print(f"✅ Generated predictions for {len(predictions)} samples")
        return predictions, targets, masks
    
    def create_confusion_matrices(self, predictions, targets, masks):
        """Create confusion matrices for each condition"""
        print("📈 Creating confusion matrices...")
        
        # Load optimal thresholds
        thresholds_df = pd.read_csv(self.results_dir / "optimal_thresholds.csv")
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        
        confusion_data = []
        
        for i, label in enumerate(self.labels):
            # Get threshold for this label
            threshold = thresholds_df[thresholds_df['Label'] == label]['Threshold'].iloc[0]
            
            # Get valid samples (where mask = 1)
            valid_mask = masks[:, i] == 1
            if valid_mask.sum() == 0:
                continue
                
            y_true = targets[valid_mask, i]
            y_pred_prob = predictions[valid_mask, i]
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Handle edge cases
            if cm.shape == (1, 1):
                if y_true[0] == 0:
                    cm = np.array([[cm[0, 0], 0], [0, 0]])  # All negatives
                else:
                    cm = np.array([[0, 0], [0, cm[0, 0]]])  # All positives
            elif cm.shape == (2, 1):
                cm = np.column_stack([cm, np.zeros(2)])
            elif cm.shape == (1, 2):
                cm = np.vstack([cm, np.zeros(2)])
            
            # Store confusion matrix data
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
            
            confusion_data.append({
                'Label': label,
                'True_Negatives': tn,
                'False_Positives': fp,
                'False_Negatives': fn,
                'True_Positives': tp,
                'Total_Samples': valid_mask.sum(),
                'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
            })
            
            # Plot confusion matrix
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            ax.set_title(f'{label}\n(n={valid_mask.sum()})', fontsize=10)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Remove empty subplots
        for j in range(len(self.labels), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(self.enhanced_dir / "confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix data
        confusion_df = pd.DataFrame(confusion_data)
        confusion_df.to_csv(self.enhanced_dir / "confusion_matrix_analysis.csv", index=False)
        
        print(f"✅ Saved confusion matrices to {self.enhanced_dir / 'confusion_matrices.png'}")
        return confusion_df
    
    def clinical_significance_analysis(self, confusion_df):
        """Analyze clinical significance of different conditions"""
        print("🏥 Performing clinical significance analysis...")
        
        # Define clinical urgency levels
        clinical_urgency = {
            'Critical': ['Pneumothorax', 'Pneumonia'],
            'Urgent': ['Pleural Effusion', 'Consolidation', 'Edema'],
            'Moderate': ['Cardiomegaly', 'Atelectasis', 'Fracture'],
            'Low': ['Lung Opacity', 'Lung Lesion', 'Support Devices', 'Pleural Other'],
            'Normal': ['No Finding'],
            'Structural': ['Enlarged Cardiomediastinum']
        }
        
        # Add urgency levels to confusion data
        urgency_map = {}
        for urgency, conditions in clinical_urgency.items():
            for condition in conditions:
                urgency_map[condition] = urgency
        
        confusion_df['Clinical_Urgency'] = confusion_df['Label'].map(urgency_map)
        
        # Calculate clinical impact metrics
        confusion_df['Critical_Miss_Rate'] = confusion_df['False_Negatives'] / (
            confusion_df['True_Positives'] + confusion_df['False_Negatives']
        )
        confusion_df['False_Alarm_Rate'] = confusion_df['False_Positives'] / (
            confusion_df['True_Negatives'] + confusion_df['False_Positives']
        )
        
        # Clinical priority scoring
        urgency_weights = {
            'Critical': 5.0,
            'Urgent': 4.0, 
            'Moderate': 3.0,
            'Low': 2.0,
            'Normal': 1.0,
            'Structural': 2.5
        }
        
        confusion_df['Urgency_Weight'] = confusion_df['Clinical_Urgency'].map(urgency_weights)
        confusion_df['Clinical_Impact_Score'] = (
            confusion_df['Urgency_Weight'] * 
            (1 - confusion_df['Critical_Miss_Rate'].fillna(0))
        )
        
        # Create clinical analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Clinical urgency distribution
        urgency_counts = confusion_df['Clinical_Urgency'].value_counts()
        ax1.pie(urgency_counts.values, labels=urgency_counts.index, autopct='%1.1f%%')
        ax1.set_title('Distribution by Clinical Urgency')
        
        # Miss rates by urgency
        miss_by_urgency = confusion_df.groupby('Clinical_Urgency')['Critical_Miss_Rate'].mean()
        ax2.bar(miss_by_urgency.index, miss_by_urgency.values)
        ax2.set_title('Miss Rate by Clinical Urgency')
        ax2.set_ylabel('False Negative Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        # Clinical impact scores
        top_impact = confusion_df.nlargest(10, 'Clinical_Impact_Score')
        ax3.barh(top_impact['Label'], top_impact['Clinical_Impact_Score'])
        ax3.set_title('Clinical Impact Scores (Higher = Better)')
        ax3.set_xlabel('Impact Score')
        
        # Sensitivity vs Specificity by urgency
        colors = {'Critical': 'red', 'Urgent': 'orange', 'Moderate': 'yellow', 
                 'Low': 'green', 'Normal': 'blue', 'Structural': 'purple'}
        for urgency in confusion_df['Clinical_Urgency'].unique():
            subset = confusion_df[confusion_df['Clinical_Urgency'] == urgency]
            ax4.scatter(subset['Specificity'], subset['Sensitivity'], 
                       label=urgency, c=colors.get(urgency, 'gray'), s=100, alpha=0.7)
        
        ax4.set_xlabel('Specificity')
        ax4.set_ylabel('Sensitivity')
        ax4.set_title('Sensitivity vs Specificity by Clinical Urgency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.enhanced_dir / "clinical_significance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save clinical analysis
        clinical_summary = confusion_df.groupby('Clinical_Urgency').agg({
            'Sensitivity': 'mean',
            'Specificity': 'mean',
            'Critical_Miss_Rate': 'mean',
            'False_Alarm_Rate': 'mean',
            'Clinical_Impact_Score': 'mean',
            'Total_Samples': 'sum'
        }).round(3)
        
        clinical_summary.to_csv(self.enhanced_dir / "clinical_urgency_summary.csv")
        
        print(f"✅ Clinical analysis saved to {self.enhanced_dir}")
        return confusion_df
    
    def failure_case_analysis(self, predictions, targets, masks):
        """Analyze common failure patterns"""
        print("🔍 Analyzing failure cases...")
        
        # Load optimal thresholds
        thresholds_df = pd.read_csv(self.results_dir / "optimal_thresholds.csv")
        
        failure_analysis = []
        
        for i, label in enumerate(self.labels):
            threshold = thresholds_df[thresholds_df['Label'] == label]['Threshold'].iloc[0]
            
            valid_mask = masks[:, i] == 1
            if valid_mask.sum() == 0:
                continue
            
            y_true = targets[valid_mask, i]
            y_pred_prob = predictions[valid_mask, i]
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            # Find failure cases
            false_positives = (y_true == 0) & (y_pred == 1)
            false_negatives = (y_true == 1) & (y_pred == 0)
            
            # Confidence analysis for failures
            fp_confidences = y_pred_prob[false_positives] if false_positives.sum() > 0 else []
            fn_confidences = y_pred_prob[false_negatives] if false_negatives.sum() > 0 else []
            
            failure_analysis.append({
                'Label': label,
                'False_Positives': false_positives.sum(),
                'False_Negatives': false_negatives.sum(),
                'FP_Mean_Confidence': np.mean(fp_confidences) if len(fp_confidences) > 0 else 0,
                'FN_Mean_Confidence': np.mean(fn_confidences) if len(fn_confidences) > 0 else 0,
                'FP_High_Confidence': (np.array(fp_confidences) > 0.8).sum() if len(fp_confidences) > 0 else 0,
                'FN_Low_Confidence': (np.array(fn_confidences) < 0.2).sum() if len(fn_confidences) > 0 else 0,
                'Total_Valid_Samples': valid_mask.sum()
            })
        
        failure_df = pd.DataFrame(failure_analysis)
        failure_df['FP_Rate'] = failure_df['False_Positives'] / failure_df['Total_Valid_Samples']
        failure_df['FN_Rate'] = failure_df['False_Negatives'] / failure_df['Total_Valid_Samples']
        
        # Create failure analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # False positive and negative rates
        x = np.arange(len(failure_df))
        width = 0.35
        ax1.bar(x - width/2, failure_df['FP_Rate'], width, label='False Positive Rate', alpha=0.7)
        ax1.bar(x + width/2, failure_df['FN_Rate'], width, label='False Negative Rate', alpha=0.7)
        ax1.set_xlabel('Conditions')
        ax1.set_ylabel('Error Rate')
        ax1.set_title('False Positive vs False Negative Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(failure_df['Label'], rotation=45, ha='right')
        ax1.legend()
        
        # Confidence analysis for failures
        ax2.scatter(failure_df['FP_Mean_Confidence'], failure_df['FN_Mean_Confidence'])
        for i, label in enumerate(failure_df['Label']):
            ax2.annotate(label, (failure_df['FP_Mean_Confidence'].iloc[i], 
                               failure_df['FN_Mean_Confidence'].iloc[i]), fontsize=8)
        ax2.set_xlabel('Mean FP Confidence')
        ax2.set_ylabel('Mean FN Confidence')
        ax2.set_title('Failure Confidence Analysis')
        ax2.grid(True, alpha=0.3)
        
        # High confidence errors (concerning)
        concerning_errors = failure_df[['Label', 'FP_High_Confidence', 'FN_Low_Confidence']].copy()
        concerning_errors['Total_Concerning'] = concerning_errors['FP_High_Confidence'] + concerning_errors['FN_Low_Confidence']
        concerning_errors = concerning_errors.sort_values('Total_Concerning', ascending=True)
        
        ax3.barh(concerning_errors['Label'], concerning_errors['Total_Concerning'])
        ax3.set_xlabel('Number of High-Confidence Errors')
        ax3.set_title('Concerning Errors (High Confidence Mistakes)')
        
        # Error patterns
        total_errors = failure_df['False_Positives'] + failure_df['False_Negatives']
        error_sorted = failure_df.nlargest(10, 'False_Positives')
        ax4.bar(range(len(error_sorted)), error_sorted['False_Positives'], alpha=0.7, label='False Positives')
        ax4.bar(range(len(error_sorted)), error_sorted['False_Negatives'], alpha=0.7, label='False Negatives')
        ax4.set_xlabel('Top Error-Prone Conditions')
        ax4.set_ylabel('Number of Errors')
        ax4.set_title('Conditions with Most Errors')
        ax4.set_xticks(range(len(error_sorted)))
        ax4.set_xticklabels(error_sorted['Label'], rotation=45, ha='right')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.enhanced_dir / "failure_case_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save failure analysis
        failure_df.to_csv(self.enhanced_dir / "failure_case_analysis.csv", index=False)
        
        print(f"✅ Failure analysis saved to {self.enhanced_dir}")
        return failure_df
    
    def generate_summary_report(self, confusion_df, failure_df):
        """Generate comprehensive summary report"""
        print("📝 Generating summary report...")
        
        # Load existing metrics
        metrics_df = pd.read_csv(self.results_dir / "classification_metrics.csv")
        metrics_df = metrics_df[metrics_df['Label'] != 'MACRO_AVERAGE']
        
        # Merge all analyses
        summary = metrics_df.merge(confusion_df[['Label', 'Clinical_Urgency', 'Clinical_Impact_Score']], on='Label')
        summary = summary.merge(failure_df[['Label', 'FP_Rate', 'FN_Rate', 'FP_High_Confidence', 'FN_Low_Confidence']], on='Label')
        
        # Create executive summary
        report_lines = [
            "# DiagXNet-Lite Enhanced Evaluation Report",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"**Overall Performance**: Mean AUROC = {summary['AUROC'].mean():.3f}",
            f"**Best Performing**: {summary.loc[summary['AUROC'].idxmax(), 'Label']} (AUROC: {summary['AUROC'].max():.3f})",
            f"**Most Challenging**: {summary.loc[summary['AUROC'].idxmin(), 'Label']} (AUROC: {summary['AUROC'].min():.3f})",
            "",
            "## Clinical Impact Analysis",
            "",
            "### High Clinical Impact Conditions (Top 5):",
        ]
        
        top_clinical = summary.nlargest(5, 'Clinical_Impact_Score')
        for _, row in top_clinical.iterrows():
            report_lines.append(f"- **{row['Label']}**: Impact Score {row['Clinical_Impact_Score']:.3f}, AUROC {row['AUROC']:.3f}")
        
        report_lines.extend([
            "",
            "## Key Findings",
            "",
            "### Strengths:",
        ])
        
        strengths = summary[summary['AUROC'] > 0.80]
        for _, row in strengths.iterrows():
            report_lines.append(f"- {row['Label']}: Excellent performance (AUROC: {row['AUROC']:.3f})")
        
        report_lines.extend([
            "",
            "### Areas for Improvement:",
        ])
        
        improvements = summary[summary['AUROC'] < 0.70]
        for _, row in improvements.iterrows():
            report_lines.append(f"- {row['Label']}: AUROC {row['AUROC']:.3f}, High FP Rate: {row['FP_Rate']:.3f}")
        
        report_lines.extend([
            "",
            "### Concerning Error Patterns:",
        ])
        
        concerning = summary[(summary['FP_High_Confidence'] > 5) | (summary['FN_Low_Confidence'] > 5)]
        for _, row in concerning.iterrows():
            report_lines.append(f"- {row['Label']}: {row['FP_High_Confidence']} high-confidence false positives, {row['FN_Low_Confidence']} low-confidence false negatives")
        
        # Save report
        with open(self.enhanced_dir / "enhanced_evaluation_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save comprehensive data
        summary.to_csv(self.enhanced_dir / "comprehensive_analysis.csv", index=False)
        
        print(f"✅ Summary report saved to {self.enhanced_dir}")
        
    def run_complete_analysis(self):
        """Run the complete enhanced evaluation"""
        print("🚀 Starting Enhanced Evaluation Analysis")
        print("="*60)
        
        # Get predictions
        predictions, targets, masks = self.get_predictions_and_targets()
        
        # Create confusion matrices
        confusion_df = self.create_confusion_matrices(predictions, targets, masks)
        
        # Clinical significance analysis
        confusion_df = self.clinical_significance_analysis(confusion_df)
        
        # Failure case analysis
        failure_df = self.failure_case_analysis(predictions, targets, masks)
        
        # Generate summary report
        self.generate_summary_report(confusion_df, failure_df)
        
        print("="*60)
        print(f"🎉 Enhanced evaluation complete! Results saved to:")
        print(f"   📁 {self.enhanced_dir}")
        print(f"   📊 Confusion matrices: confusion_matrices.png")
        print(f"   🏥 Clinical analysis: clinical_significance_analysis.png") 
        print(f"   🔍 Failure analysis: failure_case_analysis.png")
        print(f"   📝 Summary report: enhanced_evaluation_report.md")

def main():
    """Run enhanced evaluation"""
    # Find the latest experiment results
    results_base = Path("/Users/soranharuti/Desktop/diagxnet-lite/results")
    experiment_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith("diagxnet_lite_experiment")]
    
    if not experiment_dirs:
        print("❌ No experiment results found!")
        return
    
    latest_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_dir / "trained_model.pth"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📁 Using results from: {latest_dir}")
    
    # Run enhanced evaluation
    evaluator = EnhancedEvaluator(model_path, latest_dir)
    evaluator.run_complete_analysis()

if __name__ == "__main__":
    main()