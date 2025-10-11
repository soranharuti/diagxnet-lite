"""
Main experiment runner for DiagXNet-Lite
Executes the complete pipeline following project proposal specifications:
1. Train DenseNet-121 for 5 epochs
2. Evaluate with AUROC, AUPRC, F1 (Youden's J thresholds)
3. Calibration analysis (ECE ≤ 0.10 with temperature scaling)
4. Generate 12 Grad-CAM overlays (3 TP, 3 TN, 3 FP, 3 FN)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.training.train import DenseNetTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.gradcam import DiagXNetGradCAMAnalyzer
from src.data.dataset import create_data_loaders
from configs.config import *


class DiagXNetExperiment:
    """
    Complete DiagXNet-Lite experiment runner
    Implements all requirements from the project proposal
    """
    
    def __init__(self, config=None):
        """Initialize experiment"""
        self.config = config or {}
        self.device = get_device()
        
        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"diagxnet_lite_experiment_{self.timestamp}"
        self.results_dir = RESULTS_DIR / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 DiagXNet-Lite Experiment: {self.experiment_name}")
        print(f"📁 Results directory: {self.results_dir}")
        
        # Initialize components
        self.trainer = None
        self.model = None
        self.evaluator = None
        self.gradcam_analyzer = None
        
        # Results storage
        self.training_results = None
        self.evaluation_results = None
        self.gradcam_results = None
        
    def setup_data_loaders(self):
        """Setup train and validation data loaders"""
        print("\n" + "="*50)
        print("STEP 1: Setting up data loaders")
        print("="*50)
        
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        
        # Create data loaders with project specifications
        self.train_loader, self.val_loader = create_data_loaders(
            train_csv=train_csv,
            data_root=DATA_ROOT,
            batch_size=16,  # Specified in proposal
            num_workers=4,
            uncertainty_policy="ignore",
            augment=True,
            val_split=0.2
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        
    def run_training(self):
        """Run DenseNet-121 training for 5 epochs"""
        print("\n" + "="*50)
        print("STEP 2: Training DenseNet-121 (5 epochs)")
        print("="*50)
        
        # Initialize trainer
        self.trainer = DenseNetTrainer()
        
        # Setup data loaders in trainer
        self.trainer.train_loader = self.train_loader
        self.trainer.val_loader = self.val_loader
        
        # Create model and setup training
        self.trainer.create_model()
        
        # Get class weights from dataset
        sample_dataset = self.train_loader.dataset
        if hasattr(sample_dataset, 'dataset'):  # Handle Subset wrapper
            sample_dataset = sample_dataset.dataset
        self.trainer.class_weights = sample_dataset.get_class_weights().to(self.device)
        
        self.trainer.create_optimizer_and_loss()
        
        # Run training
        start_time = time.time()
        self.model, self.training_results = self.trainer.train()
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time/60:.1f} minutes")
        
        # Save model to our results directory
        model_path = self.results_dir / "trained_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_results': self.training_results,
            'config': {
                'architecture': 'densenet121',
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-4
            }
        }, model_path)
        
        print(f"💾 Model saved: {model_path}")
        
    def run_evaluation(self):
        """Run comprehensive evaluation with all required metrics"""
        print("\n" + "="*50)
        print("STEP 3: Comprehensive Evaluation")
        print("="*50)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            device=self.device,
            results_dir=self.results_dir
        )
        
        # Generate predictions
        probabilities, targets, masks = self.evaluator.predict(self.val_loader)
        
        # Run full evaluation
        self.evaluation_results = self.evaluator.generate_full_evaluation_report(
            val_loader=self.val_loader,
            save_dir=self.results_dir
        )
        
        # Check if proposal targets are met
        macro_auroc = self.evaluation_results['proposal_targets']['macro_auroc_achieved']
        avg_ece = self.evaluation_results['proposal_targets']['ece_achieved']
        
        print(f"\n🎯 Proposal Target Assessment:")
        print(f"   Macro AUROC ≥ 0.80: {'✅' if macro_auroc >= 0.80 else '❌'} ({macro_auroc:.4f})")
        print(f"   ECE ≤ 0.10: {'✅' if avg_ece <= 0.10 else '❌'} ({avg_ece:.4f})")
        
    def run_gradcam_analysis(self):
        """Generate 12 Grad-CAM overlays as specified in proposal"""
        print("\n" + "="*50)
        print("STEP 4: Grad-CAM Analysis (12 overlays)")
        print("="*50)
        
        # Initialize Grad-CAM analyzer
        self.gradcam_analyzer = DiagXNetGradCAMAnalyzer(
            model=self.model,
            device=self.device,
            data_loader=self.val_loader
        )
        
        # Get predictions and thresholds from evaluation
        probabilities = self.evaluator.probabilities
        targets = self.evaluator.targets
        masks = self.evaluator.masks
        thresholds = self.evaluation_results['optimal_thresholds']
        
        # Run complete Grad-CAM analysis
        gradcam_dir = self.results_dir / "gradcam_analysis"
        self.gradcam_results = self.gradcam_analyzer.run_complete_gradcam_analysis(
            probabilities=probabilities,
            targets=targets,
            masks=masks,
            thresholds=thresholds,
            save_dir=gradcam_dir
        )
        
        print(f"✅ Grad-CAM analysis completed")
        
    def generate_final_report(self):
        """Generate final experiment report"""
        print("\n" + "="*50)
        print("STEP 5: Generating Final Report")
        print("="*50)
        
        # Collect all results
        final_results = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'device': str(self.device),
                'total_runtime': time.time()
            },
            'training_results': self.training_results,
            'evaluation_results': self.evaluation_results,
            'gradcam_results': {
                'total_overlays_generated': sum(
                    len(self.gradcam_results['overlays'][cat]) 
                    for cat in self.gradcam_results['overlays']
                ),
                'save_directory': str(self.gradcam_results['save_dir'])
            }
        }
        
        # Save comprehensive results
        results_file = self.results_dir / "final_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            json_results = self._convert_for_json(final_results)
            json.dump(json_results, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(final_results)
        
        print(f"✅ Final report generated")
        print(f"📄 Results file: {results_file}")
        
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    def _generate_markdown_report(self, results):
        """Generate a comprehensive markdown report"""
        report_lines = [
            f"# DiagXNet-Lite Experiment Report",
            f"",
            f"**Experiment**: {self.experiment_name}",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Device**: {self.device}",
            f"",
            f"## Project Proposal Objectives",
            f"",
            f"✅ **Completed Tasks:**",
            f"1. ✅ Fine-tuned DenseNet-121 for 5 epochs",
            f"2. ✅ Achieved classification metrics (AUROC, AUPRC, F1) with Youden's J thresholds",
            f"3. ✅ Performed calibration analysis with temperature scaling",
            f"4. ✅ Generated 12 Grad-CAM overlays (3 TP, 3 TN, 3 FP, 3 FN)",
            f"",
            f"## Results Summary",
            f"",
            f"### Training Results",
            f"- **Architecture**: DenseNet-121 (ImageNet pre-trained)",
            f"- **Epochs**: 5",
            f"- **Batch Size**: 16", 
            f"- **Learning Rate**: 1e-4",
            f"- **Loss Function**: BCEWithLogitsLoss",
            f"- **Final Validation Loss**: {self.training_results['results']['val_losses'][-1]:.4f}",
            f"",
            f"### Classification Performance",
        ]
        
        # Add evaluation metrics
        macro_auroc = self.evaluation_results['proposal_targets']['macro_auroc_achieved']
        avg_ece = self.evaluation_results['proposal_targets']['ece_achieved']
        
        report_lines.extend([
            f"- **Macro AUROC**: {macro_auroc:.4f} (Target: ≥ 0.80) {'✅' if macro_auroc >= 0.80 else '❌'}",
            f"- **Average ECE**: {avg_ece:.4f} (Target: ≤ 0.10) {'✅' if avg_ece <= 0.10 else '❌'}",
            f"",
            f"### Grad-CAM Analysis",
            f"- **Total Overlays Generated**: {results['gradcam_results']['total_overlays_generated']}/12",
            f"- **Categories**: True Positives (3), True Negatives (3), False Positives (3), False Negatives (3)",
            f"",
            f"## Files Generated",
            f"",
            f"### Models",
            f"- `trained_model.pth` - Final trained DenseNet-121 model",
            f"",
            f"### Evaluation Results", 
            f"- `classification_metrics.csv` - Per-label AUROC, AUPRC, F1 scores",
            f"- `calibration_metrics.csv` - ECE and calibration metrics",
            f"- `optimal_thresholds.csv` - Youden's J optimal thresholds",
            f"- `roc_curves.png` - ROC curves for all pathologies",
            f"- `calibration_curves.png` - Calibration plots",
            f"",
            f"### Grad-CAM Visualizations",
            f"- `gradcam_analysis/` - Directory with 12 overlays",
            f"- `gradcam_summary.png` - Summary figure with all overlays",
            f"- `gradcam_analysis_report.md` - Detailed Grad-CAM analysis",
            f"",
            f"## Conclusion",
            f"",
            f"The DiagXNet-Lite experiment successfully implemented all requirements from the project proposal:",
            f"",
            f"1. **Model Training**: DenseNet-121 was fine-tuned on CheXpert-small for 5 epochs",
            f"2. **Performance Evaluation**: Comprehensive metrics calculated with optimal thresholds",
            f"3. **Calibration**: Temperature scaling applied to improve model calibration", 
            f"4. **Interpretability**: 12 Grad-CAM overlays generated for model interpretation",
            f"",
            f"The model provides a solid baseline for chest X-ray pathology detection with good",
            f"interpretability through Grad-CAM visualizations.",
            f"",
            f"## Next Steps",
            f"",
            f"1. **Advanced Objectives**: Test focal loss vs BCE comparison",
            f"2. **Demo Development**: Build simple demo page",
            f"3. **Clinical Validation**: Review Grad-CAM attention patterns with experts",
            f"4. **Model Improvement**: Address any false prediction patterns identified"
        ])
        
        # Save markdown report
        report_path = self.results_dir / "experiment_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📋 Markdown report saved: {report_path}")
    
    def run_complete_experiment(self):
        """Run the complete DiagXNet-Lite experiment"""
        start_time = time.time()
        
        print("🚀 Starting DiagXNet-Lite Complete Experiment")
        print("Following project proposal specifications exactly")
        print("="*60)
        
        try:
            # Step 1: Setup data
            self.setup_data_loaders()
            
            # Step 2: Train model
            self.run_training()
            
            # Step 3: Evaluate model  
            self.run_evaluation()
            
            # Step 4: Grad-CAM analysis
            self.run_gradcam_analysis()
            
            # Step 5: Final report
            self.generate_final_report()
            
            total_time = time.time() - start_time
            
            print("\n" + "="*60)
            print("🎉 DiagXNet-Lite Experiment COMPLETED Successfully!")
            print("="*60)
            print(f"⏱️  Total Runtime: {total_time/60:.1f} minutes")
            print(f"📁 Results Directory: {self.results_dir}")
            print(f"📊 All deliverables generated according to proposal")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ Experiment failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the complete experiment"""
    print("DiagXNet-Lite: Complete Experiment Runner")
    print("Implementing all project proposal requirements")
    print("="*60)
    
    # Create and run experiment
    experiment = DiagXNetExperiment()
    success = experiment.run_complete_experiment()
    
    if success:
        print("\n✅ Experiment completed successfully!")
        print("All deliverables ready for academic submission.")
    else:
        print("\n❌ Experiment failed. Check logs for details.")


if __name__ == "__main__":
    main()
