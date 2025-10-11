"""
Continue DiagXNet-Lite experiment from evaluation step
Since training is complete, just run evaluation and Grad-CAM analysis
"""

import torch
import torch.utils.data
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
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.gradcam import DiagXNetGradCAMAnalyzer
from src.data.dataset import create_data_loaders
from src.models.architectures import create_model
from configs.config import *

def continue_experiment():
    """Continue from evaluation step"""
    
    # Find the most recent experiment directory
    results_base = Path("/Users/soranharuti/Desktop/diagxnet-lite/results")
    experiment_dirs = [d for d in results_base.iterdir() if d.is_dir() and "diagxnet_lite_experiment" in d.name]
    if not experiment_dirs:
        print("❌ No experiment directories found")
        return False
        
    # Get most recent
    latest_experiment = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Continuing experiment: {latest_experiment.name}")
    
    # Load trained model
    model_path = latest_experiment / "trained_model.pth"
    if not model_path.exists():
        print("❌ Trained model not found")
        return False
    
    print("🔄 Loading trained model...")
    checkpoint = torch.load(model_path, map_location=get_device())
    
    # Create model and load weights
    model = create_model('densenet121').to(get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully")
    
    # Setup data loaders
    print("🔄 Setting up data loaders...")
    train_csv = CHEXPERT_ROOT / "train" / "train.csv"
    train_loader, val_loader = create_data_loaders(
        train_csv=train_csv,
        data_root=DATA_ROOT,
        batch_size=16,
        num_workers=4,
        uncertainty_policy="ignore",
        augment=True,
        val_split=0.2
    )
    print("✅ Data loaders ready")
    
    # Run evaluation
    print("\n" + "="*50)
    print("STEP 3: Comprehensive Evaluation")
    print("="*50)
    
    evaluator = ModelEvaluator(
        model=model,
        device=get_device(),
        results_dir=latest_experiment
    )
    
    # Generate predictions
    probabilities, targets, masks = evaluator.predict(val_loader)
    
    # Run full evaluation
    evaluation_results = evaluator.generate_full_evaluation_report(
        val_loader=val_loader,
        save_dir=latest_experiment
    )
    
    # Check proposal targets
    macro_auroc = evaluation_results['proposal_targets']['macro_auroc_achieved']
    avg_ece = evaluation_results['proposal_targets']['ece_achieved']
    
    print(f"\n🎯 Proposal Target Assessment:")
    print(f"   Macro AUROC ≥ 0.80: {'✅' if macro_auroc >= 0.80 else '❌'} ({macro_auroc:.4f})")
    print(f"   ECE ≤ 0.10: {'✅' if avg_ece <= 0.10 else '❌'} ({avg_ece:.4f})")
    
    # Run Grad-CAM analysis
    print("\n" + "="*50)
    print("STEP 4: Grad-CAM Analysis (12 overlays)")
    print("="*50)
    
    # Create a single-threaded data loader for Grad-CAM to avoid multiprocessing issues
    print("🔄 Setting up single-threaded data loader for Grad-CAM...")
    
    gradcam_data_loader = torch.utils.data.DataLoader(
        val_loader.dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing
        pin_memory=False
    )
    
    gradcam_analyzer = DiagXNetGradCAMAnalyzer(
        model=model,
        device=get_device(),
        data_loader=gradcam_data_loader
    )
    
    # Get predictions and thresholds from evaluation
    thresholds = evaluation_results['optimal_thresholds']
    
    # Run complete Grad-CAM analysis
    gradcam_dir = latest_experiment / "gradcam_analysis"
    gradcam_results = gradcam_analyzer.run_complete_gradcam_analysis(
        probabilities=probabilities,
        targets=targets,
        masks=masks,
        thresholds=thresholds,
        save_dir=gradcam_dir
    )
    
    print(f"✅ Grad-CAM analysis completed")
    
    # Generate final report
    print("\n" + "="*50)
    print("STEP 5: Generating Final Report")
    print("="*50)
    
    # Save final results
    final_results = {
        'experiment_info': {
            'name': latest_experiment.name,
            'device': str(get_device()),
            'model_path': str(model_path)
        },
        'training_results': checkpoint.get('training_results', {}),
        'evaluation_results': evaluation_results,
        'gradcam_results': {
            'total_overlays_generated': sum(
                len(gradcam_results['overlays'][cat]) 
                for cat in gradcam_results['overlays']
            ),
            'save_directory': str(gradcam_results['save_dir'])
        }
    }
    
    # Convert for JSON
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
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
    
    results_file = latest_experiment / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(final_results), f, indent=2)
    
    print(f"✅ Final report generated: {results_file}")
    
    print("\n" + "="*60)
    print("🎉 DiagXNet-Lite Experiment COMPLETED Successfully!")
    print("="*60)
    print(f"📊 Macro AUROC: {macro_auroc:.4f} (Target: ≥ 0.80)")
    print(f"📊 Average ECE: {avg_ece:.4f} (Target: ≤ 0.10)")
    print(f"📁 Results Directory: {latest_experiment}")
    print("="*60)
    
    return True

if __name__ == "__main__":
    continue_experiment()
