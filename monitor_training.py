#!/usr/bin/env python3
"""
Monitor ensemble training progress
Run this in a separate terminal while training is ongoing
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_training_progress():
    """Monitor training logs and model files"""
    
    models_dir = Path("models")
    
    print("\n" + "="*70)
    print(f"{'ENSEMBLE TRAINING MONITOR':^70}")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check for model files
    print("📁 Model Files:")
    print("-" * 70)
    
    densenet_model = models_dir / "densenet121_chexpert_20250906_195712_best.pth"
    inception_model = models_dir / "inception_resnet_v2_best.pth"
    ensemble_model = models_dir / "ensemble_best.pth"
    
    if densenet_model.exists():
        size_mb = densenet_model.stat().st_size / (1024 * 1024)
        print(f"  ✅ DenseNet-121:        {size_mb:.1f} MB (pre-existing)")
    else:
        print(f"  ❌ DenseNet-121:        Not found")
    
    if inception_model.exists():
        size_mb = inception_model.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(inception_model.stat().st_mtime)
        print(f"  🔄 Inception-ResNet-V2: {size_mb:.1f} MB (last updated: {mod_time.strftime('%H:%M:%S')})")
    else:
        print(f"  ⏳ Inception-ResNet-V2: Training in progress...")
    
    if ensemble_model.exists():
        size_mb = ensemble_model.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(ensemble_model.stat().st_mtime)
        print(f"  ✅ Ensemble:            {size_mb:.1f} MB (last updated: {mod_time.strftime('%H:%M:%S')})")
    else:
        print(f"  ⏳ Ensemble:            Not yet created")
    
    print()
    
    # Training status
    print("📊 Training Status:")
    print("-" * 70)
    
    if not inception_model.exists():
        print("  🏋️  Stage 1: Training Inception-ResNet-V2...")
        print("     Expected: 3-4 hours for 5 epochs")
    elif not ensemble_model.exists():
        print("  ✅ Stage 1: Inception-ResNet-V2 completed!")
        print("  🧠 Stage 2: Training meta-learner...")
        print("     Expected: 30 minutes for 3 epochs")
    else:
        print("  ✅ Stage 1: Inception-ResNet-V2 completed!")
        print("  ✅ Stage 2: Meta-learner completed!")
        print("  🎉 Training finished!")
    
    print()
    print("="*70)
    print("\n💡 Tip: Check the training terminal for detailed progress")
    print("   Or run: tail -f <training_output.log> if you're logging to file")
    print()


def main():
    """Main monitoring loop"""
    try:
        print("\n🔍 Monitoring ensemble training...")
        print("Press Ctrl+C to stop monitoring\n")
        
        while True:
            check_training_progress()
            
            # Check if training is complete
            ensemble_model = Path("models/ensemble_best.pth")
            if ensemble_model.exists():
                print("\n🎉 Training completed! Stopping monitor.")
                break
            
            print("Refreshing in 60 seconds...")
            time.sleep(60)
    
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
