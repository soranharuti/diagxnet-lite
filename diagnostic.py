"""
Quick diagnostic script to verify DiagXNet-Lite setup
Tests all components before running the full experiment
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import configs.config as config
        print("✅ Config module imported successfully")
        
        from src.data.dataset import CheXpertDataset, create_data_loaders
        print("✅ Dataset module imported successfully")
        
        from src.models.architectures import create_model
        print("✅ Models module imported successfully")
        
        from src.training.train import DenseNetTrainer
        print("✅ Training module imported successfully")
        
        from src.evaluation.metrics import ModelEvaluator
        print("✅ Evaluation module imported successfully")
        
        from src.evaluation.gradcam import DiagXNetGradCAMAnalyzer
        print("✅ Grad-CAM module imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device():
    """Test device configuration"""
    print("\n🔍 Testing device configuration...")
    
    try:
        import configs.config as config
        device = config.get_device()
        print(f"✅ Device: {device}")
        
        # Test tensor creation
        test_tensor = torch.randn(2, 3, 224, 224).to(device)
        print(f"✅ Test tensor created on {device}: {test_tensor.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_paths():
    """Test data paths and file existence"""
    print("\n🔍 Testing data paths...")
    
    try:
        import configs.config as config
        DATA_ROOT = config.DATA_ROOT
        CHEXPERT_ROOT = config.CHEXPERT_ROOT
        
        print(f"📁 Data root: {DATA_ROOT}")
        print(f"📁 CheXpert root: {CHEXPERT_ROOT}")
        
        # Check if paths exist
        if DATA_ROOT.exists():
            print("✅ Data root exists")
        else:
            print("❌ Data root missing")
            return False
            
        if CHEXPERT_ROOT.exists():
            print("✅ CheXpert root exists")
        else:
            print("❌ CheXpert root missing")
            return False
            
        # Check train CSV
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        if train_csv.exists():
            print("✅ Train CSV found")
            
            # Quick CSV check
            import pandas as pd
            df = pd.read_csv(train_csv)
            print(f"✅ Train CSV loaded: {len(df)} samples")
            
        else:
            print("❌ Train CSV missing")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Data path test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation"""
    print("\n🔍 Testing model creation...")
    
    try:
        import configs.config as config
        from src.models.architectures import create_model
        
        device = config.get_device()
        model = create_model('densenet121').to(device)
        model.eval()  # Set to eval mode to avoid batch norm issues
        print(f"✅ DenseNet-121 created successfully")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 224, 224).to(device)  # Grayscale input
        with torch.no_grad():
            output = model(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test dataset loading with small sample"""
    print("\n🔍 Testing dataset loading...")
    
    try:
        import configs.config as config
        from src.data.dataset import CheXpertDataset
        
        CHEXPERT_ROOT = config.CHEXPERT_ROOT
        DATA_ROOT = config.DATA_ROOT
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        
        # Create small dataset
        dataset = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            uncertainty_policy="ignore"
        )
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test loading one sample
        sample = dataset[0]
        image, labels, mask = sample
        print(f"✅ Sample loaded: image {image.shape}, labels {labels.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_availability():
    """Test required package availability"""
    print("\n🔍 Testing package availability...")
    
    packages = [
        'torch', 'torchvision', 'pandas', 'numpy', ('sklearn', 'scikit-learn'),
        'matplotlib', 'seaborn', 'cv2', 'tqdm', 'PIL'
    ]
    
    missing_packages = []
    
    for package in packages:
        try:
            if isinstance(package, tuple):
                import_name, display_name = package
                __import__(import_name)
                print(f"✅ {display_name} available")
            elif package == 'cv2':
                import cv2
                print(f"✅ {package} available")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ {package} available")
            else:
                __import__(package)
                print(f"✅ {package} available")
        except ImportError:
            display_name = package[1] if isinstance(package, tuple) else package
            print(f"❌ {display_name} missing")
            missing_packages.append(display_name)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        return False
    else:
        print("✅ All required packages available")
        return True

def run_full_diagnostic():
    """Run complete diagnostic"""
    print("🔧 DiagXNet-Lite Setup Diagnostic")
    print("="*50)
    
    tests = [
        ("Package Availability", test_package_availability),
        ("Imports", test_imports),
        ("Device Configuration", test_device),
        ("Data Paths", test_data_paths),
        ("Model Creation", test_model_creation),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("🔧 DIAGNOSTIC SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready to run full experiment.")
        print("Run: python run_experiment.py")
    else:
        print("❌ Some tests failed. Please fix issues before running experiment.")
    print("="*50)
    
    return all_passed

if __name__ == "__main__":
    run_full_diagnostic()
