"""
Test script to verify ensemble implementation
Run this before training to ensure everything is set up correctly
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_architectures():
    """Test that all architectures can be created"""
    print("\n" + "="*60)
    print("Testing Model Architectures")
    print("="*60)
    
    from src.models.architectures import create_model
    
    models_to_test = [
        ("densenet121", {}),
        ("efficientnet_b3", {}),
        ("inception_resnet_v2", {})
    ]
    
    for arch, kwargs in models_to_test:
        try:
            print(f"\n✓ Testing {arch}...")
            model = create_model(arch, num_classes=14, pretrained=False, **kwargs)
            
            # Test forward pass
            x = torch.randn(2, 1, 224, 224)
            output = model(x)
            
            assert output.shape == (2, 14), f"Expected shape (2, 14), got {output.shape}"
            
            params = sum(p.numel() for p in model.parameters())
            print(f"  ✓ Created successfully")
            print(f"  ✓ Parameters: {params:,}")
            print(f"  ✓ Forward pass OK: {output.shape}")
            
        except ImportError as e:
            if "timm" in str(e) and arch == "inception_resnet_v2":
                print(f"  ⚠️  {arch} requires 'timm' library")
                print(f"     Install with: pip install timm")
            else:
                print(f"  ✗ Failed: {e}")
                return False
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            return False
    
    print("\n✅ All architectures working!")
    return True


def test_ensemble():
    """Test ensemble creation"""
    print("\n" + "="*60)
    print("Testing Ensemble Models")
    print("="*60)
    
    try:
        from src.models.architectures import create_model
        from src.models.ensemble import create_ensemble
        
        print("\n✓ Creating base models...")
        model1 = create_model("densenet121", num_classes=14, pretrained=False)
        model2 = create_model("efficientnet_b3", num_classes=14, pretrained=False)
        
        print("✓ Creating stacking ensemble...")
        ensemble = create_ensemble(
            model1, model2,
            ensemble_type="stacking",
            meta_learner_type="neural_network",
            num_classes=14
        )
        
        print("✓ Testing forward pass...")
        x = torch.randn(2, 1, 224, 224)
        output, base_outputs = ensemble(x)
        
        assert output.shape == (2, 14), f"Expected shape (2, 14), got {output.shape}"
        assert 'model1' in base_outputs, "Missing model1 output"
        assert 'model2' in base_outputs, "Missing model2 output"
        
        print(f"  ✓ Ensemble output: {output.shape}")
        print(f"  ✓ Model 1 output: {base_outputs['model1'].shape}")
        print(f"  ✓ Model 2 output: {base_outputs['model2'].shape}")
        
        print("\n✅ Ensemble models working!")
        return True
        
    except Exception as e:
        print(f"\n✗ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_meta_learner():
    """Test meta-learner"""
    print("\n" + "="*60)
    print("Testing Meta-Learner")
    print("="*60)
    
    try:
        from src.models.ensemble import MetaLearner
        
        print("\n✓ Testing neural network meta-learner...")
        meta_nn = MetaLearner(
            num_base_models=2,
            num_classes=14,
            meta_learner_type="neural_network"
        )
        
        # Simulate base model predictions
        pred1 = torch.randn(4, 14)
        pred2 = torch.randn(4, 14)
        
        output = meta_nn([pred1, pred2])
        assert output.shape == (4, 14), f"Expected (4, 14), got {output.shape}"
        print(f"  ✓ Neural network meta-learner: {output.shape}")
        
        print("\n✓ Testing logistic meta-learner...")
        meta_logistic = MetaLearner(
            num_base_models=2,
            num_classes=14,
            meta_learner_type="logistic"
        )
        
        output = meta_logistic([pred1, pred2])
        assert output.shape == (4, 14), f"Expected (4, 14), got {output.shape}"
        print(f"  ✓ Logistic meta-learner: {output.shape}")
        
        print("\n✅ Meta-learners working!")
        return True
        
    except Exception as e:
        print(f"\n✗ Meta-learner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_imports():
    """Test all necessary imports"""
    print("\n" + "="*60)
    print("Testing Dependencies")
    print("="*60)
    
    imports_to_test = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("tensorboard", "TensorBoard"),
    ]
    
    all_ok = True
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    # Optional imports
    try:
        import timm
        print(f"  ✓ timm (for Inception-ResNet-V2)")
    except ImportError:
        print(f"  ⚠️  timm - NOT INSTALLED (optional for Inception-ResNet-V2)")
        print(f"     Install with: pip install timm")
    
    if all_ok:
        print("\n✅ All required dependencies installed!")
    else:
        print("\n⚠️  Some dependencies missing. Install with: pip install -r requirements.txt")
    
    return all_ok


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "ENSEMBLE IMPLEMENTATION TEST")
    print("="*70)
    
    tests = [
        ("Dependencies", test_imports),
        ("Architectures", test_architectures),
        ("Meta-Learner", test_meta_learner),
        ("Ensemble", test_ensemble),
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! You're ready to train!")
        print("="*70)
        print("\nNext steps:")
        print("1. Run: python compare_ensemble_options.py")
        print("2. Choose your ensemble configuration")
        print("3. Start training with train_stacking_ensemble.py")
    else:
        print("\n" + "="*70)
        print("⚠️  SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the issues above before training.")
        print("Check ENSEMBLE_GUIDE.md for setup instructions.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
