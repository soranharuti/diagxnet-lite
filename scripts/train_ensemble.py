"""
Smart Ensemble Training: Use existing DenseNet-121 + Train Inception-ResNet-V2
Saves ~2 hours by reusing your already-trained model!
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    SMART ENSEMBLE TRAINING                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

💡 Your existing DenseNet-121 model will be reused!
   No need to retrain it (saves ~2 hours)

Training Plan:
┌───────────────────────────────────────────────────────────────────────────┐
│ ✅ Step 1: Load DenseNet-121 (already trained)       [0 minutes]         │
│ 🏋️  Step 2: Train Inception-ResNet-V2                 [3-4 hours]         │
│ 🧠 Step 3: Train Meta-Learner                         [30 minutes]       │
├───────────────────────────────────────────────────────────────────────────┤
│ Total Time: ~4 hours (vs 6 hours if training both)                       │
│ Time Saved: ~2 hours! ⏱️                                                   │
└───────────────────────────────────────────────────────────────────────────┘

Run with:
  python train_smart_ensemble.py
  
Or with custom settings:
  python train_smart_ensemble.py --epochs-model2 10 --epochs-meta 5

""")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append('src')
from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble
from configs.config import *


def load_existing_densenet(model_path, device):
    """Load existing DenseNet-121"""
    print("\n" + "="*70)
    print("📥 LOADING EXISTING DENSENET-121")
    print("="*70)
    print(f"Path: {model_path}")
    
    model = create_model("densenet121", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"✓ Model trained for {checkpoint['epoch'] + 1} epochs")
        if 'val_loss' in checkpoint:
            print(f"✓ Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    print("✓ DenseNet-121 loaded and frozen")
    print("✓ Ready for ensemble!")
    return model


def train_inception_resnet_v2(train_loader, val_loader, class_weights, device, num_epochs=5, lr=1e-4):
    """Train Inception-ResNet-V2"""
    print("\n" + "="*70)
    print("🏋️  TRAINING INCEPTION-RESNET-V2")
    print("="*70)
    
    print("Creating model...")
    model = create_model("inception_resnet_v2", num_classes=14, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        model.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs * masks, labels * masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs * masks, labels * masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Time: {epoch_time/60:.1f} min")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss
            }, 'models/densenet121_inception_stacking/base_models/inception_resnet_v2_best.pth')
            print(f"  ✓ Saved best model!")
    
    print(f"\n✓ Inception-ResNet-V2 training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    return model


def train_meta_learner(ensemble, meta_loader, val_loader, class_weights, device, num_epochs=3, lr=1e-5):
    """Train meta-learner"""
    print("\n" + "="*70)
    print("🧠 TRAINING META-LEARNER")
    print("="*70)
    
    optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        ensemble.train()
        running_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        for batch_idx, (images, labels, masks) in enumerate(meta_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            output, _ = ensemble(images)
            loss = criterion(output * masks, labels * masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(meta_loader)} - Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(meta_loader)
        
        # Validate
        ensemble.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                output, _ = ensemble(images)
                loss = criterion(output * masks, labels * masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Time: {epoch_time/60:.1f} min")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'ensemble_state_dict': ensemble.state_dict(),
                'val_loss': avg_val_loss
            }, 'models/densenet121_inception_stacking/ensemble/ensemble_best.pth')
            print(f"  ✓ Saved best ensemble!")
    
    print(f"\n✓ Meta-learner training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1-path', default='models/densenet121_inception_stacking/base_models/densenet121_best.pth')
    parser.add_argument('--epochs-model2', type=int, default=5)
    parser.add_argument('--epochs-meta', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    # Check model exists
    if not Path(args.model1_path).exists():
        print(f"❌ Model not found: {args.model1_path}")
        return
    
    # Create data loaders
    print("\n" + "="*70)
    print("📊 CREATING DATA LOADERS")
    print("="*70)
    
    train_csv = CHEXPERT_ROOT / "train" / "train.csv"
    train_transform = get_train_transforms(augment=True)
    val_transform = get_val_transforms()
    
    full_dataset = CheXpertDataset(train_csv, DATA_ROOT, train_transform, "ignore", True)
    
    total_size = len(full_dataset)
    train_size = int(total_size * 0.6)
    val_size = int(total_size * 0.2)
    
    np.random.seed(42)
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    meta_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset_full = CheXpertDataset(train_csv, DATA_ROOT, val_transform, "ignore", True)
    val_dataset = Subset(val_dataset_full, val_indices)
    meta_dataset = Subset(val_dataset_full, meta_indices)
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    meta_loader = DataLoader(meta_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    class_weights = full_dataset.get_class_weights().to(device)
    
    print(f"✓ Train: {len(train_dataset):,}")
    print(f"✓ Val: {len(val_dataset):,}")
    print(f"✓ Meta: {len(meta_dataset):,}")
    
    # Load DenseNet
    model1 = load_existing_densenet(args.model1_path, device)
    
    # Train Inception-ResNet-V2
    model2 = train_inception_resnet_v2(train_loader, val_loader, class_weights, device, args.epochs_model2, args.lr)
    
    # Create ensemble
    print("\n" + "="*70)
    print("🔗 CREATING ENSEMBLE")
    print("="*70)
    ensemble = create_ensemble(model1, model2, "stacking", num_classes=14, meta_learner_type="neural_network")
    ensemble = ensemble.to(device)
    print("✓ Ensemble created!")
    
    # Train meta-learner
    train_meta_learner(ensemble, meta_loader, val_loader, class_weights, device, args.epochs_meta, args.lr * 0.1)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print("\nSaved models:")
    print("  ✓ models/densenet121_inception_stacking/base_models/inception_resnet_v2_best.pth")
    print("  ✓ models/densenet121_inception_stacking/ensemble/ensemble_best.pth")
    print("\nNext: Evaluate ensemble performance!")


if __name__ == "__main__":
    main()
