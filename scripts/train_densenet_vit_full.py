"""
Train DenseNet-121 + Vision Transformer Stacking Ensemble FROM SCRATCH
Trains both models fresh and saves all epoch checkpoints
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║      FULL TRAINING: DENSENET-121 + VISION TRANSFORMER ENSEMBLE           ║
╚═══════════════════════════════════════════════════════════════════════════╝

🔥 Training BOTH models from scratch!
   All epoch checkpoints will be saved for selection

Training Plan:
┌───────────────────────────────────────────────────────────────────────────┐
│ 🏋️  Step 1: Train DenseNet-121 from scratch          [2-3 hours]         │
│ 🏋️  Step 2: Train Vision Transformer (ViT-B/16)      [4-5 hours]         │
│ 🧠 Step 3: Train Meta-Learner                         [30 minutes]       │
├───────────────────────────────────────────────────────────────────────────┤
│ Total Time: ~7-9 hours                                                    │
│ Checkpoints: Saved for EVERY epoch                                       │
└───────────────────────────────────────────────────────────────────────────┘

Run with:
  python scripts/train_densenet_vit_full.py
  
Or with custom settings:
  python scripts/train_densenet_vit_full.py --epochs-densenet 10 --epochs-vit 10 --epochs-meta 5

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
# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble
from configs.config import *


def train_densenet121(train_loader, val_loader, class_weights, device, num_epochs=10, lr=1e-4):
    """Train DenseNet-121 from scratch"""
    print("\n" + "="*70)
    print("🏋️  TRAINING DENSENET-121 FROM SCRATCH")
    print("="*70)
    
    print("Creating DenseNet-121 model...")
    model = create_model("densenet121", num_classes=14, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Create checkpoint directory
    checkpoint_dir = Path('models/densenet_vit_stacking/base_models/densenet121_checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        model.train()
        running_loss = 0.0
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
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
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        print(f"{'─'*70}")
        
        # Save checkpoint for this epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        
        # Save every epoch
        torch.save(checkpoint, checkpoint_dir / f'densenet121_epoch_{epoch+1:02d}.pth')
        print(f"  ✓ Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'densenet121_best.pth')
            print(f"  🌟 New best model! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time / 60
        })
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✓ DenseNet-121 training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    
    return model


def train_vision_transformer(train_loader, val_loader, class_weights, device, num_epochs=10, lr=1e-4):
    """Train Vision Transformer from scratch"""
    print("\n" + "="*70)
    print("🏋️  TRAINING VISION TRANSFORMER (ViT-B/16) FROM SCRATCH")
    print("="*70)
    
    print("Creating Vision Transformer model...")
    model = create_model("vit_b_16", num_classes=14, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Use AdamW optimizer (better for transformers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Create checkpoint directory
    checkpoint_dir = Path('models/densenet_vit_stacking/base_models/vit_checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        model.train()
        running_loss = 0.0
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs * masks, labels * masks)
            loss.backward()
            
            # Gradient clipping (important for transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - LR: {current_lr:.6f}")
        
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
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        print(f"{'─'*70}")
        
        # Save checkpoint for this epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        
        # Save every epoch
        torch.save(checkpoint, checkpoint_dir / f'vit_b_16_epoch_{epoch+1:02d}.pth')
        print(f"  ✓ Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'vit_b_16_best.pth')
            print(f"  🌟 New best model! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time / 60
        })
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✓ Vision Transformer training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    
    return model


def train_meta_learner(ensemble, meta_loader, val_loader, class_weights, device, num_epochs=5, lr=1e-5):
    """Train meta-learner for ensemble"""
    print("\n" + "="*70)
    print("🧠 TRAINING META-LEARNER")
    print("="*70)
    
    # Only train meta-learner, freeze base models
    for param in ensemble.model1.parameters():
        param.requires_grad = False
    for param in ensemble.model2.parameters():
        param.requires_grad = False
    for param in ensemble.meta_learner.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
    # Create checkpoint directory
    checkpoint_dir = Path('models/densenet_vit_stacking/ensemble/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        ensemble.train()
        running_loss = 0.0
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
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
        
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        print(f"{'─'*70}")
        
        # Save checkpoint for this epoch
        checkpoint = {
            'epoch': epoch,
            'ensemble_state_dict': ensemble.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        # Save every epoch
        torch.save(checkpoint, checkpoint_dir / f'ensemble_epoch_{epoch+1:02d}.pth')
        print(f"  ✓ Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'ensemble_best.pth')
            print(f"  🌟 New best ensemble! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'time_minutes': epoch_time / 60
        })
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n✓ Meta-learner training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-densenet', type=int, default=10,
                       help='Number of epochs to train DenseNet-121')
    parser.add_argument('--epochs-vit', type=int, default=10,
                       help='Number of epochs to train ViT')
    parser.add_argument('--epochs-meta', type=int, default=5,
                       help='Number of epochs to train meta-learner')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr-densenet', type=float, default=1e-4,
                       help='Learning rate for DenseNet')
    parser.add_argument('--lr-vit', type=float, default=1e-4,
                       help='Learning rate for ViT')
    parser.add_argument('--lr-meta', type=float, default=1e-5,
                       help='Learning rate for meta-learner')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
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
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    meta_loader = DataLoader(meta_dataset, args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    class_weights = full_dataset.get_class_weights().to(device)
    
    print(f"✓ Train: {len(train_dataset):,}")
    print(f"✓ Val: {len(val_dataset):,}")
    print(f"✓ Meta: {len(meta_dataset):,}")
    
    # Train DenseNet-121
    densenet = train_densenet121(train_loader, val_loader, class_weights, device, 
                                 args.epochs_densenet, args.lr_densenet)
    
    # Train Vision Transformer
    vit = train_vision_transformer(train_loader, val_loader, class_weights, device, 
                                   args.epochs_vit, args.lr_vit)
    
    # Create ensemble
    print("\n" + "="*70)
    print("🔗 CREATING ENSEMBLE")
    print("="*70)
    ensemble = create_ensemble(densenet, vit, "stacking", num_classes=14, 
                              meta_learner_type="neural_network")
    ensemble = ensemble.to(device)
    print("✓ Ensemble created!")
    print("  Architecture: DenseNet-121 + Vision Transformer")
    print("  Meta-learner: Neural Network")
    
    # Train meta-learner
    train_meta_learner(ensemble, meta_loader, val_loader, class_weights, device, 
                      args.epochs_meta, args.lr_meta)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED!")
    print("="*70)
    print("\nSaved models and checkpoints:")
    print("  📁 models/densenet_vit_stacking/base_models/")
    print("     ├── densenet121_best.pth")
    print("     ├── densenet121_checkpoints/ (all epochs)")
    print("     ├── vit_b_16_best.pth")
    print("     └── vit_checkpoints/ (all epochs)")
    print("  📁 models/densenet_vit_stacking/ensemble/")
    print("     ├── ensemble_best.pth")
    print("     └── checkpoints/ (all epochs)")
    print("\nNext: Evaluate ensemble performance!")
    print("  python scripts/evaluate_densenet_vit_ensemble.py")


if __name__ == "__main__":
    main()
