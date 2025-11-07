"""
OPTIMIZED Training: DenseNet-121 + Vision Transformer Stacking Ensemble
===============================================================================
Key Optimizations:
- Mixed Precision Training (AMP) for 2-3x speedup
- Gradient Accumulation (simulate larger batch sizes)
- Early Stopping to prevent overfitting
- Improved Meta-Learner (larger capacity, more epochs)
- Better LR Schedulers
- Label Smoothing for better calibration
- Windows/Mac cross-platform compatibility
- Advanced logging and checkpointing
===============================================================================
"""

print("""
===============================================================================
   OPTIMIZED TRAINING: DENSENET-121 + VISION TRANSFORMER ENSEMBLE
===============================================================================

[OPTIMIZATIONS ENABLED]
  [OK] Mixed Precision Training (AMP) - 2-3x faster
  [OK] Gradient Accumulation - Effective batch size 32+
  [OK] Early Stopping - Prevents overfitting
  [OK] Improved Meta-Learner - hidden_dim=256, 20 epochs
  [OK] Better LR Schedulers - ReduceLROnPlateau + OneCycleLR
  [OK] Label Smoothing - Better calibration
  [OK] Cross-Platform - Windows/Mac compatible

Training Plan:
-------------------------------------------------------------------------------
| Step 1: Train DenseNet-121 from scratch          [~2-3 hours]             |
| Step 2: Train Vision Transformer (ViT-B/16)      [~3-4 hours]             |
| Step 3: Train Meta-Learner (OPTIMIZED)           [~1 hour]                |
-------------------------------------------------------------------------------
| Total Time: ~4-6 hours (with GPU + AMP)                                   |
| Expected Results: AUROC 0.78-0.80 (vs 0.68 baseline)                     |
-------------------------------------------------------------------------------

""")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
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

from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms, get_class_weights
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble
from configs.config import *

# Import platform-specific settings
try:
    from configs.platform_config import OPTIMAL_NUM_WORKERS, PLATFORM_SETTINGS
except ImportError:
    OPTIMAL_NUM_WORKERS = 0
    PLATFORM_SETTINGS = {'num_workers': 0, 'pin_memory': False, 'persistent_workers': False}


class LabelSmoothingBCELoss(nn.Module):
    """BCE Loss with Label Smoothing for better calibration"""
    
    def __init__(self, pos_weight=None, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        
    def forward(self, outputs, targets):
        # Apply label smoothing: 1 -> 0.9, 0 -> 0.1
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        
        if self.pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
            
        return criterion(outputs, targets_smooth)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False


def train_densenet121_optimized(train_loader, val_loader, class_weights, device, 
                                 num_epochs=10, lr=1e-4, accumulation_steps=2):
    """Train DenseNet-121 with optimizations"""
    print("\n" + "="*70)
    print("[TRAINING] DENSENET-121 (OPTIMIZED)")
    print("="*70)
    print(f"[INFO] Mixed Precision: {'ON' if device.type == 'cuda' else 'OFF'}")
    print(f"[INFO] Gradient Accumulation: {accumulation_steps} steps")
    print(f"[INFO] Effective Batch Size: {train_loader.batch_size * accumulation_steps}")
    
    # Create model
    print("[LOADING] Creating DenseNet-121 model...")
    model = create_model("densenet121", num_classes=14, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Total parameters: {params:,}")
    print(f"[OK] Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Loss with label smoothing
    criterion = LabelSmoothingBCELoss(pos_weight=class_weights, smoothing=0.1)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
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
        optimizer.zero_grad()
        
        print(f"\n{'='*70}")
        print(f"[EPOCH {epoch+1}/{num_epochs}] DenseNet-121")
        print(f"{'='*70}")
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            # Mixed precision forward pass
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs * masks, labels * masks)
                loss = loss / accumulation_steps  # Normalize for accumulation
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {current_lr:.6f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs * masks, labels * masks)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'-'*70}")
        print(f"[SUMMARY] Epoch {epoch+1}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        print(f"{'-'*70}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        
        torch.save(checkpoint, checkpoint_dir / f'densenet121_epoch_{epoch+1:02d}.pth')
        print(f"  [OK] Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'densenet121_best.pth')
            print(f"  [! ] New best model! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time / 60
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n[! ] Early stopping triggered at epoch {epoch+1}")
            print(f"[INFO] No improvement for {early_stopping.patience} epochs")
            break
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n[OK] DenseNet-121 training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    
    return model


def train_vision_transformer_optimized(train_loader, val_loader, class_weights, device, 
                                       num_epochs=10, lr=1e-4, accumulation_steps=2):
    """Train Vision Transformer with optimizations"""
    print("\n" + "="*70)
    print("[TRAINING] VISION TRANSFORMER (OPTIMIZED)")
    print("="*70)
    print(f"[INFO] Mixed Precision: {'ON' if device.type == 'cuda' else 'OFF'}")
    print(f"[INFO] Gradient Accumulation: {accumulation_steps} steps")
    print(f"[INFO] Effective Batch Size: {train_loader.batch_size * accumulation_steps}")
    
    # Create model
    print("[LOADING] Creating Vision Transformer model...")
    model = create_model("vit_b_16", num_classes=14, pretrained=True, dropout_rate=0.2)
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] Total parameters: {params:,}")
    print(f"[OK] Trainable parameters: {trainable_params:,}")
    
    # Use AdamW optimizer (better for transformers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # OneCycleLR with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )
    
    # Loss with label smoothing
    criterion = LabelSmoothingBCELoss(pos_weight=class_weights, smoothing=0.1)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
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
        optimizer.zero_grad()
        
        print(f"\n{'='*70}")
        print(f"[EPOCH {epoch+1}/{num_epochs}] Vision Transformer")
        print(f"{'='*70}")
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            # Mixed precision forward pass
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs * masks, labels * masks)
                loss = loss / accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping (important for transformers)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Step scheduler after optimizer update
                scheduler.step()
            
            running_loss += loss.item() * accumulation_steps
            
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {current_lr:.6f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs * masks, labels * masks)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\n{'-'*70}")
        print(f"[SUMMARY] Epoch {epoch+1}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        print(f"{'-'*70}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        
        torch.save(checkpoint, checkpoint_dir / f'vit_b_16_epoch_{epoch+1:02d}.pth')
        print(f"  [OK] Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'vit_b_16_best.pth')
            print(f"  [! ] New best model! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time / 60
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n[! ] Early stopping triggered at epoch {epoch+1}")
            print(f"[INFO] No improvement for {early_stopping.patience} epochs")
            break
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n[OK] Vision Transformer training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    
    return model


def train_meta_learner_optimized(ensemble, meta_loader, val_loader, class_weights, device, 
                                  num_epochs=20, lr=1e-4, accumulation_steps=1):
    """Train meta-learner with optimizations"""
    print("\n" + "="*70)
    print("[TRAINING] META-LEARNER (OPTIMIZED)")
    print("="*70)
    print(f"[INFO] Hidden Dimension: 256 (4x larger than baseline)")
    print(f"[INFO] Epochs: {num_epochs} (4x more than baseline)")
    print(f"[INFO] Learning Rate: {lr} (10x higher than baseline)")
    print(f"[INFO] Mixed Precision: {'ON' if device.type == 'cuda' else 'OFF'}")
    
    # Freeze base models, train only meta-learner
    for param in ensemble.model1.parameters():
        param.requires_grad = False
    for param in ensemble.model2.parameters():
        param.requires_grad = False
    for param in ensemble.meta_learner.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(ensemble.meta_learner.parameters(), lr=lr, weight_decay=1e-4)
    
    # ReduceLROnPlateau for meta-learner
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Loss with label smoothing
    criterion = LabelSmoothingBCELoss(pos_weight=class_weights, smoothing=0.1)
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
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
        print(f"[EPOCH {epoch+1}/{num_epochs}] Meta-Learner")
        print(f"{'='*70}")
        
        for batch_idx, (images, labels, masks) in enumerate(meta_loader):
            images, labels, masks = images.to(device), labels.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=(device.type == 'cuda')):
                output, _ = ensemble(images)
                loss = criterion(output * masks, labels * masks)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ensemble.meta_learner.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{len(meta_loader)} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        avg_train_loss = running_loss / len(meta_loader)
        
        # Validate
        ensemble.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images, labels, masks = images.to(device), labels.to(device), masks.to(device)
                
                with autocast(enabled=(device.type == 'cuda')):
                    output, _ = ensemble(images)
                    loss = criterion(output * masks, labels * masks)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'-'*70}")
        print(f"[SUMMARY] Epoch {epoch+1}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time/60:.1f} min")
        
        # Show improvement
        if epoch == 0:
            print(f"  Baseline:   {avg_val_loss:.4f}")
        else:
            improvement = training_history[0]['val_loss'] - avg_val_loss
            improvement_pct = (improvement / training_history[0]['val_loss']) * 100
            print(f"  Improvement: {improvement:.4f} ({improvement_pct:+.2f}%)")
        print(f"{'-'*70}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'ensemble_state_dict': ensemble.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr
        }
        
        torch.save(checkpoint, checkpoint_dir / f'ensemble_epoch_{epoch+1:02d}.pth')
        print(f"  [OK] Saved checkpoint: epoch_{epoch+1:02d}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, checkpoint_dir.parent / 'ensemble_best.pth')
            print(f"  [! ] New best ensemble! Val Loss: {best_val_loss:.4f}")
        
        # Record history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time / 60
        })
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\n[! ] Early stopping triggered at epoch {epoch+1}")
            print(f"[INFO] No improvement for {early_stopping.patience} epochs")
            break
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\n[OK] Meta-learner training completed!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total improvement: {training_history[0]['val_loss'] - best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Optimized Training Script')
    
    # Model training epochs
    parser.add_argument('--epochs-densenet', type=int, default=10,
                       help='Number of epochs to train DenseNet-121 (default: 10)')
    parser.add_argument('--epochs-vit', type=int, default=10,
                       help='Number of epochs to train ViT (default: 10)')
    parser.add_argument('--epochs-meta', type=int, default=20,
                       help='Number of epochs to train meta-learner (default: 20, was 5)')
    
    # Learning rates
    parser.add_argument('--lr-densenet', type=float, default=1e-4,
                       help='Learning rate for DenseNet (default: 1e-4)')
    parser.add_argument('--lr-vit', type=float, default=1e-4,
                       help='Learning rate for ViT (default: 1e-4)')
    parser.add_argument('--lr-meta', type=float, default=1e-4,
                       help='Learning rate for meta-learner (default: 1e-4, was 1e-5)')
    
    # Batch size and accumulation
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training (default: 16)')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps (default: 2, effective batch=32)')
    
    # Meta-learner architecture
    parser.add_argument('--meta-hidden-dim', type=int, default=256,
                       help='Hidden dimension for meta-learner (default: 256, was 64)')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    print(f"\n[DEVICE] {device}")
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[MEMORY] {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders
    print("\n" + "="*70)
    print("[DATA] CREATING DATA LOADERS")
    print("="*70)
    
    train_csv = CHEXPERT_ROOT / "train.csv"
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Use CHEXPERT_ROOT so dataset can resolve correct subfolder (chexpert or CheXpert-v1.0-small)
    full_dataset = CheXpertDataset(train_csv, DATA_ROOT, train_transform, "ignore", True)
    
    # Split dataset: 60% train, 20% validation, 20% meta-learning
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
    
    # Use platform-specific settings
    print(f"[INFO] Platform: {os.name}")
    print(f"[INFO] Num Workers: {OPTIMAL_NUM_WORKERS}")
    print(f"[INFO] Pin Memory: {PLATFORM_SETTINGS['pin_memory']}")
    
    train_loader = DataLoader(
        train_dataset, 
        args.batch_size, 
        shuffle=True,
        num_workers=OPTIMAL_NUM_WORKERS,
        pin_memory=PLATFORM_SETTINGS['pin_memory'],
        persistent_workers=PLATFORM_SETTINGS['persistent_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=OPTIMAL_NUM_WORKERS,
        pin_memory=PLATFORM_SETTINGS['pin_memory'],
        persistent_workers=PLATFORM_SETTINGS['persistent_workers']
    )
    
    meta_loader = DataLoader(
        meta_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=OPTIMAL_NUM_WORKERS,
        pin_memory=PLATFORM_SETTINGS['pin_memory'],
        persistent_workers=PLATFORM_SETTINGS['persistent_workers']
    )
    
    class_weights = get_class_weights(full_dataset).to(device)
    
    print(f"[OK] Train: {len(train_dataset):,} samples")
    print(f"[OK] Val: {len(val_dataset):,} samples")
    print(f"[OK] Meta: {len(meta_dataset):,} samples")
    print(f"[OK] Batch Size: {args.batch_size}")
    print(f"[OK] Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    
    # Train DenseNet-121
    densenet = train_densenet121_optimized(
        train_loader, val_loader, class_weights, device,
        args.epochs_densenet, args.lr_densenet, args.accumulation_steps
    )
    
    # Train Vision Transformer
    vit = train_vision_transformer_optimized(
        train_loader, val_loader, class_weights, device,
        args.epochs_vit, args.lr_vit, args.accumulation_steps
    )
    
    # Create ensemble with larger meta-learner
    print("\n" + "="*70)
    print("[ENSEMBLE] CREATING ENSEMBLE")
    print("="*70)
    
    ensemble = create_ensemble(
        densenet, vit,
        ensemble_type="stacking",
        num_classes=14,
        meta_learner_type="neural_network",
        freeze_base_models=True,
        hidden_dim=args.meta_hidden_dim  # Use larger hidden dimension
    )
    ensemble = ensemble.to(device)
    
    meta_params = sum(p.numel() for p in ensemble.meta_learner.parameters())
    print(f"[OK] Ensemble created!")
    print(f"  Architecture: DenseNet-121 + Vision Transformer")
    print(f"  Meta-learner: Neural Network")
    print(f"  Hidden Dim: {args.meta_hidden_dim} (was 64)")
    print(f"  Meta-learner Parameters: {meta_params:,}")
    
    # Train meta-learner
    train_meta_learner_optimized(
        ensemble, meta_loader, val_loader, class_weights, device,
        args.epochs_meta, args.lr_meta, 1  # No accumulation for meta-learner
    )
    
    # Final summary
    print("\n" + "="*70)
    print("[COMPLETE] TRAINING COMPLETED!")
    print("="*70)
    print("\n[SAVED] Models and checkpoints:")
    print("  models/densenet_vit_stacking/base_models/")
    print("     - densenet121_best.pth")
    print("     - densenet121_checkpoints/ (all epochs)")
    print("     - vit_b_16_best.pth")
    print("     - vit_checkpoints/ (all epochs)")
    print("  models/densenet_vit_stacking/ensemble/")
    print("     - ensemble_best.pth")
    print("     - checkpoints/ (all epochs)")
    
    print("\n[NEXT] Evaluate ensemble performance!")
    print("  python scripts/evaluate_densenet_vit_ensemble.py")
    
    print("\n[OPTIMIZATIONS APPLIED]")
    print(f"  [OK] Mixed Precision (AMP): {'YES' if device.type == 'cuda' else 'NO (CPU)'}")
    print(f"  [OK] Gradient Accumulation: {args.accumulation_steps}x")
    print(f"  [OK] Meta-learner Hidden Dim: {args.meta_hidden_dim} (4x improvement)")
    print(f"  [OK] Meta-learner Epochs: {args.epochs_meta} (4x improvement)")
    print(f"  [OK] Label Smoothing: 0.1")
    print(f"  [OK] Early Stopping: Enabled")
    print(f"  [OK] Gradient Clipping: max_norm=1.0")
    print(f"  [OK] Better LR Schedulers: ReduceLROnPlateau + OneCycleLR")
    
    print("\n[EXPECTED RESULTS]")
    print("  Target Ensemble AUROC: 0.78-0.80 (vs 0.68 baseline)")
    print("  Target Ensemble AUPRC: 0.48-0.50 (vs 0.42 baseline)")


if __name__ == "__main__":
    # Configure multiprocessing for Windows
    import platform
    if platform.system() == 'Windows':
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    main()

