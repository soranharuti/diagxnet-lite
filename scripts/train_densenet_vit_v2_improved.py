"""
DiagXNet-Lite V2: Improved Training with Class Imbalance Solutions
=====================================================================

Improvements over V1:
1. Focal Loss (gamma=2.0, alpha=0.25) - focuses on hard examples
2. Balanced Sampling - oversamples rare conditions (2x)
3. Exclude "Support Devices" - focus on pathologies only (13 labels)
4. Same architecture for fair comparison

Expected improvements:
- Better rare condition detection (Lung Lesion, Pneumonia)
- More balanced performance across all pathologies
- Higher mean AUROC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_v2 import CheXpertDatasetV2, get_train_transforms, get_val_transforms
from src.data.balanced_sampler import BalancedBatchSampler
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble
from src.training.focal_loss import FocalLoss
from configs.config import get_device, DATA_ROOT, CHEXPERT_ROOT


# Print banner
print("""
================================================================================
              DIAGXNET-LITE V2: IMPROVED TRAINING PIPELINE
================================================================================
Improvements:
  [OK] Focal Loss for class imbalance
  [OK] Balanced sampling for rare conditions
  [OK] Support Devices excluded (13 pathologies)
  [OK] Same architecture for fair comparison
================================================================================
""")


def train_base_model(model, train_loader, val_loader, device, model_name="model",
                     num_epochs=10, learning_rate=1e-4, save_dir="models/densenet_vit_stacking_v2/base_models",
                     use_cosine_schedule=False):
    """
    Train a base model with Focal Loss
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        model_name: Name for saving
        num_epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save checkpoints
        use_cosine_schedule: Whether to use cosine annealing LR schedule
    """
    print(f"\n{'='*80}")
    print(f"TRAINING BASE MODEL: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Setup directories
    save_path = Path(save_dir)
    checkpoint_dir = save_path / f"{model_name}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss function: Focal Loss V2
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    print(f"[OK] Using Focal Loss (alpha=0.25, gamma=2.0)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    if use_cosine_schedule:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        print(f"[OK] Using Cosine Annealing LR Scheduler")
    else:
        scheduler = None
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Training history
    history = []
    best_val_loss = float('inf')
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Loss: Focal Loss")
    print(f"  Mixed Precision: Enabled")
    print(f"  Device: {device}")
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels, masks)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels, masks)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step()
        
        # Time tracking
        epoch_time = (time.time() - epoch_start) / 60
        
        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time
        })
        
        # Print progress
        print(f"\n{'-'*80}")
        print(f"Epoch [{epoch}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time:.2f} min")
        print(f"{'-'*80}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / f"{model_name}_epoch_{epoch:02d}.pth")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_path / f"{model_name}_best.pth")
            print(f"  [OK] Best model saved (Val Loss: {best_val_loss:.6f})")
    
    # Save training history
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] {model_name.upper()} TRAINING COMPLETED")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Total Time: {sum(h['time_minutes'] for h in history):.2f} min")
    print(f"{'='*80}\n")
    
    return history, best_val_loss


def train_ensemble_meta_learner(model1, model2, train_loader, val_loader, device,
                                num_epochs=12, learning_rate=1e-4,
                                save_dir="models/densenet_vit_stacking_v2/ensemble"):
    """
    Train ensemble meta-learner with frozen base models
    """
    print(f"\n{'='*80}")
    print(f"TRAINING ENSEMBLE META-LEARNER")
    print(f"{'='*80}")
    
    save_path = Path(save_dir)
    checkpoint_dir = save_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ensemble
    ensemble = create_ensemble(
        model1=model1,
        model2=model2,
        ensemble_type="stacking",
        num_classes=13,  # V2: 13 classes instead of 14
        hidden_dim=256
    )
    ensemble = ensemble.to(device)
    
    # Freeze base models
    for param in ensemble.model1.parameters():
        param.requires_grad = False
    for param in ensemble.model2.parameters():
        param.requires_grad = False
    
    print("[OK] Base models frozen")
    print(f"[OK] Meta-learner trainable parameters: {sum(p.numel() for p in ensemble.meta_learner.parameters()):,}")
    
    # Loss function: Focal Loss
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    
    # Optimizer (only meta-learner parameters)
    optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Mixed precision
    scaler = GradScaler()
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        ensemble.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (images, labels, masks) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs, _ = ensemble(images)
                loss = criterion(outputs, labels, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ensemble.meta_learner.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"  Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        ensemble.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels, masks in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                
                with autocast():
                    outputs, _ = ensemble(images)
                    loss = criterion(outputs, labels, masks)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = (time.time() - epoch_start) / 60
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': current_lr,
            'time_minutes': epoch_time
        })
        
        print(f"\n{'-'*80}")
        print(f"Epoch [{epoch}/{num_epochs}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  Time:       {epoch_time:.2f} min")
        print(f"{'-'*80}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'ensemble_state_dict': ensemble.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_dir / f"ensemble_epoch_{epoch:02d}.pth")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, save_path / "ensemble_best.pth")
            print(f"  [OK] Best ensemble saved (Val Loss: {best_val_loss:.6f})")
        
        # Learning rate reduction at epoch 10
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 2
            print(f"  [OK] Learning rate reduced to {learning_rate/2:.6f}")
    
    with open(checkpoint_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"[OK] ENSEMBLE TRAINING COMPLETED")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Total Time: {sum(h['time_minutes'] for h in history):.2f} min")
    print(f"{'='*80}\n")
    
    return history, best_val_loss


def main(args):
    """Main training pipeline V2"""
    print(f"\nStarting DiagXNet-Lite V2 Training")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Load datasets with V2 (13 labels, no Support Devices)
    print("\n" + "="*80)
    print("LOADING DATASETS (V2)")
    print("="*80)
    
    train_csv = CHEXPERT_ROOT / "train.csv"
    val_csv = CHEXPERT_ROOT / "valid.csv"
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    train_dataset = CheXpertDatasetV2(
        train_csv,
        DATA_ROOT,
        train_transform,
        uncertainty_policy='ignore',
        frontal_only=False
    )
    
    val_dataset = CheXpertDatasetV2(
        val_csv,
        DATA_ROOT,
        val_transform,
        uncertainty_policy='ignore',
        frontal_only=False
    )
    
    print(f"\n[OK] Training samples: {len(train_dataset):,}")
    print(f"[OK] Validation samples: {len(val_dataset):,}")
    
    # Create balanced sampler for training
    print("\n" + "="*80)
    print("CREATING BALANCED SAMPLER")
    print("="*80)
    
    balanced_sampler = BalancedBatchSampler(
        train_dataset,
        rare_threshold=0.10,
        oversample_factor=2.0
    )
    
    # DataLoaders
    from configs.platform_config import OPTIMAL_NUM_WORKERS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=balanced_sampler,
        num_workers=OPTIMAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=OPTIMAL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"\n[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    
    # Stage 1: Train DenseNet-121
    if args.train_densenet:
        print("\n" + "="*80)
        print("STAGE 1: DENSENET-121 TRAINING")
        print("="*80)
        
        densenet = create_model(
            "densenet121",
            num_classes=13,  # V2: 13 classes
            pretrained=True,
            dropout_rate=0.2
        )
        densenet = densenet.to(device)
        
        train_base_model(
            densenet,
            train_loader,
            val_loader,
            device,
            model_name="densenet121",
            num_epochs=args.base_epochs,
            learning_rate=args.base_lr,
            save_dir="models/densenet_vit_stacking_v2/base_models",
            use_cosine_schedule=False
        )
    
    # Stage 2: Train Vision Transformer
    if args.train_vit:
        print("\n" + "="*80)
        print("STAGE 2: VISION TRANSFORMER TRAINING")
        print("="*80)
        
        vit = create_model(
            "vit_b_16",
            num_classes=13,  # V2: 13 classes
            pretrained=True,
            dropout_rate=0.2
        )
        vit = vit.to(device)
        
        train_base_model(
            vit,
            train_loader,
            val_loader,
            device,
            model_name="vit_b_16",
            num_epochs=args.base_epochs,
            learning_rate=args.base_lr,
            save_dir="models/densenet_vit_stacking_v2/base_models",
            use_cosine_schedule=True  # Use cosine for ViT
        )
    
    # Stage 3: Train Ensemble
    if args.train_ensemble:
        print("\n" + "="*80)
        print("STAGE 3: ENSEMBLE META-LEARNER TRAINING")
        print("="*80)
        
        # Load best base models
        densenet = create_model("densenet121", num_classes=13, pretrained=False, dropout_rate=0.2)
        vit = create_model("vit_b_16", num_classes=13, pretrained=False, dropout_rate=0.2)
        
        densenet_ckpt = torch.load(
            "models/densenet_vit_stacking_v2/base_models/densenet121_best.pth",
            map_location=device
        )
        vit_ckpt = torch.load(
            "models/densenet_vit_stacking_v2/base_models/vit_b_16_best.pth",
            map_location=device
        )
        
        densenet.load_state_dict(densenet_ckpt['model_state_dict'])
        vit.load_state_dict(vit_ckpt['model_state_dict'])
        
        densenet = densenet.to(device)
        vit = vit.to(device)
        
        print("[OK] Base models loaded")
        
        train_ensemble_meta_learner(
            densenet,
            vit,
            train_loader,
            val_loader,
            device,
            num_epochs=args.meta_epochs,
            learning_rate=args.meta_lr,
            save_dir="models/densenet_vit_stacking_v2/ensemble"
        )
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETED!")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved to: models/densenet_vit_stacking_v2/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiagXNet-Lite V2 Training Pipeline")
    
    # Training stages
    parser.add_argument('--train-densenet', action='store_true', default=True,
                       help='Train DenseNet-121')
    parser.add_argument('--train-vit', action='store_true', default=True,
                       help='Train Vision Transformer')
    parser.add_argument('--train-ensemble', action='store_true', default=True,
                       help='Train ensemble meta-learner')
    
    # Skip flags
    parser.add_argument('--skip-densenet', action='store_true',
                       help='Skip DenseNet-121 training')
    parser.add_argument('--skip-vit', action='store_true',
                       help='Skip Vision Transformer training')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble training')
    
    # Hyperparameters
    parser.add_argument('--base-epochs', type=int, default=10,
                       help='Epochs for base models')
    parser.add_argument('--meta-epochs', type=int, default=12,
                       help='Epochs for meta-learner')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--base-lr', type=float, default=1e-4,
                       help='Learning rate for base models')
    parser.add_argument('--meta-lr', type=float, default=1e-4,
                       help='Learning rate for meta-learner')
    
    args = parser.parse_args()
    
    # Handle skip flags
    if args.skip_densenet:
        args.train_densenet = False
    if args.skip_vit:
        args.train_vit = False
    if args.skip_ensemble:
        args.train_ensemble = False
    
    main(args)

