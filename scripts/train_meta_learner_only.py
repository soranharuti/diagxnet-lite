"""
Train ONLY the Meta-Learner for DenseNet-121 + Vision Transformer Ensemble
Uses pre-trained base models
"""

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    TRAIN META-LEARNER ONLY                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

Using pre-trained models:
  ✓ DenseNet-121 (already trained)
  ✓ Vision Transformer (already trained)

Training:
  🧠 Meta-Learner only (~30-40 minutes for 5 epochs)

""")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import time
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
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs to train meta-learner')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate for meta-learner')
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    # Paths to trained models
    densenet_path = 'models/densenet_vit_stacking/base_models/densenet121_best.pth'
    vit_path = 'models/densenet_vit_stacking/base_models/vit_b_16_best.pth'
    
    # Check models exist
    if not Path(densenet_path).exists():
        print(f"❌ DenseNet model not found: {densenet_path}")
        return
    if not Path(vit_path).exists():
        print(f"❌ ViT model not found: {vit_path}")
        return
    
    print("\n" + "="*70)
    print("📥 LOADING PRE-TRAINED MODELS")
    print("="*70)
    
    # Load DenseNet-121
    print("\nLoading DenseNet-121...")
    densenet = create_model("densenet121", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(densenet_path, map_location=device)
    densenet.load_state_dict(checkpoint['model_state_dict'])
    densenet = densenet.to(device)
    densenet.eval()
    print("  ✓ DenseNet-121 loaded")
    
    # Load Vision Transformer
    print("\nLoading Vision Transformer...")
    vit = create_model("vit_b_16", num_classes=14, pretrained=False, dropout_rate=0.2)
    checkpoint = torch.load(vit_path, map_location=device)
    vit.load_state_dict(checkpoint['model_state_dict'])
    vit = vit.to(device)
    vit.eval()
    print("  ✓ Vision Transformer loaded")
    
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
    val_indices = indices[train_size:train_size + val_size]
    meta_indices = indices[train_size + val_size:]
    
    val_dataset_full = CheXpertDataset(train_csv, DATA_ROOT, val_transform, "ignore", True)
    val_dataset = Subset(val_dataset_full, val_indices)
    meta_dataset = Subset(val_dataset_full, meta_indices)
    
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    meta_loader = DataLoader(meta_dataset, args.batch_size, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    class_weights = full_dataset.get_class_weights().to(device)
    
    print(f"✓ Val: {len(val_dataset):,}")
    print(f"✓ Meta: {len(meta_dataset):,}")
    
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
                      args.epochs, args.lr)
    
    print("\n" + "="*70)
    print("🎉 META-LEARNER TRAINING COMPLETED!")
    print("="*70)
    print("\nSaved models and checkpoints:")
    print("  📁 models/densenet_vit_stacking/ensemble/")
    print("     ├── ensemble_best.pth")
    print("     └── checkpoints/ (all epochs)")
    print("\nNext: Evaluate ensemble performance!")
    print("  python scripts/evaluate_densenet_vit_ensemble.py")


if __name__ == "__main__":
    main()
