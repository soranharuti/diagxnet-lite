"""
Pair-wise Stacking Training Script for DiagXNet-Lite
Trains two base models independently, then trains a meta-learner to combine predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Import modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from src.models.ensemble import create_ensemble, StackingEnsemble
from configs.config import *


class PairwiseStackingTrainer:
    """
    Trainer for pair-wise stacking ensemble
    
    Training Pipeline:
    1. Train model 1 independently
    2. Train model 2 independently  
    3. Extract predictions from both models on validation set
    4. Train meta-learner to combine predictions
    5. (Optional) Fine-tune entire ensemble
    """
    
    def __init__(
        self,
        model1_arch: str = "densenet121",
        model2_arch: str = "efficientnet_b3",  # or "inception_resnet_v2"
        meta_learner_type: str = "neural_network",
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs_base: int = 5,
        num_epochs_meta: int = 3,
        freeze_base_models: bool = True
    ):
        """
        Args:
            model1_arch: Architecture for first base model
            model2_arch: Architecture for second base model  
            meta_learner_type: Type of meta-learner ('neural_network' or 'logistic')
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs_base: Number of epochs to train each base model
            num_epochs_meta: Number of epochs to train meta-learner
            freeze_base_models: Whether to freeze base models during meta-training
        """
        self.model1_arch = model1_arch
        self.model2_arch = model2_arch
        self.meta_learner_type = meta_learner_type
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs_base = num_epochs_base
        self.num_epochs_meta = num_epochs_meta
        self.freeze_base_models = freeze_base_models
        self.num_classes = 14
        
        # Device
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Paths
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"stacking_{model1_arch}_{model2_arch}_{self.timestamp}"
        
        # Logging
        self.writer = SummaryWriter(f"{self.results_dir}/tensorboard/{self.run_name}")
        
        # Initialize models (will be created in setup)
        self.model1 = None
        self.model2 = None
        self.ensemble = None
    
    def create_data_loaders(self):
        """Create data loaders"""
        print("Creating data loaders...")
        
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        
        train_transform = get_train_transforms(augment=True)
        val_transform = get_val_transforms()
        
        # Full dataset
        full_dataset = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=train_transform,
            uncertainty_policy="ignore",
            frontal_only=True
        )
        
        # Split: 60% train, 20% validation (for base models), 20% meta-train
        total_size = len(full_dataset)
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)
        meta_size = total_size - train_size - val_size
        
        # Create indices
        np.random.seed(42)
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        meta_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = Subset(full_dataset, train_indices)
        
        val_dataset_full = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=val_transform,
            uncertainty_policy="ignore",
            frontal_only=True
        )
        val_dataset = Subset(val_dataset_full, val_indices)
        meta_dataset = Subset(val_dataset_full, meta_indices)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        self.meta_loader = DataLoader(
            meta_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        print(f"Meta-training samples: {len(meta_dataset):,}")
        
        # Class weights
        self.class_weights = full_dataset.get_class_weights().to(self.device)
    
    def train_base_model(
        self,
        model: nn.Module,
        model_name: str,
        num_epochs: int
    ) -> nn.Module:
        """Train a single base model"""
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        model = model.to(self.device)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Train
            model.train()
            running_loss = 0.0
            
            for images, labels, masks in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                # Masked loss
                masked_outputs = outputs * masks
                masked_labels = labels * masks
                loss = criterion(masked_outputs, masked_labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            
            # Validate
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels, masks in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = model(images)
                    masked_outputs = outputs * masks
                    masked_labels = labels * masks
                    loss = criterion(masked_outputs, masked_labels)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            
            # Log to tensorboard
            self.writer.add_scalar(f'{model_name}/Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar(f'{model_name}/Loss/Validation', avg_val_loss, epoch)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }, self.models_dir / f"{self.run_name}_{model_name}_best.pth")
        
        print(f"✓ {model_name} training completed. Best val loss: {best_val_loss:.4f}")
        
        return model
    
    def extract_base_predictions(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extract predictions from both base models on meta-training set
        
        Returns:
            Dictionary with 'model1' and 'model2' keys, each containing
            (predictions, labels, masks) tuples
        """
        print("\nExtracting base model predictions for meta-training...")
        
        self.model1.eval()
        self.model2.eval()
        
        all_preds1 = []
        all_preds2 = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for images, labels, masks in self.meta_loader:
                images = images.to(self.device)
                
                # Get predictions from both models
                preds1 = self.model1(images)
                preds2 = self.model2(images)
                
                all_preds1.append(preds1.cpu())
                all_preds2.append(preds2.cpu())
                all_labels.append(labels)
                all_masks.append(masks)
        
        # Concatenate
        all_preds1 = torch.cat(all_preds1, dim=0)
        all_preds2 = torch.cat(all_preds2, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        print(f"✓ Extracted predictions: {all_preds1.shape}")
        
        return {
            'model1': (all_preds1, all_labels, all_masks),
            'model2': (all_preds2, all_labels, all_masks)
        }
    
    def train_meta_learner(self, num_epochs: int):
        """Train the meta-learner on base model predictions"""
        print(f"\n{'='*60}")
        print(f"Training Meta-Learner ({self.meta_learner_type})")
        print(f"{'='*60}")
        
        # Only meta-learner parameters are trainable
        optimizer = optim.Adam(
            self.ensemble.meta_learner.parameters(),
            lr=self.learning_rate * 0.1,  # Lower LR for meta-learner
            weight_decay=1e-4
        )
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Train
            self.ensemble.train()
            running_loss = 0.0
            
            for images, labels, masks in self.meta_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward through ensemble
                ensemble_output, _ = self.ensemble(images)
                
                # Masked loss
                masked_output = ensemble_output * masks
                masked_labels = labels * masks
                loss = criterion(masked_output, masked_labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(self.meta_loader)
            
            # Validate on validation set
            self.ensemble.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels, masks in self.val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    masks = masks.to(self.device)
                    
                    ensemble_output, _ = self.ensemble(images)
                    masked_output = ensemble_output * masks
                    masked_labels = labels * masks
                    loss = criterion(masked_output, masked_labels)
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(self.val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            
            # Log
            self.writer.add_scalar('MetaLearner/Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('MetaLearner/Loss/Validation', avg_val_loss, epoch)
            
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'ensemble_state_dict': self.ensemble.state_dict(),
                    'val_loss': avg_val_loss
                }, self.models_dir / f"{self.run_name}_ensemble_best.pth")
        
        print(f"✓ Meta-learner training completed. Best val loss: {best_val_loss:.4f}")
    
    def train_full_pipeline(self):
        """Complete training pipeline for pair-wise stacking"""
        print(f"\n{'='*70}")
        print(f"PAIR-WISE STACKING ENSEMBLE TRAINING")
        print(f"Model 1: {self.model1_arch}")
        print(f"Model 2: {self.model2_arch}")
        print(f"Meta-learner: {self.meta_learner_type}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Step 1: Create data loaders
        self.create_data_loaders()
        
        # Step 2: Create and train model 1
        print(f"\n{'='*60}")
        print(f"STEP 1: Training Base Model 1 ({self.model1_arch})")
        print(f"{'='*60}")
        self.model1 = create_model(
            self.model1_arch,
            num_classes=self.num_classes,
            pretrained=True,
            dropout_rate=0.2
        )
        self.model1 = self.train_base_model(self.model1, "model1", self.num_epochs_base)
        
        # Step 3: Create and train model 2
        print(f"\n{'='*60}")
        print(f"STEP 2: Training Base Model 2 ({self.model2_arch})")
        print(f"{'='*60}")
        self.model2 = create_model(
            self.model2_arch,
            num_classes=self.num_classes,
            pretrained=True,
            dropout_rate=0.2
        )
        self.model2 = self.train_base_model(self.model2, "model2", self.num_epochs_base)
        
        # Step 4: Create ensemble
        print(f"\n{'='*60}")
        print(f"STEP 3: Creating Ensemble")
        print(f"{'='*60}")
        self.ensemble = create_ensemble(
            self.model1,
            self.model2,
            ensemble_type="stacking",
            meta_learner_type=self.meta_learner_type,
            freeze_base_models=self.freeze_base_models,
            num_classes=self.num_classes
        )
        self.ensemble = self.ensemble.to(self.device)
        print("✓ Ensemble created")
        
        # Step 5: Train meta-learner
        print(f"\n{'='*60}")
        print(f"STEP 4: Training Meta-Learner")
        print(f"{'='*60}")
        self.train_meta_learner(self.num_epochs_meta)
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Models saved in: {self.models_dir}")
        print(f"{'='*70}")
        
        # Save training summary
        summary = {
            'run_name': self.run_name,
            'config': {
                'model1_arch': self.model1_arch,
                'model2_arch': self.model2_arch,
                'meta_learner_type': self.meta_learner_type,
                'num_epochs_base': self.num_epochs_base,
                'num_epochs_meta': self.num_epochs_meta,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'freeze_base_models': self.freeze_base_models
            },
            'results': {
                'total_training_time': total_time
            }
        }
        
        summary_path = self.results_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.writer.close()
        
        return self.ensemble, summary


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pair-wise stacking ensemble')
    parser.add_argument('--model1', type=str, default='densenet121',
                       help='First base model architecture')
    parser.add_argument('--model2', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b3', 'inception_resnet_v2'],
                       help='Second base model architecture')
    parser.add_argument('--meta-learner', type=str, default='neural_network',
                       choices=['neural_network', 'logistic'],
                       help='Meta-learner type')
    parser.add_argument('--epochs-base', type=int, default=5,
                       help='Epochs for base models')
    parser.add_argument('--epochs-meta', type=int, default=3,
                       help='Epochs for meta-learner')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"DiagXNet-Lite: Pair-wise Stacking Ensemble")
    print(f"{'='*70}\n")
    
    # Create trainer
    trainer = PairwiseStackingTrainer(
        model1_arch=args.model1,
        model2_arch=args.model2,
        meta_learner_type=args.meta_learner,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs_base=args.epochs_base,
        num_epochs_meta=args.epochs_meta
    )
    
    # Train
    ensemble, summary = trainer.train_full_pipeline()
    
    print("\n🎉 Ensemble training completed!")
    print("Next steps:")
    print("1. Run ensemble evaluation")
    print("2. Compare with individual model performance")
    print("3. Generate ensemble Grad-CAM visualizations")


if __name__ == "__main__":
    main()
