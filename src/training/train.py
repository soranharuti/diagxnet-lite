"""
Training script for DiagXNet-Lite: DenseNet-121 fine-tuning on CheXpert
Following the project proposal specifications exactly
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from configs.config import *

class DenseNetTrainer:
    """
    Trainer class for DenseNet-121 fine-tuning on CheXpert dataset
    Following exact specifications from project proposal
    """
    
    def __init__(self, config=None):
        """Initialize trainer with project specifications"""
        # Project-specific parameters (from proposal)
        self.batch_size = 16          # Specified in proposal
        self.learning_rate = 1e-4     # Specified in proposal  
        self.num_epochs = 5           # Specified in proposal
        self.num_classes = 14         # CheXpert labels
        
        # Device setup
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Paths
        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"densenet121_chexpert_{self.timestamp}"
        
        # Setup logging
        self.writer = SummaryWriter(f"{self.results_dir}/tensorboard/{self.run_name}")
        
    def create_data_loaders(self):
        """Create data loaders following project specifications"""
        print("Creating data loaders...")
        
        # Data paths
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        
        # Project-specific preprocessing: Resize 256 → CenterCrop 224, ImageNet normalization
        train_transform = get_train_transforms(augment=True)  # Minimal augmentation
        val_transform = get_val_transforms()
        
        # Create datasets with frontal view only (typical for chest X-ray analysis)
        full_dataset = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=train_transform,
            uncertainty_policy="ignore",  # Standard practice
            frontal_only=True
        )
        
        # Split data (80/20 as common practice)
        total_size = len(full_dataset)
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        
        # Create indices for reproducible split
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        
        # Validation dataset with different transforms
        val_dataset_full = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=val_transform,
            uncertainty_policy="ignore",
            frontal_only=True
        )
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
        
        # Create data loaders with project specifications
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,  # Batch 16 from proposal
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
        
        print(f"Training samples: {len(train_dataset):,}")
        print(f"Validation samples: {len(val_dataset):,}")
        print(f"Training batches: {len(self.train_loader):,}")
        print(f"Validation batches: {len(self.val_loader):,}")
        
        # Get class weights for imbalanced data
        self.class_weights = full_dataset.get_class_weights().to(self.device)
        print(f"Class weights calculated: {self.class_weights}")
        
    def create_model(self):
        """Create DenseNet-121 model as specified in proposal"""
        print("Creating DenseNet-121 model...")
        
        # Create ImageNet pre-trained DenseNet-121
        self.model = create_model(
            architecture="densenet121",
            num_classes=self.num_classes,
            pretrained=True,  # ImageNet pre-trained as specified
            input_channels=1  # Grayscale chest X-rays
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model created: DenseNet-121")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    def create_optimizer_and_loss(self):
        """Create optimizer and loss function as specified"""
        print("Setting up optimizer and loss function...")
        
        # Adam optimizer with lr=1e-4 as specified
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,  # 1e-4 from proposal
            weight_decay=1e-4
        )
        
        # BCEWithLogitsLoss as specified in proposal
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights  # Handle class imbalance
        )
        
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Loss function: BCEWithLogitsLoss with class weights")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels, masks) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Apply mask for uncertain labels (ignore policy)
            masked_outputs = outputs * masks
            masked_labels = labels * masks
            
            # Calculate loss
            loss = self.criterion(masked_outputs, masked_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, "
                      f"Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        all_masks = []
        
        with torch.no_grad():
            for images, labels, masks in self.val_loader:
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Apply mask
                masked_outputs = outputs * masks
                masked_labels = labels * masks
                
                # Calculate loss
                loss = self.criterion(masked_outputs, masked_labels)
                running_loss += loss.item()
                
                # Store for metrics calculation
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                all_masks.append(masks.cpu())
        
        avg_loss = running_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate basic metrics (full metrics calculation in separate module)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        
        # Simple accuracy for monitoring
        probs = torch.sigmoid(all_outputs)
        preds = (probs > 0.5).float()
        
        # Masked accuracy
        correct = ((preds == all_labels) * all_masks).sum()
        total = all_masks.sum()
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy.item()
    
    def save_checkpoint(self, epoch, metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'class_weights': self.class_weights,
            'config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'architecture': 'densenet121'
            }
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        # Save checkpoint
        checkpoint_path = self.models_dir / f"{self.run_name}_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if epoch == self.num_epochs - 1:  # Last epoch
            best_path = self.models_dir / f"{self.run_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop - 5 epochs as specified"""
        print(f"\n{'='*60}")
        print(f"Starting DiagXNet-Lite Training")
        print(f"Model: DenseNet-121 (ImageNet pre-trained)")
        print(f"Dataset: CheXpert-small")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, {'val_loss': val_loss, 'val_accuracy': val_acc})
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per epoch: {total_time/self.num_epochs:.1f} seconds")
        print(f"Final validation loss: {self.val_losses[-1]:.4f}")
        print(f"Final validation accuracy: {val_acc:.4f}")
        print(f"{'='*60}")
        
        # Save training summary
        summary = {
            'run_name': self.run_name,
            'config': {
                'epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'architecture': 'densenet121'
            },
            'results': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'final_val_accuracy': val_acc,
                'total_training_time': total_time
            }
        }
        
        summary_path = self.results_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.writer.close()
        
        return self.model, summary
    
    def setup_and_train(self):
        """Complete setup and training pipeline"""
        try:
            # Setup
            self.create_data_loaders()
            self.create_model()
            self.create_optimizer_and_loss()
            
            # Train
            model, summary = self.train()
            
            print("\n✅ Training completed successfully!")
            print(f"📁 Results saved in: {self.results_dir}")
            print(f"💾 Model saved as: {self.run_name}_best.pth")
            
            return model, summary
            
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            raise e


def main():
    """Main function to run training"""
    print("DiagXNet-Lite: Training DenseNet-121 on CheXpert")
    print("=" * 60)
    
    # Create trainer
    trainer = DenseNetTrainer()
    
    # Run training
    model, summary = trainer.setup_and_train()
    
    print("\n🎉 DiagXNet-Lite training completed!")
    print("Next steps:")
    print("1. Run evaluation metrics (AUROC, AUPRC, F1)")
    print("2. Perform calibration analysis")
    print("3. Generate Grad-CAM visualizations")


if __name__ == "__main__":
    main()
