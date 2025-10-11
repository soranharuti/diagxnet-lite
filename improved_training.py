"""
Improved Training Script for DiagXNet-Lite
Addresses class imbalance, overconfidence, and performance issues identified in enhanced evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.append('/Users/soranharuti/Desktop/diagxnet-lite')

from configs.config import CHEXPERT_LABELS, get_device, MODELS_DIR, RESULTS_DIR
from src.data.dataset import CheXpertDataset, get_train_transforms, get_val_transforms
from src.models.architectures import create_model
from src.evaluation.metrics import ModelEvaluator


class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance and hard examples"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets, masks=None):
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Ground truth [batch_size, num_classes]  
            masks: Valid label mask [batch_size, num_classes]
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight: (1 - p_t)^gamma
        # For positive targets: p_t = p, for negative: p_t = 1-p
        p_t = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combine weights
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        # Apply mask for uncertain labels
        if masks is not None:
            focal_loss = focal_loss * masks
            
        # Reduction
        if self.reduction == 'mean':
            if masks is not None:
                return focal_loss.sum() / masks.sum()
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss with per-class weights"""
    
    def __init__(self, class_weights):
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
    
    def forward(self, inputs, targets, masks=None):
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Ground truth [batch_size, num_classes]
            masks: Valid label mask [batch_size, num_classes]
        """
        # Expand class weights to batch dimension
        weights = self.class_weights.unsqueeze(0).expand_as(targets)
        
        # Compute weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_loss = bce_loss * weights
        
        # Apply mask
        if masks is not None:
            weighted_loss = weighted_loss * masks
            return weighted_loss.sum() / masks.sum()
        else:
            return weighted_loss.mean()


class TemperatureScaling(nn.Module):
    """Temperature scaling for confidence calibration"""
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature


class ImprovedTrainer:
    """Enhanced trainer with class balancing and advanced techniques"""
    
    def __init__(self, improvement_strategy='focal'):
        """
        Args:
            improvement_strategy: 'focal', 'weighted', 'balanced_sampling', or 'combined'
        """
        self.device = get_device()
        self.strategy = improvement_strategy
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.num_epochs = 5
        self.labels = CHEXPERT_LABELS
        
        # Create timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"improved_diagxnet_{self.strategy}_{self.timestamp}"
        self.results_dir = RESULTS_DIR / self.experiment_name
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Improved Training - Strategy: {self.strategy}")
        print(f"📁 Results will be saved to: {self.results_dir}")
    
    def create_balanced_data_loaders(self):
        """Create data loaders with various balancing strategies"""
        print("📊 Setting up improved data loaders...")
        
        # Import configs to get correct paths
        from configs.config import CHEXPERT_ROOT, DATA_ROOT
        
        train_csv = CHEXPERT_ROOT / "train" / "train.csv"
        
        # Create datasets
        train_dataset = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=get_train_transforms(augment=True),
            uncertainty_policy="ignore",
            frontal_only=True
        )
        
        val_dataset = CheXpertDataset(
            csv_path=train_csv,
            data_root=DATA_ROOT,
            transform=get_val_transforms(),
            uncertainty_policy="ignore", 
            frontal_only=True
        )
        
        # Split data
        total_size = len(train_dataset)
        val_size = int(total_size * 0.2)
        train_size = total_size - val_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, total_size))
        
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset, val_indices)
        
        # Get class weights
        self.class_weights = train_dataset.get_class_weights().to(self.device)
        print(f"📊 Class weights calculated: {self.class_weights}")
        
        # Create data loaders based on strategy
        if self.strategy == 'balanced_sampling' or self.strategy == 'combined':
            # Create weighted sampler for balanced sampling
            sample_weights = self._calculate_sample_weights(train_dataset, train_indices)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
        else:
            self.train_loader = DataLoader(
                train_subset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
        
        self.val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Data loaders created - Train: {len(self.train_loader)}, Val: {len(self.val_loader)}")
    
    def _calculate_sample_weights(self, dataset, indices):
        """Calculate per-sample weights for balanced sampling"""
        print("⚖️ Calculating sample weights for balanced sampling...")
        
        sample_weights = []
        
        for idx in indices:
            _, labels, masks = dataset[idx]
            
            # Calculate weight as inverse of positive class frequency
            sample_weight = 0
            valid_labels = 0
            
            for i, (label, mask) in enumerate(zip(labels, masks)):
                if mask == 1:  # Valid label
                    class_weight = self.class_weights[i].item()
                    if label == 1:  # Positive
                        sample_weight += class_weight
                    else:  # Negative
                        sample_weight += 1.0
                    valid_labels += 1
            
            # Average weight across valid labels
            if valid_labels > 0:
                sample_weight = sample_weight / valid_labels
            else:
                sample_weight = 1.0
            
            sample_weights.append(sample_weight)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float)
        print(f"✅ Sample weights calculated (range: {sample_weights.min():.3f} - {sample_weights.max():.3f})")
        
        return sample_weights
    
    def create_model_and_loss(self):
        """Create model and loss function based on strategy"""
        print(f"🧠 Creating model with {self.strategy} strategy...")
        
        # Create model
        self.model = create_model('densenet121', num_classes=len(self.labels), pretrained=True)
        self.model.to(self.device)
        
        # Create loss function based on strategy
        if self.strategy == 'focal' or self.strategy == 'combined':
            # Use focal loss with alpha based on class imbalance
            alpha = 0.25  # Standard focal loss alpha
            gamma = 2.0   # Standard focal loss gamma
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
            print(f"📐 Using Focal Loss (alpha={alpha}, gamma={gamma})")
            
        elif self.strategy == 'weighted':
            self.criterion = WeightedBCELoss(self.class_weights)
            print("📐 Using Weighted BCE Loss")
            
        else:  # balanced_sampling
            self.criterion = nn.BCEWithLogitsLoss()
            print("📐 Using standard BCE Loss with balanced sampling")
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        print(f"🎯 Using Adam optimizer (lr={self.learning_rate})")
        
        # Temperature scaling for calibration
        self.temperature_model = TemperatureScaling().to(self.device)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, targets, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"    Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1} - Average Training Loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_outputs = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch_idx, (images, targets, masks) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets, masks)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(masks.cpu())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        outputs_tensor = torch.cat(all_outputs)
        targets_tensor = torch.cat(all_targets)
        masks_tensor = torch.cat(all_masks)
        
        # Convert to numpy
        predictions = torch.sigmoid(outputs_tensor).numpy()
        targets_np = targets_tensor.numpy()
        masks_np = masks_tensor.numpy()
        
        # Calculate AUROC for monitoring
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(len(self.labels)):
            valid_mask = masks_np[:, i] == 1
            if valid_mask.sum() > 0 and len(np.unique(targets_np[valid_mask, i])) > 1:
                auc = roc_auc_score(targets_np[valid_mask, i], predictions[valid_mask, i])
                aucs.append(auc)
        
        mean_auc = np.mean(aucs) if aucs else 0.0
        print(f"  Epoch {epoch + 1} - Validation Loss: {avg_loss:.4f}, Mean AUROC: {mean_auc:.4f}")
        
        return avg_loss, mean_auc
    
    def calibrate_temperature(self):
        """Calibrate model confidence using temperature scaling"""
        print("🌡️ Calibrating model confidence with temperature scaling...")
        
        self.model.eval()
        self.temperature_model.train()
        
        # Collect validation logits and targets
        all_logits = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for images, targets, masks in self.val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                all_masks.append(masks.cpu())
        
        logits_tensor = torch.cat(all_logits).to(self.device)
        targets_tensor = torch.cat(all_targets).to(self.device)
        masks_tensor = torch.cat(all_masks).to(self.device)
        
        # Optimize temperature
        temp_optimizer = torch.optim.LBFGS([self.temperature_model.temperature], lr=0.01, max_iter=50)
        
        def temp_loss():
            temp_optimizer.zero_grad()
            scaled_logits = self.temperature_model(logits_tensor)
            loss = F.binary_cross_entropy_with_logits(scaled_logits, targets_tensor, reduction='none')
            loss = (loss * masks_tensor).sum() / masks_tensor.sum()
            loss.backward()
            return loss
        
        temp_optimizer.step(temp_loss)
        
        print(f"✅ Temperature scaling complete. Optimal temperature: {self.temperature_model.temperature.item():.3f}")
    
    def train_complete_model(self):
        """Train the complete improved model"""
        print("🚀 Starting improved training...")
        
        # Setup
        self.create_balanced_data_loaders()
        self.create_model_and_loss()
        
        # Training loop
        train_losses = []
        val_losses = []
        val_aucs = []
        
        best_auc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\n📚 Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Validate  
            val_loss, val_auc = self.validate_epoch(epoch)
            val_losses.append(val_loss)
            val_aucs.append(val_auc)
            
            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'strategy': self.strategy,
                    'class_weights': self.class_weights
                }, self.results_dir / 'best_model.pth')
                print(f"    💾 Saved new best model (AUROC: {val_auc:.4f})")
        
        # Calibrate temperature
        self.calibrate_temperature()
        
        # Save final model with temperature
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'temperature_state_dict': self.temperature_model.state_dict(),
            'epoch': self.num_epochs - 1,
            'final_val_auc': val_aucs[-1],
            'best_val_auc': best_auc,
            'strategy': self.strategy,
            'class_weights': self.class_weights,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,  
                'val_aucs': val_aucs
            }
        }, self.results_dir / 'final_calibrated_model.pth')
        
        print(f"\n🎉 Training complete!")
        print(f"   Best validation AUROC: {best_auc:.4f}")
        print(f"   Final validation AUROC: {val_aucs[-1]:.4f}")
        print(f"   Models saved to: {self.results_dir}")
        
        return {
            'best_auc': best_auc,
            'final_auc': val_aucs[-1],
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_aucs': val_aucs
            }
        }


def main():
    """Run improved training with different strategies"""
    
    strategies = ['focal', 'weighted', 'balanced_sampling']
    
    print("🎯 IMPROVED TRAINING FOR DIAGXNET-LITE")
    print("="*60)
    print("Available strategies:")
    print("1. focal - Focal Loss for hard examples")
    print("2. weighted - Weighted BCE with class weights") 
    print("3. balanced_sampling - Balanced data sampling")
    print("4. combined - Focal loss + balanced sampling")
    
    # For now, let's try focal loss (most effective for imbalance)
    strategy = 'focal'
    
    print(f"\n🚀 Running strategy: {strategy}")
    
    trainer = ImprovedTrainer(improvement_strategy=strategy)
    results = trainer.train_complete_model()
    
    print("\n📊 IMPROVEMENT RESULTS:")
    print(f"Strategy: {strategy}")
    print(f"Best AUROC: {results['best_auc']:.4f}")
    print(f"Final AUROC: {results['final_auc']:.4f}")
    
    return results

if __name__ == "__main__":
    main()