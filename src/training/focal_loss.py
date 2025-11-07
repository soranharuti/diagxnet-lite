"""
Focal Loss Implementation for Class Imbalance
Addresses hard-to-classify examples and rare conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
        gamma: Exponent of the modulating factor (1 - p_t)^gamma
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: (batch_size, num_classes) - logits from model
            targets: (batch_size, num_classes) - ground truth labels (0 or 1)
            mask: (batch_size, num_classes) - mask for valid labels (1 for valid, 0 for uncertain/missing)
            
        Returns:
            focal_loss: scalar tensor
        """
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate pt (probability of correct class)
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # Calculate Focal Loss
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        
        # Apply mask if provided (for uncertain labels)
        if mask is not None:
            focal_loss = focal_loss * mask
            
            if self.reduction == 'mean':
                return focal_loss.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for class imbalance
    
    Args:
        pos_weights: Tensor of shape (num_classes,) with positive class weights
    """
    
    def __init__(self, pos_weights=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weights = pos_weights
        
    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs: (batch_size, num_classes) - logits from model
            targets: (batch_size, num_classes) - ground truth labels
            mask: (batch_size, num_classes) - mask for valid labels
            
        Returns:
            loss: scalar tensor
        """
        if self.pos_weights is not None:
            loss = F.binary_cross_entropy_with_logits(
                inputs, targets, 
                pos_weight=self.pos_weights.to(inputs.device),
                reduction='none'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply mask
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        else:
            return loss.mean()


def calculate_class_weights(dataset, num_classes):
    """
    Calculate inverse frequency weights for each class
    
    Args:
        dataset: CheXpertDataset instance
        num_classes: Number of classes
        
    Returns:
        weights: Tensor of shape (num_classes,) with weights for each class
    """
    print("\nCalculating class weights...")
    
    # Count positive samples for each class
    pos_counts = torch.zeros(num_classes)
    total_valid = torch.zeros(num_classes)
    
    for idx in range(len(dataset)):
        _, labels, masks = dataset[idx]
        pos_counts += labels * masks
        total_valid += masks
    
    # Calculate positive frequency
    pos_freq = pos_counts / (total_valid + 1e-8)
    
    # Calculate weights (inverse frequency)
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (pos_freq + 0.01)
    
    # Normalize weights to have mean = 1.0
    weights = weights / weights.mean()
    
    print("\nClass Prevalence and Weights:")
    print("-" * 60)
    for i in range(num_classes):
        print(f"  Class {i:2d}: Prevalence={pos_freq[i]:.4f}, Weight={weights[i]:.4f}")
    print("-" * 60)
    
    return weights


if __name__ == "__main__":
    # Test Focal Loss
    print("Testing Focal Loss...")
    
    batch_size = 4
    num_classes = 13
    
    # Simulate logits and targets
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()
    mask = torch.ones(batch_size, num_classes)
    
    # Test Focal Loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss_fn(inputs, targets, mask)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Weighted BCE Loss
    pos_weights = torch.ones(num_classes) * 2.0
    weighted_bce_fn = WeightedBCELoss(pos_weights)
    loss = weighted_bce_fn(inputs, targets, mask)
    print(f"Weighted BCE Loss: {loss.item():.4f}")
    
    print("\n[OK] Focal Loss implementation tested successfully!")

