"""
Ensemble models for pair-wise stacking
Combines predictions from multiple base models using a meta-learner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np


class MetaLearner(nn.Module):
    """
    Meta-learner for stacking ensemble
    Takes predictions from base models and learns optimal combination
    """
    
    def __init__(
        self,
        num_base_models: int = 2,
        num_classes: int = 14,
        meta_learner_type: str = "neural_network",
        hidden_dim: int = 64
    ):
        """
        Args:
            num_base_models: Number of base models in ensemble
            num_classes: Number of output classes
            meta_learner_type: Type of meta-learner ('neural_network' or 'logistic')
            hidden_dim: Hidden dimension for neural network meta-learner
        """
        super().__init__()
        
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        self.meta_learner_type = meta_learner_type
        
        input_dim = num_base_models * num_classes  # Concatenated predictions
        
        if meta_learner_type == "neural_network":
            # Simple feedforward network
            self.meta_model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        elif meta_learner_type == "logistic":
            # Simple linear layer (logistic regression)
            self.meta_model = nn.Linear(input_dim, num_classes)
        else:
            raise ValueError(f"Unknown meta_learner_type: {meta_learner_type}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, base_predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            base_predictions: List of prediction tensors from base models
                             Each tensor shape: (batch_size, num_classes)
        
        Returns:
            Final ensemble predictions: (batch_size, num_classes)
        """
        # Concatenate base model predictions
        combined = torch.cat(base_predictions, dim=1)  # (batch_size, num_base_models * num_classes)
        
        # Meta-learner prediction
        output = self.meta_model(combined)
        
        return output


class StackingEnsemble(nn.Module):
    """
    Pair-wise stacking ensemble
    Combines two base models using a meta-learner
    """
    
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        num_classes: int = 14,
        meta_learner_type: str = "neural_network",
        freeze_base_models: bool = True,
        hidden_dim: int = 64
    ):
        """
        Args:
            model1: First base model
            model2: Second base model
            num_classes: Number of output classes
            meta_learner_type: Type of meta-learner
            freeze_base_models: Whether to freeze base model weights during meta-training
            hidden_dim: Hidden dimension for meta-learner neural network
        """
        super().__init__()
        
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes
        self.freeze_base_models = freeze_base_models
        
        # Freeze base models if specified
        if freeze_base_models:
            for param in self.model1.parameters():
                param.requires_grad = False
            for param in self.model2.parameters():
                param.requires_grad = False
        
        # Create meta-learner
        self.meta_learner = MetaLearner(
            num_base_models=2,
            num_classes=num_classes,
            meta_learner_type=meta_learner_type,
            hidden_dim=hidden_dim
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
        
        Returns:
            ensemble_output: Final ensemble predictions (batch_size, num_classes)
            base_outputs: Dictionary of base model outputs
        """
        # Get predictions from base models
        with torch.set_grad_enabled(not self.freeze_base_models):
            output1 = self.model1(x)
            output2 = self.model2(x)
        
        # Meta-learner combines predictions
        ensemble_output = self.meta_learner([output1, output2])
        
        # Store base outputs for analysis
        base_outputs = {
            'model1': output1,
            'model2': output2
        }
        
        return ensemble_output, base_outputs
    
    def get_base_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get predictions from base models only"""
        self.eval()
        with torch.no_grad():
            output1 = self.model1(x)
            output2 = self.model2(x)
        
        return {
            'model1': output1,
            'model2': output2
        }
    
    def unfreeze_base_models(self):
        """Unfreeze base models for fine-tuning"""
        self.freeze_base_models = False
        for param in self.model1.parameters():
            param.requires_grad = True
        for param in self.model2.parameters():
            param.requires_grad = True
    
    def freeze_base_models_fn(self):
        """Freeze base models"""
        self.freeze_base_models = True
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False


class WeightedAverageEnsemble(nn.Module):
    """
    Simple weighted average ensemble (alternative to stacking)
    Learns weights for averaging base model predictions
    """
    
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        num_classes: int = 14,
        learnable_weights: bool = True
    ):
        """
        Args:
            model1: First base model
            model2: Second base model  
            num_classes: Number of output classes
            learnable_weights: Whether weights are learnable or fixed (0.5 each)
        """
        super().__init__()
        
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes
        self.learnable_weights = learnable_weights
        
        # Freeze base models
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        
        # Initialize weights
        if learnable_weights:
            # Learnable weights (will be softmax normalized)
            self.weight1 = nn.Parameter(torch.tensor(0.0))
            self.weight2 = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed equal weights
            self.register_buffer('weight1', torch.tensor(0.5))
            self.register_buffer('weight2', torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass"""
        # Get predictions from base models
        with torch.no_grad():
            output1 = self.model1(x)
            output2 = self.model2(x)
        
        # Normalize weights if learnable
        if self.learnable_weights:
            weights = F.softmax(torch.stack([self.weight1, self.weight2]), dim=0)
            w1, w2 = weights[0], weights[1]
        else:
            w1, w2 = self.weight1, self.weight2
        
        # Weighted average
        ensemble_output = w1 * output1 + w2 * output2
        
        base_outputs = {
            'model1': output1,
            'model2': output2,
            'weight1': w1.item() if isinstance(w1, torch.Tensor) else w1,
            'weight2': w2.item() if isinstance(w2, torch.Tensor) else w2
        }
        
        return ensemble_output, base_outputs


def create_ensemble(
    model1: nn.Module,
    model2: nn.Module,
    ensemble_type: str = "stacking",
    num_classes: int = 14,
    **kwargs
) -> nn.Module:
    """
    Factory function to create ensemble models
    
    Args:
        model1: First base model
        model2: Second base model
        ensemble_type: Type of ensemble ('stacking' or 'weighted_average')
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific ensemble types
    
    Returns:
        Ensemble model
    """
    if ensemble_type == "stacking":
        return StackingEnsemble(
            model1=model1,
            model2=model2,
            num_classes=num_classes,
            meta_learner_type=kwargs.get('meta_learner_type', 'neural_network'),
            freeze_base_models=kwargs.get('freeze_base_models', True),
            hidden_dim=kwargs.get('hidden_dim', 64)
        )
    elif ensemble_type == "weighted_average":
        return WeightedAverageEnsemble(
            model1=model1,
            model2=model2,
            num_classes=num_classes,
            learnable_weights=kwargs.get('learnable_weights', True)
        )
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


if __name__ == "__main__":
    # Test ensemble creation
    print("Testing ensemble models...")
    
    from architectures import create_model
    
    # Create base models
    print("\nCreating base models...")
    model1 = create_model("densenet121", num_classes=14, pretrained=False)
    model2 = create_model("efficientnet_b3", num_classes=14, pretrained=False)
    
    print("✓ Base models created")
    
    # Test stacking ensemble
    print("\nTesting stacking ensemble...")
    ensemble = create_ensemble(
        model1, model2,
        ensemble_type="stacking",
        meta_learner_type="neural_network",
        num_classes=14
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 224, 224)
    output, base_outputs = ensemble(x)
    
    print(f"✓ Stacking ensemble output shape: {output.shape}")
    print(f"  Model 1 output: {base_outputs['model1'].shape}")
    print(f"  Model 2 output: {base_outputs['model2'].shape}")
    
    # Test weighted average ensemble
    print("\nTesting weighted average ensemble...")
    ensemble2 = create_ensemble(
        model1, model2,
        ensemble_type="weighted_average",
        learnable_weights=True,
        num_classes=14
    )
    
    output2, base_outputs2 = ensemble2(x)
    print(f"✓ Weighted average output shape: {output2.shape}")
    print(f"  Weight 1: {base_outputs2['weight1']:.4f}")
    print(f"  Weight 2: {base_outputs2['weight2']:.4f}")
    
    print("\n✅ All tests passed!")
