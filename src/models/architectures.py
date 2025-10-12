"""
Model architectures for DiagXNet-Lite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Optional
import warnings

# Import configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'configs'))
from config import CHEXPERT_LABELS


class BaseModel(nn.Module):
    """Base model class with common functionality"""
    
    def __init__(self, num_classes: int = len(CHEXPERT_LABELS)):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, x):
        raise NotImplementedError
        
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DenseNetModel(BaseModel):
    """
    DenseNet-based model for chest X-ray classification
    
    Features:
    - Pre-trained DenseNet backbone
    - Custom classifier head
    - Optional dropout and batch normalization
    """
    
    def __init__(
        self,
        architecture: str = "densenet121",
        num_classes: int = len(CHEXPERT_LABELS),
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_bn: bool = True
    ):
        super().__init__(num_classes)
        
        # Get DenseNet backbone
        if architecture == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = 1024
        elif architecture == "densenet169":
            self.backbone = models.densenet169(pretrained=pretrained)
            feature_dim = 1664
        elif architecture == "densenet201":
            self.backbone = models.densenet201(pretrained=pretrained)
            feature_dim = 1920
        else:
            raise ValueError(f"Unsupported DenseNet architecture: {architecture}")
        
        # Modify first conv layer for grayscale input
        original_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Copy weights for first channel if pretrained
        if pretrained:
            with torch.no_grad():
                self.backbone.features.conv0.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        classifier_layers = []
        
        if use_bn:
            classifier_layers.append(nn.BatchNorm1d(feature_dim))
        
        classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier weights
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global average pooling is already done in DenseNet
        # features shape: (batch_size, feature_dim)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class ResNetModel(BaseModel):
    """
    ResNet-based model for chest X-ray classification
    """
    
    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = len(CHEXPERT_LABELS),
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_bn: bool = True
    ):
        super().__init__(num_classes)
        
        # Get ResNet backbone
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        elif architecture == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")
        
        # Modify first conv layer for grayscale input
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Copy weights for first channel if pretrained
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier head
        classifier_layers = []
        
        if use_bn:
            classifier_layers.append(nn.BatchNorm1d(feature_dim))
        
        classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier weights
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class EfficientNetModel(BaseModel):
    """
    EfficientNet-based model for chest X-ray classification
    """
    
    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = len(CHEXPERT_LABELS),
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_bn: bool = True
    ):
        super().__init__(num_classes)
        
        # Feature dimensions for different EfficientNet variants
        feature_dims = {
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
            "efficientnet_b4": 1792,
            "efficientnet_b5": 2048,
            "efficientnet_b6": 2304,
            "efficientnet_b7": 2560
        }
        
        if architecture not in feature_dims:
            raise ValueError(f"Unsupported EfficientNet architecture: {architecture}")
        
        # Get EfficientNet backbone
        self.backbone = getattr(models, architecture)(pretrained=pretrained)
        feature_dim = feature_dims[architecture]
        
        # Modify first conv layer for grayscale input
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Copy weights for first channel if pretrained
        if pretrained:
            with torch.no_grad():
                self.backbone.features[0][0].weight[:, 0, :, :] = original_conv.weight.mean(dim=1)
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        classifier_layers = []
        
        if use_bn:
            classifier_layers.append(nn.BatchNorm1d(feature_dim))
        
        classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier weights
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class InceptionResNetV2Model(BaseModel):
    """
    Inception-ResNet-V2 model for chest X-ray classification
    Using timm library for Inception-ResNet-V2
    """
    
    def __init__(
        self,
        num_classes: int = len(CHEXPERT_LABELS),
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        use_bn: bool = True
    ):
        super().__init__(num_classes)
        
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        # Load Inception-ResNet-V2
        self.backbone = timm.create_model(
            'inception_resnet_v2',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            in_chans=1  # Grayscale input
        )
        
        feature_dim = 1536  # Inception-ResNet-V2 feature dimension
        
        # Custom classifier head
        classifier_layers = []
        
        if use_bn:
            classifier_layers.append(nn.BatchNorm1d(feature_dim))
        
        classifier_layers.append(nn.Dropout(dropout_rate))
        classifier_layers.append(nn.Linear(feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize classifier weights
        if hasattr(self.classifier[-1], 'weight'):
            nn.init.xavier_uniform_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class AttentionBlock(nn.Module):
    """Attention mechanism for highlighting important regions"""
    
    def __init__(self, feature_dim: int, attention_dim: int = 256):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, features):
        # features: (batch_size, num_patches, feature_dim)
        attention_weights = self.attention(features)  # (batch_size, num_patches, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        attended_features = (features * attention_weights).sum(dim=1)  # (batch_size, feature_dim)
        
        return attended_features, attention_weights


class AttentionModel(BaseModel):
    """
    Attention-based model for interpretable chest X-ray classification
    
    Uses spatial attention to highlight important regions in the image
    """
    
    def __init__(
        self,
        backbone_architecture: str = "resnet50",
        num_classes: int = len(CHEXPERT_LABELS),
        pretrained: bool = True,
        attention_dim: int = 256,
        dropout_rate: float = 0.2
    ):
        super().__init__(num_classes)
        
        # Get backbone model
        if backbone_architecture.startswith("resnet"):
            self.backbone = ResNetModel(
                architecture=backbone_architecture,
                num_classes=0,  # No classifier
                pretrained=pretrained,
                dropout_rate=0.0  # No dropout in backbone
            )
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
            if backbone_architecture in ["resnet18", "resnet34"]:
                feature_dim = 512
            else:
                feature_dim = 2048
                
        elif backbone_architecture.startswith("densenet"):
            self.backbone = DenseNetModel(
                architecture=backbone_architecture,
                num_classes=0,  # No classifier
                pretrained=pretrained,
                dropout_rate=0.0  # No dropout in backbone
            )
            # Remove classifier
            self.backbone.classifier = nn.Identity()
            
            if backbone_architecture == "densenet121":
                feature_dim = 1024
            elif backbone_architecture == "densenet169":
                feature_dim = 1664
            else:  # densenet201
                feature_dim = 1920
        else:
            raise ValueError(f"Unsupported backbone: {backbone_architecture}")
        
        # Modify backbone to return feature maps instead of global pooled features
        self._modify_backbone_for_attention(backbone_architecture)
        
        # Attention mechanism
        self.attention = AttentionBlock(feature_dim, attention_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)
    
    def _modify_backbone_for_attention(self, architecture: str):
        """Modify backbone to return spatial feature maps"""
        if architecture.startswith("resnet"):
            # Remove global average pooling
            self.backbone.backbone.avgpool = nn.Identity()
        elif architecture.startswith("densenet"):
            # DenseNet doesn't have explicit global pooling, 
            # we'll handle this in forward pass
            pass
    
    def forward(self, x):
        # Extract feature maps
        if hasattr(self.backbone.backbone, 'features'):  # DenseNet
            feature_maps = self.backbone.backbone.features(x)
            # Global average pooling to get spatial features
            feature_maps = F.adaptive_avg_pool2d(feature_maps, (7, 7))
        else:  # ResNet
            # For ResNet, we need to extract features before the final pooling
            x = self.backbone.backbone.conv1(x)
            x = self.backbone.backbone.bn1(x)
            x = self.backbone.backbone.relu(x)
            x = self.backbone.backbone.maxpool(x)
            
            x = self.backbone.backbone.layer1(x)
            x = self.backbone.backbone.layer2(x)
            x = self.backbone.backbone.layer3(x)
            feature_maps = self.backbone.backbone.layer4(x)
        
        # Reshape for attention: (batch_size, channels, height, width) -> (batch_size, height*width, channels)
        batch_size, channels, height, width = feature_maps.shape
        feature_patches = feature_maps.view(batch_size, channels, -1).permute(0, 2, 1)
        
        # Apply attention
        attended_features, attention_weights = self.attention(feature_patches)
        
        # Classification
        logits = self.classifier(attended_features)
        
        return logits, attention_weights.view(batch_size, height, width)


def create_model(
    architecture: str,
    num_classes: int = len(CHEXPERT_LABELS),
    pretrained: bool = True,
    **kwargs
) -> BaseModel:
    """
    Factory function to create models
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific arguments
    
    Returns:
        Model instance
    """
    
    architecture = architecture.lower()
    
    if architecture.startswith("densenet"):
        # Filter kwargs for DenseNet - remove input_channels as it's not needed
        densenet_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout_rate', 'use_bn']}
        return DenseNetModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **densenet_kwargs
        )
    elif architecture.startswith("resnet"):
        # Filter kwargs for ResNet
        resnet_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout_rate', 'use_bn']}
        return ResNetModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **resnet_kwargs
        )
    elif architecture.startswith("efficientnet"):
        # Filter kwargs for EfficientNet
        efficientnet_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout_rate', 'use_bn']}
        return EfficientNetModel(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            **efficientnet_kwargs
        )
    elif architecture == "inception_resnet_v2":
        # Filter kwargs for Inception-ResNet-V2
        inception_kwargs = {k: v for k, v in kwargs.items() if k in ['dropout_rate', 'use_bn']}
        return InceptionResNetV2Model(
            num_classes=num_classes,
            pretrained=pretrained,
            **inception_kwargs
        )
    elif architecture.startswith("attention"):
        # attention_resnet50, attention_densenet121, etc.
        backbone = architecture.replace("attention_", "")
        return AttentionModel(
            backbone_architecture=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def get_model_info(model: BaseModel) -> Dict:
    """Get information about a model"""
    return {
        "architecture": model.__class__.__name__,
        "num_parameters": model.get_num_parameters(),
        "num_classes": model.num_classes,
        "model_size_mb": model.get_num_parameters() * 4 / (1024 * 1024)  # Assuming float32
    }


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test different architectures
    architectures = [
        "densenet121",
        "resnet50", 
        "efficientnet_b0",
        "attention_resnet50"
    ]
    
    for arch in architectures:
        print(f"\nTesting {arch}:")
        try:
            model = create_model(arch, num_classes=14, pretrained=False)
            info = get_model_info(model)
            
            print(f"  ✓ Created successfully")
            print(f"  Parameters: {info['num_parameters']:,}")
            print(f"  Size: {info['model_size_mb']:.1f} MB")
            
            # Test forward pass
            x = torch.randn(2, 1, 224, 224)
            
            if arch.startswith("attention"):
                output, attention = model(x)
                print(f"  Output shape: {output.shape}")
                print(f"  Attention shape: {attention.shape}")
            else:
                output = model(x)
                print(f"  Output shape: {output.shape}")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\nModel testing completed!")
