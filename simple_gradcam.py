"""
Simplified Grad-CAM generation for DiagXNet-Lite
Avoiding multiprocessing issues by using direct dataset access
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
import random

# Import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs.config import CHEXPERT_LABELS, get_device
from src.data.dataset import create_data_loaders, CheXpertDataset
from src.models.architectures import create_model


class SimpleGradCAM:
    """Simple Grad-CAM implementation without complex data loading"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx):
        """Generate CAM for specific class"""
        # Forward pass
        output = self.model(input_image)
        
        # Clear gradients
        self.model.zero_grad()
        
        # Backward pass for specific class
        class_loss = output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()


def load_latest_model():
    """Load the latest trained model"""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("No results directory found")
    
    # Find latest experiment
    experiments = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('diagxnet_lite_experiment')]
    if not experiments:
        raise FileNotFoundError("No experiment directories found")
    
    latest_exp = max(experiments, key=lambda x: x.name)
    model_path = latest_exp / "trained_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")
    
    print(f"Loading model from: {model_path}")
    device = get_device()
    model = create_model(architecture='densenet121', num_classes=len(CHEXPERT_LABELS)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device, latest_exp


def get_prediction_category(prob, target, threshold=0.5):
    """Determine if prediction is TP, TN, FP, or FN"""
    pred = prob >= threshold
    
    if target == 1 and pred == 1:
        return 'TP'
    elif target == 0 and pred == 0:
        return 'TN'
    elif target == 0 and pred == 1:
        return 'FP'
    elif target == 1 and pred == 0:
        return 'FN'


def generate_gradcam_samples():
    """Generate Grad-CAM samples without multiprocessing"""
    
    # Load model
    model, device, exp_dir = load_latest_model()
    
    # Get target layer (last conv layer in DenseNet features)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    print(f"Using target layer: {target_layer}")
    
    # Initialize Grad-CAM
    gradcam = SimpleGradCAM(model, target_layer)
    
    # Get dataset (without data loader to avoid multiprocessing)
    from configs.config import DATA_ROOT
    
    train_csv = DATA_ROOT / "CheXpert-v1.0-small" / "train" / "train.csv"
    val_csv = DATA_ROOT / "CheXpert-v1.0-small" / "valid" / "valid.csv"
    
    train_loader, val_loader = create_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv, 
        batch_size=1, 
        num_workers=0
    )
    dataset = val_loader.dataset
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    gradcam_dir = exp_dir / "simple_gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    
    # Generate predictions for a subset to find examples
    print("Finding examples for each category...")
    examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    
    # Sample more indices to find examples
    sample_size = min(20, len(dataset))  # Start with just 20 samples for debugging
    sample_indices = list(range(sample_size))
    
    total_processed = 0
    total_uncertain = 0
    total_valid = 0
    
    for idx in sample_indices:
        if len(examples['TP']) >= 3 and len(examples['TN']) >= 3 and \
           len(examples['FP']) >= 3 and len(examples['FN']) >= 3:
            break
            
        try:
            image, labels, masks = dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image)
                probs = torch.sigmoid(outputs)
            
            total_processed += 1
            
            # Print first sample completely for debugging
            if idx == 0:
                print(f"First sample debug:")
                print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
                print(f"Masks shape: {masks.shape}, dtype: {masks.dtype}")
                print(f"Labels: {labels}")
                print(f"Masks: {masks}")
                print(f"Probs: {probs.squeeze()}")
            
            # Try multiple pathologies to find examples
            for label_idx in range(len(CHEXPERT_LABELS)):
                prob = probs[0, label_idx].item()
                target = labels[label_idx].item()
                mask = masks[label_idx].item()
                
                # Check for NaN values
                if torch.isnan(labels[label_idx]) or mask == 0:  # Invalid/uncertain
                    total_uncertain += 1
                    continue
                    
                total_valid += 1
                category = get_prediction_category(prob, target, threshold=0.5)
                
                # Debug: Print first few valid examples
                if total_valid <= 5:
                    print(f"Valid example: Sample {idx}, Label {CHEXPERT_LABELS[label_idx]}: prob={prob:.3f}, target={target}, mask={mask}, category={category}")
                
                if category and len(examples[category]) < 3:
                    examples[category].append({
                        'idx': idx,
                        'prob': prob,
                        'target': target,
                        'label_idx': label_idx,
                        'label_name': CHEXPERT_LABELS[label_idx]
                    })
                    print(f"*** Found {category} example: idx={idx}, prob={prob:.3f}, target={target}, label={CHEXPERT_LABELS[label_idx]}")
                
                # Break early if we have enough examples
                if len(examples['TP']) >= 3 and len(examples['TN']) >= 3 and \
                   len(examples['FP']) >= 3 and len(examples['FN']) >= 3:
                    break
                
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Processed {total_processed} samples, found {total_uncertain} uncertain labels, {total_valid} valid labels")
    
    # Generate Grad-CAM for best examples
    print("\nGenerating Grad-CAM overlays...")
    generated = 0
    
    for category in ['TP', 'TN', 'FP', 'FN']:
        print(f"\nProcessing {category} examples...")
        
        # Select available examples (up to 3)
        category_examples = examples[category][:3]
        
        if not category_examples:
            print(f"No {category} examples found")
            continue
        
        for i, example in enumerate(category_examples):
            try:
                idx = example['idx']
                label_idx = example['label_idx']
                label_name = example['label_name']
                
                # Get image
                image, labels, masks = dataset[idx]
                image = image.unsqueeze(0).to(device)
                
                # Generate CAM
                cam = gradcam.generate_cam(image, label_idx)
                
                # Convert image to numpy
                original = image.squeeze().cpu().numpy()
                
                # Resize CAM to match image
                cam_resized = cv2.resize(cam, (original.shape[1], original.shape[0]))
                
                # Create overlay
                heatmap_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                # Normalize original image
                original_norm = (original - original.min()) / (original.max() - original.min())
                original_rgb = np.stack([original_norm] * 3, axis=-1)
                
                # Create overlay
                overlay = 0.6 * original_rgb + 0.4 * heatmap_colored / 255.0
                
                # Save visualization
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(original, cmap='gray')
                plt.title(f'Original\n{category} - {label_name}\n(prob={example["prob"]:.3f})')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(cam_resized, cmap='jet')
                plt.title('Grad-CAM')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title('Overlay')
                plt.axis('off')
                
                save_path = gradcam_dir / f"{category}_{i+1}_{label_name}_idx{idx}.png"
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                generated += 1
                print(f"✅ Generated {category} {i+1}: {save_path}")
                
            except Exception as e:
                print(f"❌ Failed to generate {category} {i+1}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n🎯 Generated {generated} Grad-CAM overlays in {gradcam_dir}")
    print(f"Found examples: TP={len(examples['TP'])}, TN={len(examples['TN'])}, FP={len(examples['FP'])}, FN={len(examples['FN'])}")


if __name__ == "__main__":
    generate_gradcam_samples()
