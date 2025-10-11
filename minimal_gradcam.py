"""
Minimal Grad-CAM implementation for DiagXNet-Lite
Designed to avoid multiprocessing issues and complete the 12 overlay requirement
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random

# Set up paths
import sys
sys.path.append('/Users/soranharuti/Desktop/diagxnet-lite')

from configs.config import CHEXPERT_LABELS, get_device
from src.data.dataset import CheXpertDataset, get_val_transforms
from src.models.architectures import create_model


class MinimalGradCAM:
    """Minimal Grad-CAM implementation"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, class_idx):
        """Generate CAM for specific class"""
        self.model.eval()
        
        # Forward pass
        input_image.requires_grad_()
        output = self.model(input_image)
        
        # Clear gradients
        self.model.zero_grad()
        
        # Backward pass for specific class
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def main():
    """Generate 12 Grad-CAM overlays"""
    print("🎯 Generating 12 Grad-CAM overlays for DiagXNet-Lite")
    
    # Load model
    device = get_device()
    print(f"Using device: {device}")
    
    model = create_model(architecture="densenet121", num_classes=14, pretrained=True, input_channels=1)
    
    # Load trained weights
    model_path = "/Users/soranharuti/Desktop/diagxnet-lite/results/diagxnet_lite_experiment_20250906_195656/trained_model.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get target layer (last conv layer)
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    
    print(f"Target layer: {target_layer}")
    
    # Initialize Grad-CAM
    gradcam = MinimalGradCAM(model, target_layer)
    
    # Create dataset (use training data since validation is empty)
    dataset = CheXpertDataset(
        csv_path="/Users/soranharuti/Desktop/diagxnet-lite/data/chexpert_small/CheXpert-v1.0-small/train/train.csv",
        data_root="/Users/soranharuti/Desktop/diagxnet-lite/data",
        transform=get_val_transforms(),
        uncertainty_policy="ignore",
        frontal_only=False  # Include all views to get more samples
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create output directory
    output_dir = Path("/Users/soranharuti/Desktop/diagxnet-lite/results/diagxnet_lite_experiment_20250906_195656/final_gradcam")
    output_dir.mkdir(exist_ok=True)
    
    # Find examples for each category
    print("🔍 Finding examples...")
    examples = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
    
    # Sample random indices
    sample_indices = random.sample(range(len(dataset)), min(500, len(dataset)))
    
    for idx in sample_indices:
        if all(len(examples[cat]) >= 5 for cat in examples.keys()):
            break
            
        try:
            image, labels, masks = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_batch)
                probs = torch.sigmoid(outputs)
            
            # Use most common pathology (Support Devices - index 13)
            label_idx = 13
            
            if masks[label_idx].item() == 1:  # Valid label
                prob = probs[0, label_idx].item()
                target = labels[label_idx].item()
                
                threshold = 0.5
                pred = prob >= threshold
                
                if target == 1 and pred == 1:
                    category = 'TP'
                elif target == 0 and pred == 0:
                    category = 'TN'
                elif target == 0 and pred == 1:
                    category = 'FP'
                elif target == 1 and pred == 0:
                    category = 'FN'
                else:
                    continue
                
                if len(examples[category]) < 5:
                    examples[category].append({
                        'idx': idx,
                        'prob': prob,
                        'target': target,
                        'label_idx': label_idx,
                        'image': image,
                        'category': category
                    })
                    print(f"Found {category}: prob={prob:.3f}, target={target}")
                    
        except Exception as e:
            print(f"Error processing {idx}: {e}")
            continue
    
    # Generate overlays
    print("\n🎨 Generating Grad-CAM overlays...")
    generated = 0
    
    for category in ['TP', 'TN', 'FP', 'FN']:
        category_examples = examples[category][:3]  # Take first 3
        
        for i, example in enumerate(category_examples):
            try:
                idx = example['idx']
                label_idx = example['label_idx']
                image = example['image']
                prob = example['prob']
                target = example['target']
                
                # Prepare image
                image_batch = image.unsqueeze(0).to(device)
                
                # Generate CAM
                cam = gradcam.generate_cam(image_batch, label_idx)
                
                # Convert to numpy
                original = image.squeeze().cpu().numpy()
                
                # Resize CAM
                cam_resized = cv2.resize(cam, (original.shape[1], original.shape[0]))
                
                # Create overlay
                original_norm = (original - original.min()) / (original.max() - original.min())
                
                # Create colored heatmap
                heatmap_colored = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
                
                # Create overlay
                original_rgb = np.stack([original_norm] * 3, axis=-1)
                overlay = 0.6 * original_rgb + 0.4 * heatmap_colored
                
                # Save
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(original, cmap='gray')
                plt.title(f'Original X-ray\n{category} (P={prob:.3f}, T={target})')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(cam_resized, cmap='jet')
                plt.title('Grad-CAM Heatmap')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title('Overlay')
                plt.axis('off')
                
                save_path = output_dir / f"{category}_{i+1}_sample{idx}.png"
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                generated += 1
                print(f"✅ Generated {save_path.name}")
                
            except Exception as e:
                print(f"❌ Failed to generate {category} {i+1}: {e}")
    
    print(f"\n🎉 Generated {generated} Grad-CAM overlays!")
    print(f"📁 Saved in: {output_dir}")
    
    # Create summary
    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"DiagXNet-Lite Grad-CAM Analysis\n")
        f.write(f"Generated: {generated}/12 overlays\n")
        f.write(f"Categories found:\n")
        for cat in examples:
            f.write(f"  {cat}: {len(examples[cat])} examples\n")
    
    print("✅ Grad-CAM analysis complete!")


if __name__ == "__main__":
    main()