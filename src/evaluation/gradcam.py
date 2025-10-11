"""
Grad-CAM visualization for DiagXNet-Lite following project proposal specifications:
- Create 12 Grad-CAM overlays (3 TP, 3 TN, 3 FP, 3 FN)
- Show sensible image areas for true and false predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import random
from typing import List, Tuple, Dict

# Import configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from configs.config import CHEXPERT_LABELS, RESULTS_DIR


class GradCAM:
    """
    Grad-CAM implementation for chest X-ray interpretation
    """
    
    def __init__(self, model, target_layer_name='features'):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained model
            target_layer_name: Name of target layer for activation maps
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def forward_hook(module, input, output):
            self.activations = output
            
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            # Fallback: find the last convolutional layer
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                    target_layer = module
                elif 'features' in name and hasattr(module, 'norm5'):
                    # DenseNet specific
                    target_layer = module.norm5
        
        if target_layer is None:
            raise ValueError("Could not find suitable target layer")
            
        print(f"Using target layer: {target_layer}")
        
        # Register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, class_idx):
        """
        Generate Class Activation Map
        
        Args:
            input_tensor: Input image tensor [1, C, H, W]
            class_idx: Index of target class
            
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Generate CAM
        gradients = self.gradients.data[0]  # [C, H, W]
        activations = self.activations.data[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image as numpy array [H, W] or [H, W, C]
            heatmap: Heatmap as numpy array [H, W]
            alpha: Transparency factor
            colormap: OpenCV colormap
            
        Returns:
            Overlayed image
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to uint8
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlayed


class DiagXNetGradCAMAnalyzer:
    """
    Grad-CAM analyzer specifically for DiagXNet-Lite project
    Generates the 12 required overlays following proposal specifications
    """
    
    def __init__(self, model, device, data_loader, labels=CHEXPERT_LABELS):
        """
        Initialize analyzer
        
        Args:
            model: Trained DiagXNet model
            device: torch device
            data_loader: Data loader for analysis
            labels: List of pathology labels
        """
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.labels = labels
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(model)
        
        # Storage for examples
        self.examples = {
            'TP': [],  # True Positives
            'TN': [],  # True Negatives  
            'FP': [],  # False Positives
            'FN': []   # False Negatives
        }
        
    def collect_examples(self, probabilities, targets, masks, thresholds, max_per_category=50):
        """
        Collect examples for each category (TP, TN, FP, FN)
        
        Args:
            probabilities: Model predictions [n_samples, n_classes]
            targets: True labels [n_samples, n_classes]
            masks: Valid label masks [n_samples, n_classes]
            thresholds: Optimal thresholds per class
            max_per_category: Maximum examples to collect per category
        """
        print("Collecting examples for Grad-CAM analysis...")
        
        # Convert probabilities to predictions using thresholds
        predictions = np.zeros_like(probabilities)
        for i, label in enumerate(self.labels):
            threshold = thresholds.get(label, 0.5)
            predictions[:, i] = (probabilities[:, i] >= threshold).astype(int)
        
        # Collect examples for each label and category
        for label_idx, label in enumerate(self.labels):
            # Get valid samples for this label
            valid_mask = masks[:, label_idx] == 1
            
            if valid_mask.sum() == 0:
                continue
                
            valid_indices = np.where(valid_mask)[0]
            label_targets = targets[valid_indices, label_idx]
            label_predictions = predictions[valid_indices, label_idx]
            label_probabilities = probabilities[valid_indices, label_idx]
            
            # Categorize predictions
            tp_indices = valid_indices[(label_targets == 1) & (label_predictions == 1)]
            tn_indices = valid_indices[(label_targets == 0) & (label_predictions == 0)]
            fp_indices = valid_indices[(label_targets == 0) & (label_predictions == 1)]
            fn_indices = valid_indices[(label_targets == 1) & (label_predictions == 0)]
            
            # Store examples with confidence scores
            for category, indices in [('TP', tp_indices), ('TN', tn_indices), 
                                    ('FP', fp_indices), ('FN', fn_indices)]:
                if len(indices) == 0:
                    continue
                    
                # Sample random examples
                sample_size = min(len(indices), max_per_category)
                sampled_indices = np.random.choice(indices, sample_size, replace=False)
                
                for idx in sampled_indices:
                    self.examples[category].append({
                        'sample_idx': idx,
                        'label_idx': label_idx,
                        'label_name': label,
                        'probability': label_probabilities[np.where(valid_indices == idx)[0][0]],
                        'target': label_targets[np.where(valid_indices == idx)[0][0]],
                        'prediction': label_predictions[np.where(valid_indices == idx)[0][0]]
                    })
        
        # Print summary
        for category in ['TP', 'TN', 'FP', 'FN']:
            print(f"  {category}: {len(self.examples[category])} examples collected")
    
    def select_representative_examples(self, n_per_category=3):
        """
        Select 3 representative examples per category as specified in proposal
        
        Args:
            n_per_category: Number of examples per category (3 from proposal)
            
        Returns:
            Dictionary with selected examples
        """
        print(f"Selecting {n_per_category} representative examples per category...")
        
        selected_examples = {}
        
        for category in ['TP', 'TN', 'FP', 'FN']:
            if len(self.examples[category]) == 0:
                print(f"Warning: No {category} examples available")
                selected_examples[category] = []
                continue
            
            # Strategy: Select examples with diverse confidence scores
            examples = self.examples[category]
            
            if len(examples) <= n_per_category:
                selected_examples[category] = examples
            else:
                # Sort by probability and select diverse range
                sorted_examples = sorted(examples, key=lambda x: x['probability'])
                
                if n_per_category == 3:
                    # Select low, medium, high confidence
                    indices = [0, len(sorted_examples)//2, -1]
                else:
                    # Evenly spaced selection
                    indices = np.linspace(0, len(sorted_examples)-1, n_per_category, dtype=int)
                
                selected_examples[category] = [sorted_examples[i] for i in indices]
        
        return selected_examples
    
    def generate_gradcam_overlay(self, target_data_index, label_idx, save_path=None):
        """
        Generate Grad-CAM overlay for a specific example
        
        Args:
            target_data_index: Index in the dataset
            label_idx: Index of the label to analyze
            save_path: Optional path to save the overlay
            
        Returns:
            (original_image, heatmap, overlay) as numpy arrays
        """
        self.model.eval()
        
        # Create a single-sample dataset to avoid multiprocessing issues
        dataset = self.data_loader.dataset
        image, label, mask = dataset[target_data_index]
        
        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(self.device)
        
        # Generate CAM
        cam = self.gradcam.generate_cam(image, label_idx)
        
        # Convert to numpy for visualization
        original = image.squeeze().cpu().numpy()
        
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (original.shape[1], original.shape[0]))
        
        # Create overlay
        overlay = self.gradcam.overlay_heatmap(original, cam_resized)
        
        if save_path:
            # Save individual components
            save_dir = os.path.dirname(save_path)
            base_name = os.path.splitext(os.path.basename(save_path))[0]
            
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(original, cmap='gray')
            plt.title('Original')
            plt.axis('off')
            
            # Heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title('Grad-CAM')
            plt.axis('off')
            
            # Overlay
            plt.subplot(1, 3, 3)
            plt.imshow(overlay)
            plt.title('Overlay')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return original, cam_resized, overlay
    
    def generate_required_overlays(self, selected_examples, save_dir):
        """
        Generate the 12 Grad-CAM overlays required by the proposal
        (3 TP, 3 TN, 3 FP, 3 FN)
        
        Args:
            selected_examples: Dictionary with selected examples
            save_dir: Directory to save overlays
            
        Returns:
            Dictionary with generated overlays
        """
        print("\nGenerating 12 Grad-CAM overlays as specified in proposal...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        overlays = {}
        
        for category in ['TP', 'TN', 'FP', 'FN']:
            category_overlays = []
            
            for i, example in enumerate(selected_examples[category]):
                if i >= 3:  # Only 3 per category as specified
                    break
                    
                sample_idx = example['sample_idx']
                label_idx = example['label_idx']
                label_name = example['label_name']
                probability = example['probability']
                
                # Generate overlay
                save_path = save_dir / f"{category}_{i+1}_{label_name}_prob{probability:.3f}.png"
                
                try:
                    original, heatmap, overlay = self.generate_gradcam_overlay(
                        sample_idx, label_idx, save_path
                    )
                    
                    category_overlays.append({
                        'original_image': original,
                        'heatmap': heatmap,
                        'overlay': overlay,
                        'example_info': example,
                        'save_path': save_path
                    })
                    
                    print(f"  ✅ {category} {i+1}: {label_name} (prob: {probability:.3f})")
                    
                except Exception as e:
                    print(f"  ❌ Failed to generate {category} {i+1}: {e}")
            
            overlays[category] = category_overlays
        
        # Create summary figure
        self._create_summary_figure(overlays, save_dir / "gradcam_summary.png")
        
        return overlays
    
    def _create_summary_figure(self, overlays, save_path):
        """Create a summary figure with all 12 overlays"""
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        
        categories = ['TP', 'TN', 'FP', 'FN']
        category_names = {
            'TP': 'True Positives',
            'TN': 'True Negatives', 
            'FP': 'False Positives',
            'FN': 'False Negatives'
        }
        
        for row, category in enumerate(categories):
            for col in range(3):
                ax = axes[row, col]
                
                if col < len(overlays[category]):
                    overlay_data = overlays[category][col]
                    overlay = overlay_data['overlay']
                    example = overlay_data['example_info']
                    
                    ax.imshow(overlay)
                    ax.set_title(f"{category_names[category]} {col+1}\n"
                               f"{example['label_name']}\n"
                               f"Prob: {example['probability']:.3f}")
                else:
                    ax.text(0.5, 0.5, 'No example\navailable', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{category_names[category]} {col+1}")
                
                ax.axis('off')
        
        plt.suptitle('DiagXNet-Lite: Grad-CAM Analysis\n12 Representative Examples', 
                     fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📋 Summary figure saved: {save_path}")
    
    def generate_analysis_report(self, overlays, save_path):
        """
        Generate analysis report for Grad-CAM results
        
        Args:
            overlays: Dictionary with generated overlays
            save_path: Path to save report
        """
        report_lines = [
            "# DiagXNet-Lite: Grad-CAM Analysis Report",
            "",
            "## Overview",
            "This report presents Grad-CAM visualizations for DiagXNet-Lite following project proposal specifications.",
            "Generated 12 overlays: 3 True Positives (TP), 3 True Negatives (TN), 3 False Positives (FP), 3 False Negatives (FN).",
            "",
            "## Analysis of Sensible Image Areas",
            ""
        ]
        
        categories = ['TP', 'TN', 'FP', 'FN']
        category_descriptions = {
            'TP': 'True Positives - Model correctly identified pathology',
            'TN': 'True Negatives - Model correctly identified absence of pathology',
            'FP': 'False Positives - Model incorrectly identified pathology',
            'FN': 'False Negatives - Model missed actual pathology'
        }
        
        for category in categories:
            report_lines.append(f"### {category_descriptions[category]}")
            report_lines.append("")
            
            for i, overlay_data in enumerate(overlays[category]):
                example = overlay_data['example_info']
                report_lines.extend([
                    f"**{category} {i+1}: {example['label_name']}**",
                    f"- Probability: {example['probability']:.3f}",
                    f"- Target: {example['target']}, Prediction: {example['prediction']}",
                    f"- Interpretation: [Manual analysis of attention areas would go here]",
                    ""
                ])
        
        report_lines.extend([
            "## Key Findings",
            "1. **True Positives**: Grad-CAM should highlight clinically relevant areas",
            "2. **True Negatives**: Minimal activation in pathological regions", 
            "3. **False Positives**: May show spurious activations in normal regions",
            "4. **False Negatives**: May miss or weakly activate in pathological areas",
            "",
            "## Clinical Relevance",
            "The Grad-CAM visualizations provide interpretability for the DenseNet-121 model's",
            "decision-making process, which is crucial for clinical acceptance and trust.",
            "",
            "## Recommendations",
            "1. Review false predictions for model improvement opportunities",
            "2. Validate attention patterns with clinical experts",
            "3. Consider attention mechanisms in future model architectures"
        ])
        
        # Save report
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Analysis report saved: {save_path}")
    
    def run_complete_gradcam_analysis(self, probabilities, targets, masks, thresholds, save_dir):
        """
        Run complete Grad-CAM analysis as specified in proposal
        
        Args:
            probabilities: Model predictions
            targets: True labels
            masks: Valid label masks
            thresholds: Optimal thresholds
            save_dir: Directory to save all results
            
        Returns:
            Dictionary with complete analysis results
        """
        print("\n" + "="*60)
        print("DiagXNet-Lite: Grad-CAM Analysis")
        print("Generating 12 overlays (3 TP, 3 TN, 3 FP, 3 FN)")
        print("="*60)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Step 1: Collect examples
        self.collect_examples(probabilities, targets, masks, thresholds)
        
        # Step 2: Select representative examples
        selected_examples = self.select_representative_examples(n_per_category=3)
        
        # Step 3: Generate overlays
        overlays = self.generate_required_overlays(selected_examples, save_dir)
        
        # Step 4: Generate analysis report
        self.generate_analysis_report(overlays, save_dir / "gradcam_analysis_report.md")
        
        # Summary
        total_overlays = sum(len(overlays[cat]) for cat in overlays)
        print(f"\n✅ Grad-CAM analysis completed!")
        print(f"📊 Generated {total_overlays} overlays")
        print(f"💾 Results saved in: {save_dir}")
        print("="*60)
        
        return {
            'selected_examples': selected_examples,
            'overlays': overlays,
            'save_dir': save_dir
        }


def main():
    """Test the Grad-CAM analyzer"""
    print("DiagXNet-Lite Grad-CAM Analysis Module")
    print("Generates 12 overlays as specified in project proposal")


if __name__ == "__main__":
    main()
