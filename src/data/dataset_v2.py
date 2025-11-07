"""
CheXpert Dataset Implementation V2 for DiagXNet-Lite
Improved version that excludes "Support Devices" and focuses on pathologies only
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CheXpertDatasetV2(Dataset):
    """
    CheXpert Dataset V2 - Excludes "Support Devices" label
    
    Changes from V1:
    - Removes "Support Devices" from labels (not a pathology)
    - Now has 13 labels instead of 14
    - Focuses purely on disease detection
    
    Args:
        csv_path (str): Path to the CSV file with labels
        data_root (str): Root directory containing the images
        transform (callable, optional): Transform to apply to images
        uncertainty_policy (str): How to handle uncertain labels (-1)
        frontal_only (bool): Whether to use only frontal images
    """
    
    def __init__(self, csv_path, data_root, transform=None, uncertainty_policy='ignore', frontal_only=True):
        self.csv_path = csv_path
        self.data_root = Path(data_root)
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy
        self.frontal_only = frontal_only
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Resolve images root
        chexpert_lower = self.data_root / "chexpert"
        chexpert_small = self.data_root / "CheXpert-v1.0-small"
        if chexpert_lower.is_dir():
            self.images_root = chexpert_lower
        elif chexpert_small.is_dir():
            self.images_root = chexpert_small
        else:
            self.images_root = self.data_root
        
        # Filter for frontal images if requested
        if frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        self.df = self.df.reset_index(drop=True)
        
        # V2: Exclude "Support Devices" from labels
        self.label_columns = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture'
        ]
        # NOTE: "Support Devices" removed - now 13 labels instead of 14
        
        print(f"\n" + "="*70)
        print("CHEXPERT DATASET V2 LOADED")
        print("="*70)
        print(f"Samples: {len(self.df)}")
        print(f"Labels: {len(self.label_columns)} (Support Devices excluded)")
        print(f"Uncertainty policy: {uncertainty_policy}")
        print(f"Frontal only: {frontal_only}")
        print("="*70)
        
        # Calculate and display label statistics
        self._print_label_statistics()
    
    def _print_label_statistics(self):
        """Print statistics about label distribution"""
        print("\nLabel Distribution:")
        print("-" * 70)
        
        for label in self.label_columns:
            if label in self.df.columns:
                # Count positive, negative, and uncertain
                positive = (self.df[label] == 1.0).sum()
                negative = (self.df[label] == 0.0).sum()
                uncertain = (self.df[label] == -1.0).sum()
                missing = self.df[label].isna().sum()
                total = len(self.df)
                
                prevalence = positive / total if total > 0 else 0
                status = "RARE" if prevalence < 0.10 else "Common"
                
                print(f"  {label:30s} Pos: {positive:5d} ({prevalence:6.2%}) - {status}")
        
        print("-" * 70)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image path
        rel_path_str = str(row['Path']).lstrip('/').replace('\\', '/')
        
        # Normalize dataset folder references
        if rel_path_str.startswith('CheXpert-v1.0-small/'):
            parts = rel_path_str.split('/', 1)
            if len(parts) > 1:
                rel_path_str = parts[1]
        elif rel_path_str.startswith('chexpert/'):
            parts = rel_path_str.split('/', 1)
            if len(parts) > 1:
                rel_path_str = parts[1]
        
        rel_path = Path(rel_path_str)
        
        if rel_path.is_absolute():
            image_path = rel_path
        elif rel_path_str.startswith('train/') or rel_path_str.startswith('valid/'):
            image_path = self.images_root / rel_path
        elif rel_path_str.startswith('chexpert/') or rel_path_str.startswith('CheXpert-v1.0-small/'):
            image_path = self.data_root / rel_path
        else:
            image_path = self.data_root / rel_path
        
        # Load image
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', (224, 224), color='black')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get labels (13 labels, no Support Devices)
        labels = []
        masks = []
        
        for label_col in self.label_columns:
            if label_col in row:
                val = row[label_col]
                
                # Handle NaN
                if pd.isna(val):
                    labels.append(0.0)
                    masks.append(0.0)  # Mask out
                    continue
                
                # Handle uncertainty based on policy
                if val == -1.0:  # Uncertain
                    if self.uncertainty_policy == 'ignore':
                        labels.append(0.0)
                        masks.append(0.0)  # Mask out
                    elif self.uncertainty_policy == 'positive':
                        labels.append(1.0)
                        masks.append(1.0)
                    elif self.uncertainty_policy == 'negative':
                        labels.append(0.0)
                        masks.append(1.0)
                    else:
                        labels.append(0.0)
                        masks.append(0.0)
                else:
                    labels.append(float(val))
                    masks.append(1.0)
            else:
                # Label not present in CSV
                labels.append(0.0)
                masks.append(0.0)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        masks = torch.tensor(masks, dtype=torch.float32)
        
        return image, labels, masks
    
    def get_class_distribution(self):
        """Return distribution of positive samples per class"""
        num_classes = len(self.label_columns)
        pos_counts = np.zeros(num_classes)
        total_valid = np.zeros(num_classes)
        
        for idx in range(len(self)):
            _, labels, masks = self[idx]
            pos_counts += labels.numpy() * masks.numpy()
            total_valid += masks.numpy()
        
        return pos_counts, total_valid


def get_train_transforms():
    """Training transforms with data augmentation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.CenterCrop(224),  # Remove ColorJitter for grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])


def get_val_transforms():
    """Validation transforms without augmentation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])


if __name__ == "__main__":
    # Test dataset v2
    print("Testing CheXpertDatasetV2...")
    
    from configs.config import DATA_ROOT, CHEXPERT_ROOT
    
    valid_csv = CHEXPERT_ROOT / "valid.csv"
    val_transform = get_val_transforms()
    
    dataset_v2 = CheXpertDatasetV2(
        valid_csv,
        DATA_ROOT,
        val_transform,
        uncertainty_policy='ignore',
        frontal_only=False
    )
    
    print(f"\n[OK] Dataset V2 loaded: {len(dataset_v2)} samples")
    print(f"[OK] Number of labels: {len(dataset_v2.label_columns)}")
    print(f"[OK] Labels: {dataset_v2.label_columns}")
    
    # Test getting a sample
    image, labels, masks = dataset_v2[0]
    print(f"\n[OK] Sample shape: {image.shape}")
    print(f"[OK] Labels shape: {labels.shape}")
    print(f"[OK] Masks shape: {masks.shape}")
    
    assert labels.shape[0] == 13, f"Expected 13 labels, got {labels.shape[0]}"
    assert masks.shape[0] == 13, f"Expected 13 masks, got {masks.shape[0]}"
    
    print("\n[OK] All tests passed!")

