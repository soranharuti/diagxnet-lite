"""
CheXpert Dataset Implementation for DiagXNet-Lite
Handles loading, preprocessing, and uncertainty label management
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CheXpertDataset(Dataset):
    """
    CheXpert Dataset for chest X-ray classification
    
    Args:
        csv_path (str): Path to the CSV file with labels
        data_root (str): Root directory containing the images
        transform (callable, optional): Transform to apply to images
        uncertainty_policy (str): How to handle uncertain labels (-1)
            - 'ignore': Ignore uncertain labels (mask them)
            - 'positive': Treat uncertain as positive (1)
            - 'negative': Treat uncertain as negative (0)
            - 'uignore': U-Ignore policy from CheXpert paper
            - 'uzeros': U-Zeros policy from CheXpert paper
        frontal_only (bool): Whether to use only frontal images
    """
    
    def __init__(self, csv_path, data_root, transform=None, uncertainty_policy='ignore', frontal_only=True):
        self.csv_path = csv_path
        self.data_root = Path(data_root)
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy
        self.frontal_only = frontal_only
        
        # Load and process the CSV
        self.df = pd.read_csv(csv_path)
        
        # Resolve the actual CheXpert images root under data_root
        # Supports both "chexpert" and "CheXpert-v1.0-small" layouts
        chexpert_lower = self.data_root / "chexpert"
        chexpert_small = self.data_root / "CheXpert-v1.0-small"
        if chexpert_lower.is_dir():
            self.images_root = chexpert_lower
        elif chexpert_small.is_dir():
            self.images_root = chexpert_small
        else:
            # Fallback: assume data_root itself contains train/ and valid/
            self.images_root = self.data_root
        
        # Filter for frontal images if requested
        if frontal_only:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Get the label columns (exclude metadata columns)
        self.label_columns = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Uncertainty policy: {uncertainty_policy}")
        print(f"Frontal only: {frontal_only}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image path - handle absolute and multiple relative layouts
        rel_path_str = str(row['Path']).lstrip('/').replace('\\', '/')
        # Normalize any leading dataset folder to be relative to images_root
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
            # Standard CheXpert CSV uses train/ or valid/
            image_path = self.images_root / rel_path
        elif rel_path_str.startswith('chexpert/') or rel_path_str.startswith('CheXpert-v1.0-small/'):
            # Path already includes the dataset folder name relative to data_root
            image_path = self.data_root / rel_path
        else:
            # Fallback: treat as relative to resolved images_root
            image_path = self.images_root / rel_path
        
        # Load image
        try:
            image = Image.open(image_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('L', (224, 224), 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = row[self.label_columns].values.astype(np.float32)
        
        # Handle uncertainty labels based on policy
        if self.uncertainty_policy == 'ignore':
            # Convert uncertain labels to NaN and create mask
            labels[labels == -1] = float('nan')
            masks = ~np.isnan(labels)
            labels = np.nan_to_num(labels, nan=0.0)
        elif self.uncertainty_policy == 'positive':
            labels[labels == -1] = 1
            masks = np.ones_like(labels, dtype=bool)
        elif self.uncertainty_policy == 'negative':
            labels[labels == -1] = 0
            masks = np.ones_like(labels, dtype=bool)
        elif self.uncertainty_policy == 'uignore':
            # U-Ignore: treat uncertain as negative for training
            labels[labels == -1] = 0
            masks = np.ones_like(labels, dtype=bool)
        elif self.uncertainty_policy == 'uzeros':
            # U-Zeros: treat uncertain as negative
            labels[labels == -1] = 0
            masks = np.ones_like(labels, dtype=bool)
        else:
            # Default: ignore uncertain labels
            labels[labels == -1] = float('nan')
            masks = ~np.isnan(labels)
            labels = np.nan_to_num(labels, nan=0.0)
        
        return image, torch.tensor(labels, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool)

def get_train_transforms():
    """Get training transforms with augmentation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])

def get_val_transforms():
    """Get validation transforms without augmentation"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])

def create_data_loaders(train_csv, batch_size=32, num_workers=4, uncertainty_policy='ignore', 
                       train_split=0.8, val_split=0.2, random_seed=42):
    """
    Create train and validation data loaders
    
    Args:
        train_csv (str): Path to training CSV file
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes
        uncertainty_policy (str): How to handle uncertain labels
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        random_seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create full dataset
    full_dataset = CheXpertDataset(
        csv_path=train_csv,
        data_root="data",  # Relative to project root
        transform=None,  # We'll apply transforms in the loaders
        uncertainty_policy=uncertainty_policy,
        frontal_only=False  # Use all views for more data
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Create indices for splitting
    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    
    return train_loader, val_loader

def get_class_weights(dataset):
    """
    Calculate class weights for handling class imbalance
    
    Args:
        dataset: CheXpertDataset instance
    
    Returns:
        torch.Tensor: Class weights for each label
    """
    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        _, labels, masks = dataset[i]
        valid_labels = labels[masks]
        all_labels.append(valid_labels)
    
    # Concatenate all valid labels
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate class weights (inverse frequency)
    class_counts = torch.sum(all_labels, dim=0)
    total_samples = len(all_labels)
    
    # Avoid division by zero
    class_weights = total_samples / (class_counts + 1e-6)
    
    return class_weights

if __name__ == "__main__":
    # Test the dataset
    print("Testing CheXpertDataset...")
    
    # Create a test dataset
    test_dataset = CheXpertDataset(
        csv_path="data/chexpert_small/CheXpert-v1.0-small/train/train.csv",
        data_root="data",
        transform=get_val_transforms(),
        uncertainty_policy='ignore',
        frontal_only=False
    )
    
    print(f"Dataset size: {len(test_dataset)}")
    
    if len(test_dataset) > 0:
        # Test getting a sample
        image, labels, masks = test_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Masks shape: {masks.shape}")
        print(f"Sample labels: {labels}")
        print(f"Sample masks: {masks}")
    else:
        print("No samples found in dataset")
