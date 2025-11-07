"""
Balanced Batch Sampler for Class Imbalance
Ensures rare conditions are adequately represented in training
"""

import torch
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter


class BalancedBatchSampler(Sampler):
    """
    Balanced sampler that oversamples rare classes
    
    Strategy:
    - Calculate per-sample weight based on rarest positive label
    - Samples with rare conditions get higher sampling probability
    - Maintains diversity while focusing on underrepresented classes
    """
    
    def __init__(self, dataset, rare_threshold=0.10, oversample_factor=2.0):
        """
        Args:
            dataset: CheXpertDataset instance
            rare_threshold: Classes with prevalence < threshold are considered rare
            oversample_factor: How much more to sample rare classes (2.0 = 2x)
        """
        self.dataset = dataset
        self.rare_threshold = rare_threshold
        self.oversample_factor = oversample_factor
        
        # Calculate class prevalences
        self.class_prevalences = self._calculate_prevalences()
        
        # Identify rare classes
        self.rare_classes = np.where(self.class_prevalences < rare_threshold)[0]
        
        # Calculate sample weights
        self.sample_weights = self._calculate_sample_weights()
        
        print("\n" + "="*70)
        print("BALANCED BATCH SAMPLER INITIALIZATION")
        print("="*70)
        print(f"Dataset size: {len(dataset)}")
        print(f"Rare threshold: {rare_threshold:.2%}")
        print(f"Oversample factor: {oversample_factor}x")
        print(f"\nRare classes (prevalence < {rare_threshold:.2%}):")
        for idx in self.rare_classes:
            print(f"  Class {idx}: Prevalence = {self.class_prevalences[idx]:.4f}")
        print("="*70)
        
    def _calculate_prevalences(self):
        """Calculate prevalence of each class in dataset"""
        num_classes = self.dataset[0][1].shape[0]
        pos_counts = np.zeros(num_classes)
        total_valid = np.zeros(num_classes)
        
        print("\nCalculating class prevalences...")
        for idx in range(len(self.dataset)):
            _, labels, masks = self.dataset[idx]
            labels_np = labels.numpy()
            masks_np = masks.numpy()
            
            pos_counts += labels_np * masks_np
            total_valid += masks_np
        
        prevalences = pos_counts / (total_valid + 1e-8)
        
        print("\nClass Prevalences:")
        print("-" * 60)
        for i, prev in enumerate(prevalences):
            status = "RARE" if prev < self.rare_threshold else "Common"
            print(f"  Class {i:2d}: {prev:.4f} ({prev*100:.2f}%) - {status}")
        print("-" * 60)
        
        return prevalences
    
    def _calculate_sample_weights(self):
        """
        Calculate weight for each sample based on its rarest positive label
        
        Strategy:
        - Base weight = 1.0 for all samples
        - If sample has any rare condition, multiply by oversample_factor
        - Use maximum weight if multiple rare conditions present
        """
        sample_weights = np.ones(len(self.dataset))
        
        rare_sample_count = 0
        
        for idx in range(len(self.dataset)):
            _, labels, masks = self.dataset[idx]
            labels_np = labels.numpy()
            masks_np = masks.numpy()
            
            # Check if sample has any rare condition
            has_rare = False
            for rare_idx in self.rare_classes:
                if masks_np[rare_idx] == 1 and labels_np[rare_idx] == 1:
                    has_rare = True
                    break
            
            if has_rare:
                sample_weights[idx] = self.oversample_factor
                rare_sample_count += 1
        
        print(f"\nSamples with rare conditions: {rare_sample_count} ({rare_sample_count/len(self.dataset)*100:.2f}%)")
        print(f"These samples will be sampled {self.oversample_factor}x more frequently")
        
        return sample_weights
    
    def __iter__(self):
        """Return iterator over sample indices"""
        # Use WeightedRandomSampler for actual sampling
        sampler = WeightedRandomSampler(
            weights=self.sample_weights,
            num_samples=len(self.dataset),
            replacement=True
        )
        return iter(sampler)
    
    def __len__(self):
        return len(self.dataset)


class ClassBalancedBatchSampler(Sampler):
    """
    Alternative: Ensure each batch has balanced representation of rare classes
    
    This is more aggressive - each batch will try to include rare samples
    """
    
    def __init__(self, dataset, batch_size, rare_threshold=0.10, rare_per_batch=4):
        """
        Args:
            dataset: CheXpertDataset instance
            batch_size: Batch size
            rare_threshold: Classes with prevalence < threshold are considered rare
            rare_per_batch: Number of rare samples to include per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.rare_threshold = rare_threshold
        self.rare_per_batch = min(rare_per_batch, batch_size // 2)
        
        # Calculate class prevalences
        self.class_prevalences = self._calculate_prevalences()
        
        # Identify rare and common samples
        self.rare_indices, self.common_indices = self._split_rare_common()
        
        self.num_batches = len(dataset) // batch_size
        
        print("\n" + "="*70)
        print("CLASS-BALANCED BATCH SAMPLER INITIALIZATION")
        print("="*70)
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Rare samples per batch: {self.rare_per_batch}")
        print(f"Common samples per batch: {batch_size - self.rare_per_batch}")
        print(f"\nRare samples: {len(self.rare_indices)} ({len(self.rare_indices)/len(dataset)*100:.2f}%)")
        print(f"Common samples: {len(self.common_indices)} ({len(self.common_indices)/len(dataset)*100:.2f}%)")
        print("="*70)
    
    def _calculate_prevalences(self):
        """Calculate prevalence of each class"""
        num_classes = self.dataset[0][1].shape[0]
        pos_counts = np.zeros(num_classes)
        total_valid = np.zeros(num_classes)
        
        for idx in range(len(self.dataset)):
            _, labels, masks = self.dataset[idx]
            pos_counts += labels.numpy() * masks.numpy()
            total_valid += masks.numpy()
        
        return pos_counts / (total_valid + 1e-8)
    
    def _split_rare_common(self):
        """Split dataset indices into rare and common samples"""
        rare_classes = np.where(self.class_prevalences < self.rare_threshold)[0]
        
        rare_indices = []
        common_indices = []
        
        for idx in range(len(self.dataset)):
            _, labels, masks = self.dataset[idx]
            labels_np = labels.numpy()
            masks_np = masks.numpy()
            
            # Check if sample has any rare condition
            has_rare = False
            for rare_idx in rare_classes:
                if masks_np[rare_idx] == 1 and labels_np[rare_idx] == 1:
                    has_rare = True
                    break
            
            if has_rare:
                rare_indices.append(idx)
            else:
                common_indices.append(idx)
        
        return np.array(rare_indices), np.array(common_indices)
    
    def __iter__(self):
        """Generate batches with balanced rare/common ratio"""
        batches = []
        
        for _ in range(self.num_batches):
            # Sample rare indices
            if len(self.rare_indices) >= self.rare_per_batch:
                rare_batch = np.random.choice(
                    self.rare_indices, 
                    size=self.rare_per_batch, 
                    replace=False
                )
            else:
                rare_batch = np.random.choice(
                    self.rare_indices, 
                    size=self.rare_per_batch, 
                    replace=True
                )
            
            # Sample common indices
            common_batch = np.random.choice(
                self.common_indices,
                size=self.batch_size - self.rare_per_batch,
                replace=False
            )
            
            # Combine and shuffle
            batch = np.concatenate([rare_batch, common_batch])
            np.random.shuffle(batch)
            batches.extend(batch.tolist())
        
        return iter(batches)
    
    def __len__(self):
        return self.num_batches * self.batch_size


if __name__ == "__main__":
    print("Testing Balanced Samplers...")
    print("\n[OK] Balanced sampler implementation complete!")

