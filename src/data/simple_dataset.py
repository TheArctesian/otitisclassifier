"""
Simple Otoscopic Dataset for Testing and Debugging

A simplified dataset class for initial model training and testing before
scaling to the full multi-dataset approach. Focuses on ease of use and debugging.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter

logger = logging.getLogger(__name__)


class SimpleOtoscopicDataset(Dataset):
    """
    Simplified dataset for otoscopic image classification.
    
    Loads from a single processed directory with a metadata CSV file.
    Designed for initial testing and debugging before using the full
    multi-dataset loader.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_path: Optional[Union[str, Path]] = None,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 384,
        create_csv: bool = True
    ):
        """
        Initialize SimpleOtoscopicDataset.
        
        Args:
            data_dir: Path to processed data directory with class subdirectories
            csv_path: Path to metadata CSV file (will be created if not exists)
            transform: Additional transforms to apply (ImageNet norm applied automatically)
            image_size: Target image size for resizing
            create_csv: Whether to automatically create CSV if it doesn't exist
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Set up CSV path
        if csv_path is None:
            csv_path = self.data_dir / "metadata.csv"
        self.csv_path = Path(csv_path)
        
        # Create CSV if it doesn't exist and create_csv is True
        if not self.csv_path.exists() and create_csv:
            logger.info(f"Creating metadata CSV at {self.csv_path}")
            create_metadata_csv(self.data_dir, self.csv_path)
        
        # Load metadata
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.csv_path}")
        
        self.metadata = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.metadata)} samples from {self.csv_path}")
        
        # Set up transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.additional_transform = transform
        
        # Create class mapping
        self.class_names = sorted(self.metadata['class_name'].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, class_idx)
        """
        if idx >= len(self.metadata):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.metadata)}")
        
        # Get sample info
        sample = self.metadata.iloc[idx]
        image_path = self.data_dir / sample['image_path']
        class_idx = int(sample['class_idx'])
        
        try:
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            
            # Apply base transforms (resize, normalize)
            image = self.base_transform(image)
            
            # Apply additional transforms if provided
            if self.additional_transform is not None:
                image = self.additional_transform(image)
            
            return image, class_idx
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return fallback black image
            fallback_image = torch.zeros((3, self.image_size, self.image_size))
            fallback_image = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(fallback_image)
            return fallback_image, class_idx
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of samples across classes.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        distribution = self.metadata['class_name'].value_counts().to_dict()
        return dict(sorted(distribution.items()))
    
    def get_sample_images(self, n: int = 5, random_seed: int = 42) -> List[Dict]:
        """
        Get sample images for visualization.
        
        Args:
            n: Number of sample images to return
            random_seed: Random seed for reproducible sampling
            
        Returns:
            List of dictionaries with image info
        """
        random.seed(random_seed)
        
        # Try to get at least one sample from each class
        samples = []
        sampled_indices = set()
        
        # First, get one sample per class if possible
        for class_name in self.class_names:
            class_samples = self.metadata[self.metadata['class_name'] == class_name]
            if len(class_samples) > 0:
                idx = random.choice(class_samples.index.tolist())
                if idx not in sampled_indices:
                    sample = self.metadata.iloc[idx]
                    samples.append({
                        'index': idx,
                        'image_path': sample['image_path'],
                        'class_name': sample['class_name'],
                        'class_idx': sample['class_idx']
                    })
                    sampled_indices.add(idx)
                    
                    if len(samples) >= n:
                        break
        
        # Fill remaining slots with random samples
        remaining = n - len(samples)
        if remaining > 0:
            available_indices = [i for i in range(len(self.metadata)) if i not in sampled_indices]
            additional_indices = random.sample(available_indices, min(remaining, len(available_indices)))
            
            for idx in additional_indices:
                sample = self.metadata.iloc[idx]
                samples.append({
                    'index': idx,
                    'image_path': sample['image_path'],
                    'class_name': sample['class_name'],
                    'class_idx': sample['class_idx']
                })
        
        return samples
    
    def validate_images(self) -> Dict[str, Union[int, List[str]]]:
        """
        Validate that all images can be loaded.
        
        Returns:
            Dictionary with validation results
        """
        valid_count = 0
        invalid_images = []
        
        logger.info("Validating all images...")
        
        for idx, sample in self.metadata.iterrows():
            image_path = self.data_dir / sample['image_path']
            
            try:
                # Try to load the image
                with Image.open(image_path) as img:
                    # Check basic properties
                    if img.size[0] > 0 and img.size[1] > 0:
                        valid_count += 1
                    else:
                        invalid_images.append(f"{image_path}: Invalid dimensions {img.size}")
                        
            except Exception as e:
                invalid_images.append(f"{image_path}: {str(e)}")
        
        results = {
            'total_images': len(self.metadata),
            'valid_images': valid_count,
            'invalid_images': len(invalid_images),
            'invalid_paths': invalid_images,
            'success_rate': valid_count / len(self.metadata) if len(self.metadata) > 0 else 0.0
        }
        
        logger.info(f"Validation complete: {valid_count}/{len(self.metadata)} images valid "
                   f"({results['success_rate']:.2%} success rate)")
        
        if invalid_images:
            logger.warning(f"Found {len(invalid_images)} invalid images")
            for invalid in invalid_images[:5]:  # Show first 5 errors
                logger.warning(f"  {invalid}")
            if len(invalid_images) > 5:
                logger.warning(f"  ... and {len(invalid_images) - 5} more")
        
        return results
    
    def get_class_weights(self, method: str = 'inverse') -> torch.Tensor:
        """
        Calculate class weights for handling class imbalance.
        
        Args:
            method: Weight calculation method ('inverse', 'sqrt', 'log')
            
        Returns:
            Tensor of class weights
        """
        class_counts = self.metadata['class_idx'].value_counts().sort_index()
        total_samples = len(self.metadata)
        num_classes = len(self.class_names)
        
        weights = torch.zeros(num_classes)
        
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            
            if method == 'inverse':
                weight = total_samples / (num_classes * count)
            elif method == 'sqrt':
                weight = np.sqrt(total_samples / (num_classes * count))
            elif method == 'log':
                weight = np.log(total_samples / (num_classes * count) + 1)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            weights[class_idx] = weight
        
        logger.info(f"Calculated class weights using {method} method")
        return weights


def create_metadata_csv(data_dir: Union[str, Path], csv_path: Union[str, Path]) -> None:
    """
    Create metadata CSV by scanning processed directory structure.
    
    Expected directory structure:
    data_dir/
    ├── class1/
    │   ├── image1.png
    │   └── image2.png
    └── class2/
        ├── image3.png
        └── image4.png
    
    Args:
        data_dir: Path to processed data directory
        csv_path: Output path for metadata CSV
    """
    data_dir = Path(data_dir)
    csv_path = Path(csv_path)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Standard class name mappings for otitis classification
    class_mappings = {
        'normal': 'Normal_Tympanic_Membrane',
        'aom': 'Acute_Otitis_Media', 
        'chronic': 'Chronic_Suppurative_Otitis_Media',
        'chornic': 'Chronic_Suppurative_Otitis_Media',  # Handle typo
        'earwax': 'Earwax_Cerumen_Impaction',
        'buson': 'Earwax_Cerumen_Impaction',  # Alternative name
        'earventulation': 'Ear_Ventilation_Tube',
        'chrneftup': 'Ear_Ventilation_Tube',  # Alternative name
        'foreign': 'Foreign_Bodies',
        'yabancisim': 'Foreign_Bodies',  # Alternative name
        'otitis_externa': 'Otitis_Externa',
        'tympanoskleros': 'Tympanoskleros_Myringosclerosis',
        'pseudo_membranes': 'Pseudo_Membranes'
    }
    
    samples = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    
    logger.info(f"Scanning directory: {data_dir}")
    
    # Scan class directories
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_folder_name = class_dir.name.lower()
        
        # Map folder name to standard class name
        class_name = class_mappings.get(class_folder_name, class_dir.name)
        
        logger.info(f"Processing class directory: {class_dir.name} -> {class_name}")
        
        # Find all image files
        image_count = 0
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() in valid_extensions:
                # Create relative path from data_dir
                relative_path = image_path.relative_to(data_dir)
                
                samples.append({
                    'image_path': str(relative_path).replace('\\', '/'),  # Use forward slashes
                    'class_name': class_name,
                    'original_folder': class_dir.name
                })
                image_count += 1
        
        logger.info(f"  Found {image_count} images in {class_dir.name}")
    
    if not samples:
        raise ValueError(f"No valid images found in {data_dir}")
    
    # Convert to DataFrame and add class indices
    df = pd.DataFrame(samples)
    
    # Create class index mapping
    unique_classes = sorted(df['class_name'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    
    # Add class indices
    df['class_idx'] = df['class_name'].map(class_to_idx)
    
    # Sort by class name and image path for consistent ordering
    df = df.sort_values(['class_name', 'image_path']).reset_index(drop=True)
    
    # Save to CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    # Log summary
    class_distribution = df['class_name'].value_counts()
    logger.info(f"Created metadata CSV with {len(df)} samples")
    logger.info(f"Found {len(unique_classes)} classes:")
    for class_name in unique_classes:
        count = class_distribution.get(class_name, 0)
        idx = class_to_idx[class_name]
        logger.info(f"  {idx}: {class_name} ({count} samples)")
    
    logger.info(f"Saved metadata to: {csv_path}")


def create_simple_dataloader(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    csv_path: Optional[Union[str, Path]] = None,
    train_transforms: bool = True,
    image_size: int = 384
) -> DataLoader:
    """
    Factory function to create a simple DataLoader.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for loading
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        csv_path: Path to metadata CSV (auto-created if None)
        train_transforms: Whether to apply training augmentations
        image_size: Target image size
        
    Returns:
        PyTorch DataLoader
    """
    # Set up training augmentations
    additional_transforms = None
    if train_transforms:
        additional_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ])
    
    # Create dataset
    dataset = SimpleOtoscopicDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        transform=additional_transforms,
        image_size=image_size
    )
    
    logger.info(f"Created simple dataset with {len(dataset)} samples")
    logger.info(f"Class distribution: {dataset.get_class_distribution()}")
    
    # Create DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )


# Utility functions

def print_dataset_summary(dataset: SimpleOtoscopicDataset) -> None:
    """Print a summary of the dataset."""
    print(f"\n{'='*50}")
    print(f"Simple Otoscopic Dataset Summary")
    print(f"{'='*50}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_names)}")
    print(f"Image size: {dataset.image_size}x{dataset.image_size}")
    
    print(f"\nClass Distribution:")
    distribution = dataset.get_class_distribution()
    for class_name, count in distribution.items():
        percentage = count / len(dataset) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nClass Mappings:")
    for idx, class_name in dataset.idx_to_class.items():
        print(f"  {idx}: {class_name}")
    
    print(f"{'='*50}\n")


def validate_dataset_setup(data_dir: Union[str, Path]) -> bool:
    """
    Validate that the dataset directory is properly set up.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        True if valid, False otherwise
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False
    
    # Check for class subdirectories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.error(f"No class subdirectories found in {data_dir}")
        return False
    
    # Check for images in class directories
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    total_images = 0
    
    for class_dir in class_dirs:
        image_count = 0
        for file_path in class_dir.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                image_count += 1
        
        if image_count == 0:
            logger.warning(f"No images found in class directory: {class_dir}")
        else:
            logger.info(f"Found {image_count} images in {class_dir.name}")
            total_images += image_count
    
    if total_images == 0:
        logger.error("No valid images found in any class directory")
        return False
    
    logger.info(f"Validation passed: {len(class_dirs)} classes, {total_images} total images")
    return True