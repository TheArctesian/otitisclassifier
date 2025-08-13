"""
Simple data loader - Do one thing: load image data
Unix philosophy: Focused, composable data loading
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

from ..core.transforms import create_transform_pipeline
from .metadata import ensure_metadata_csv, load_metadata_csv

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    Simple image dataset following Unix philosophy.
    Does one thing: loads images and labels from CSV metadata.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        csv_path: Optional[Union[str, Path]] = None,
        image_size: int = 384,
        use_radimagenet: bool = False,
        training: bool = False
    ):
        """Initialize simple image dataset."""
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Ensure metadata exists
        self.csv_path = ensure_metadata_csv(data_dir, csv_path)
        
        # Load metadata
        self.metadata = load_metadata_csv(self.csv_path)
        
        # Create transform pipeline
        self.transform = create_transform_pipeline(
            image_size=image_size,
            use_radimagenet=use_radimagenet,
            training=training
        )
        
        # Create class mappings
        self.class_names = sorted(self.metadata['class_name'].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        logger.info(f"Loaded dataset: {len(self.metadata)} samples, {len(self.class_names)} classes")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get single sample. Core functionality."""
        if idx >= len(self.metadata):
            raise IndexError(f"Index {idx} out of range")
        
        # Get sample info
        sample = self.metadata.iloc[idx]
        image_path = self.data_dir / sample['image_path']
        class_idx = int(sample['class_idx'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            return image, class_idx
            
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            # Return fallback
            fallback = torch.zeros((3, self.image_size, self.image_size))
            return fallback, class_idx
    
    def get_class_distribution(self) -> dict:
        """Get class distribution. Simple utility."""
        return self.metadata['class_name'].value_counts().to_dict()
    
    def get_sample_paths(self, n: int = 5) -> list:
        """Get sample image paths. Simple utility."""
        sample_df = self.metadata.sample(min(n, len(self.metadata)))
        return [
            {
                'path': row['image_path'],
                'class': row['class_name'],
                'idx': row['class_idx']
            }
            for _, row in sample_df.iterrows()
        ]


def create_simple_dataset(
    data_dir: Union[str, Path],
    image_size: int = 384,
    use_radimagenet: bool = False,
    training: bool = False
) -> ImageDataset:
    """Factory function for simple dataset creation."""
    return ImageDataset(
        data_dir=data_dir,
        image_size=image_size,
        use_radimagenet=use_radimagenet,
        training=training
    )