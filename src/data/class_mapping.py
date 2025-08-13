"""
Class mapping utilities - Do one thing: unified class taxonomy mapping
Unix philosophy: Single responsibility for class standardization across datasets
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd

from .loader import ImageDataset
from ..core.classes import create_unified_classes, get_valid_classes
from .weights import calculate_class_weights

logger = logging.getLogger(__name__)


def filter_valid_classes(metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata to only include valid unified classes."""
    valid_classes = get_valid_classes()
    return metadata[metadata['class_name'].isin(valid_classes)].copy()


def map_to_unified_classes(metadata: pd.DataFrame) -> pd.DataFrame:
    """Map class names to unified class indices."""
    unified_classes = create_unified_classes()
    
    # Filter to valid classes first
    filtered_df = filter_valid_classes(metadata)
    
    # Map to unified indices
    filtered_df['unified_class_idx'] = filtered_df['class_name'].map(unified_classes)
    
    # Remove any that didn't map
    mapped_df = filtered_df.dropna(subset=['unified_class_idx'])
    mapped_df['unified_class_idx'] = mapped_df['unified_class_idx'].astype(int)
    
    return mapped_df


class UnifiedDataset(Dataset):
    """
    Dataset that maps classes to unified taxonomy.
    Single responsibility: unified class mapping.
    """
    
    def __init__(self, base_dataset: ImageDataset):
        """Wrap base dataset with unified class mapping."""
        self.base_dataset = base_dataset
        
        # Map metadata to unified classes
        self.unified_metadata = map_to_unified_classes(base_dataset.metadata)
        
        # Create index mapping from original to filtered indices
        self.valid_indices = self.unified_metadata.index.tolist()
        
        logger.info(f"Unified dataset: {len(self.valid_indices)} valid samples")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        # Map to original dataset index
        original_idx = self.valid_indices[idx]
        image, _ = self.base_dataset[original_idx]  # Ignore original label
        
        # Get unified label
        unified_label = self.unified_metadata.iloc[idx]['unified_class_idx']
        
        return image, unified_label
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of unified classes."""
        return self.unified_metadata['unified_class_idx'].value_counts().to_dict()