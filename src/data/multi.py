"""
Multi-dataset composition - Do one thing: combine datasets
Unix philosophy: Simple composition of single-purpose components
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


def create_multi_dataset(
    dataset_dirs: Dict[str, Union[str, Path]],
    image_size: int = 384,
    use_radimagenet: bool = False,
    training: bool = False
) -> ConcatDataset:
    """
    Create multi-dataset by composing individual datasets.
    Unix philosophy: compose simple parts into complex whole.
    """
    datasets = []
    
    for dataset_name, data_dir in dataset_dirs.items():
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Create base dataset
        base_dataset = ImageDataset(
            data_dir=data_dir,
            image_size=image_size,
            use_radimagenet=use_radimagenet,
            training=training
        )
        
        # Wrap with unified class mapping
        unified_dataset = UnifiedDataset(base_dataset)
        datasets.append(unified_dataset)
    
    # Combine all datasets
    combined = ConcatDataset(datasets)
    
    logger.info(f"Created multi-dataset with {len(combined)} total samples")
    return combined


def get_multi_dataset_weights(multi_dataset: ConcatDataset) -> torch.Tensor:
    """Calculate weights for multi-dataset."""
    all_labels = []
    
    # Collect all labels
    for dataset in multi_dataset.datasets:
        if hasattr(dataset, 'get_class_distribution'):
            dist = dataset.get_class_distribution()
            for class_idx, count in dist.items():
                all_labels.extend([class_idx] * count)
    
    # Count occurrences
    from collections import Counter
    class_counts = dict(Counter(all_labels))
    
    return calculate_class_weights(class_counts)


# Predefined dataset configurations
DATASET_CONFIGS = {
    'processed': {
        'ebasaran': 'data/processed/ebasaran-kaggale',
        'uci': 'data/processed/uci-kaggle',
        'vanak': 'data/processed/vanak-figshare'
    },
    'raw': {
        'ebasaran': 'data/raw/ebasaran-kaggale',
        'uci': 'data/raw/uci-kaggle', 
        'vanak': 'data/raw/vanak-figshare'
    }
}


def create_standard_multi_dataset(
    config: str = 'processed',
    datasets: List[str] = None,
    **kwargs
) -> ConcatDataset:
    """Create multi-dataset using standard configurations."""
    if config not in DATASET_CONFIGS:
        raise ValueError(f"Unknown config: {config}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config_datasets = DATASET_CONFIGS[config]
    
    if datasets is None:
        # Use all datasets in config
        selected_datasets = config_datasets
    else:
        # Use only specified datasets
        selected_datasets = {k: v for k, v in config_datasets.items() if k in datasets}
    
    return create_multi_dataset(selected_datasets, **kwargs)