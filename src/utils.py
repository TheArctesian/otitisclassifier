"""
Utility functions following Unix philosophy.
Each function does one thing well and composes with others.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration. Single responsibility."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(levelname)s: %(message)s'
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader with sensible defaults. Simple factory function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # Drop last batch only when shuffling (training)
        persistent_workers=num_workers > 0
    )


def print_dataset_info(dataset, name: str = "Dataset") -> None:
    """Print dataset information. Simple utility."""
    print(f"\n{name} Information:")
    print(f"  Total samples: {len(dataset)}")
    
    if hasattr(dataset, 'get_class_distribution'):
        dist = dataset.get_class_distribution()
        print(f"  Number of classes: {len(dist)}")
        print("  Class distribution:")
        for class_name, count in dist.items():
            percentage = count / len(dataset) * 100
            print(f"    {class_name}: {count} ({percentage:.1f}%)")


def validate_paths(paths: List[Union[str, Path]]) -> Dict[str, List[str]]:
    """Validate list of paths. Returns existing and missing."""
    existing = []
    missing = []
    
    for path in paths:
        path = Path(path)
        if path.exists():
            existing.append(str(path))
        else:
            missing.append(str(path))
    
    return {'existing': existing, 'missing': missing}


def ensure_directories(dirs: List[Union[str, Path]]) -> None:
    """Ensure directories exist. Simple utility."""
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)