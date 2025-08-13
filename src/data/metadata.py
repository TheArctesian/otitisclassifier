"""
Metadata utilities - Do one thing: handle CSV metadata
Unix philosophy: Simple CSV operations that compose well
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from ..core.paths import find_images, relative_path
from ..core.classes import normalize_class_name

logger = logging.getLogger(__name__)


def scan_directory_for_metadata(data_dir: Union[str, Path]) -> List[Dict[str, str]]:
    """Scan directory and create metadata entries. Single responsibility."""
    data_dir = Path(data_dir)
    samples = []
    
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return samples
    
    # Scan class directories
    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_folder_name = class_dir.name
        normalized_class_name = normalize_class_name(class_folder_name)
        
        # Find all images in this class directory
        images = find_images(class_dir)
        
        for image_path in images:
            relative_image_path = relative_path(image_path, data_dir)
            
            samples.append({
                'image_path': str(relative_image_path).replace('\\', '/'),
                'class_name': normalized_class_name,
                'original_folder': class_folder_name
            })
    
    return samples


def create_csv_from_samples(samples: List[Dict[str, str]], csv_path: Union[str, Path]) -> None:
    """Create CSV from sample list. Single purpose function."""
    if not samples:
        raise ValueError("No samples provided")
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Create class index mapping
    unique_classes = sorted(df['class_name'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
    
    # Add class indices
    df['class_idx'] = df['class_name'].map(class_to_idx)
    
    # Sort for consistency
    df = df.sort_values(['class_name', 'image_path']).reset_index(drop=True)
    
    # Save CSV
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Created CSV with {len(df)} samples at {csv_path}")


def load_metadata_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load metadata CSV. Simple function."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    return pd.read_csv(csv_path)


def ensure_metadata_csv(data_dir: Union[str, Path], 
                       csv_path: Union[str, Path] = None) -> Path:
    """Ensure metadata CSV exists, create if needed. Composable."""
    data_dir = Path(data_dir)
    
    if csv_path is None:
        csv_path = data_dir / "metadata.csv"
    else:
        csv_path = Path(csv_path)
    
    # Create if doesn't exist
    if not csv_path.exists():
        logger.info(f"Creating metadata CSV at {csv_path}")
        samples = scan_directory_for_metadata(data_dir)
        create_csv_from_samples(samples, csv_path)
    
    return csv_path


def get_class_distribution(csv_path: Union[str, Path]) -> Dict[str, int]:
    """Get class distribution from CSV. Simple utility."""
    df = load_metadata_csv(csv_path)
    return df['class_name'].value_counts().to_dict()