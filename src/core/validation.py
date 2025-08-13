"""
Validation utilities - Do one thing: validate data
Unix philosophy: Simple validation functions that compose well
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

from PIL import Image

from .paths import find_images, is_image_file

logger = logging.getLogger(__name__)


def validate_image_file(image_path: Union[str, Path]) -> Dict[str, Union[bool, str]]:
    """Validate single image file. Single responsibility."""
    image_path = Path(image_path)
    
    result = {
        'valid': False,
        'exists': False,
        'is_image': False,
        'can_load': False,
        'error': None
    }
    
    # Check if file exists
    if not image_path.exists():
        result['error'] = 'File does not exist'
        return result
    result['exists'] = True
    
    # Check if it's an image file by extension
    if not is_image_file(image_path):
        result['error'] = 'Not a valid image extension'
        return result
    result['is_image'] = True
    
    # Try to load the image
    try:
        with Image.open(image_path) as img:
            # Check basic properties
            if img.size[0] > 0 and img.size[1] > 0:
                result['can_load'] = True
                result['valid'] = True
            else:
                result['error'] = f'Invalid dimensions: {img.size}'
    except Exception as e:
        result['error'] = str(e)
    
    return result


def validate_directory_structure(data_dir: Union[str, Path]) -> Dict[str, Union[bool, List[str]]]:
    """Validate dataset directory structure. Single purpose."""
    data_dir = Path(data_dir)
    
    result = {
        'valid': True,
        'errors': [],
        'warnings': []
    }
    
    if not data_dir.exists():
        result['valid'] = False
        result['errors'].append(f"Directory does not exist: {data_dir}")
        return result
    
    # Check for class subdirectories
    class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        result['valid'] = False
        result['errors'].append("No class subdirectories found")
        return result
    
    # Check each class directory for images
    for class_dir in class_dirs:
        images = find_images(class_dir)
        if not images:
            result['warnings'].append(f"No valid images found in {class_dir.name}")
    
    return result


def validate_images_batch(image_paths: List[Union[str, Path]]) -> Dict[str, Union[int, List[str]]]:
    """Validate batch of images. Composable function."""
    valid_count = 0
    invalid_files = []
    
    for image_path in image_paths:
        result = validate_image_file(image_path)
        if result['valid']:
            valid_count += 1
        else:
            invalid_files.append(f"{image_path}: {result['error']}")
    
    return {
        'total': len(image_paths),
        'valid': valid_count,
        'invalid': len(invalid_files),
        'invalid_files': invalid_files,
        'success_rate': valid_count / len(image_paths) if image_paths else 0.0
    }


def check_dataset_health(data_dir: Union[str, Path]) -> Dict:
    """Complete dataset health check. Composes other functions."""
    data_dir = Path(data_dir)
    
    # Structure validation
    structure = validate_directory_structure(data_dir)
    if not structure['valid']:
        return {
            'healthy': False,
            'structure': structure,
            'images': None
        }
    
    # Image validation
    all_images = find_images(data_dir)
    image_validation = validate_images_batch(all_images)
    
    # Overall health
    healthy = (structure['valid'] and 
              image_validation['success_rate'] > 0.9)  # 90% success rate threshold
    
    return {
        'healthy': healthy,
        'structure': structure,
        'images': image_validation
    }