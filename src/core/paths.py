"""
Path utilities - Do one thing: manage file paths
Unix philosophy: Simple, composable path operations
"""

import os
from pathlib import Path
from typing import List, Optional, Union


def find_images(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    """Find all image files in directory. Does one thing well."""
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
    
    directory = Path(directory)
    if not directory.exists():
        return []
    
    images = []
    for ext in extensions:
        images.extend(directory.rglob(f"*{ext}"))
        images.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(images)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists. Simple utility."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_path(file_path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """Get relative path from base directory."""
    return Path(file_path).relative_to(Path(base_dir))


def change_extension(file_path: Union[str, Path], new_ext: str) -> Path:
    """Change file extension. Simple transformation."""
    path = Path(file_path)
    return path.with_suffix(new_ext)


def is_image_file(file_path: Union[str, Path]) -> bool:
    """Check if file is an image. Single responsibility."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif'}
    return Path(file_path).suffix.lower() in valid_extensions