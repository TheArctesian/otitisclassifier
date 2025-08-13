"""
Class weight utilities - Do one thing: calculate class weights
Unix philosophy: Simple weight calculation functions
"""

import logging
from typing import Dict, Union
from collections import Counter

import torch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_class_weights(
    class_counts: Dict[int, int], 
    method: str = 'inverse'
) -> torch.Tensor:
    """
    Calculate class weights from counts.
    Single responsibility: weight calculation only.
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = torch.zeros(num_classes)
    
    for class_idx, count in class_counts.items():
        if count == 0:
            weight = 0.0
        elif method == 'inverse':
            weight = total_samples / (num_classes * count)
        elif method == 'sqrt':
            weight = np.sqrt(total_samples / (num_classes * count))
        elif method == 'log':
            weight = np.log(total_samples / (num_classes * count) + 1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        weights[class_idx] = weight
    
    logger.info(f"Calculated weights using {method}: {weights}")
    return weights


def weights_from_samples(samples: pd.DataFrame, method: str = 'inverse') -> torch.Tensor:
    """Calculate weights from sample DataFrame."""
    class_counts = samples['class_idx'].value_counts().to_dict()
    return calculate_class_weights(class_counts, method)


def weights_from_counter(counter: Counter, method: str = 'inverse') -> torch.Tensor:
    """Calculate weights from Counter object."""
    class_counts = dict(counter)
    return calculate_class_weights(class_counts, method)


def balanced_weights(num_classes: int) -> torch.Tensor:
    """Create balanced weights (all equal)."""
    return torch.ones(num_classes)


def get_weight_methods() -> list:
    """Get available weight calculation methods."""
    return ['inverse', 'sqrt', 'log', 'balanced']