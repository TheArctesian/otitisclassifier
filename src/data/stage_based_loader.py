# File: src/data/stage_based_loader.py

from typing import Dict, List, Tuple, Optional, Union, cast, Any, TypeVar, Generic, Protocol, Set
from collections.abc import Sized
from pathlib import Path
import logging
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pandas as pd
from sklearn.model_selection import train_test_split

from .loader import ImageDataset
from .class_mapping import UnifiedDataset
from ..core.classes import create_unified_classes

logger = logging.getLogger(__name__)

T = TypeVar('T')

class TrainingStage(Enum):
    """Medical AI training stages with strict data isolation."""
    BASE_TRAINING = "base_training"
    FINE_TUNING = "fine_tuning" 
    VALIDATION = "validation"
    CLINICAL_VALIDATION = "clinical_validation"

class StageBasedDatasetManager:
    """
    Manages datasets across medical AI training stages with strict isolation.
    
    Key principles:
    - No data leakage between stages
    - Progressive domain adaptation 
    - Clinical validation standards
    - FDA-compliant evaluation protocols
    """
    
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:  # Fix 1: Added proper type annotation
        """
        Initialize with stage-based configuration.
        
        Args:
            config: Dict mapping stage names to dataset configurations
        """
        self.config = config
        self.stage_datasets: Dict[str, Union[UnifiedDataset, ConcatDataset]] = {}
        self.data_isolation_log: List[Dict[str, Any]] = []
        
        # Validate configuration
        self._validate_data_isolation()
        
        # Load datasets for each stage
        self._load_stage_datasets()
    
    def _validate_data_isolation(self) -> None:
        """Ensure no dataset is used in multiple training stages."""
        all_datasets: List[str] = []  # Fix 4: Added type annotation
        training_stages = [TrainingStage.BASE_TRAINING, TrainingStage.FINE_TUNING]
        
        for stage_name, stage_config in self.config.items():
            if stage_name in [s.value for s in training_stages]:
                all_datasets.extend(stage_config['datasets'])
        
        # Check for overlaps in training stages
        unique_datasets = set(all_datasets)
        if len(unique_datasets) != len(all_datasets):
            overlaps = [d for d in all_datasets if all_datasets.count(d) > 1]
            raise ValueError(
                f"Data leakage detected! Datasets used in multiple training stages: {overlaps}"
            )
        
        logger.info("✓ Data isolation validation passed - no leakage between training stages")
    
    def _load_stage_datasets(self) -> None:
        """Load datasets for each stage with appropriate preprocessing."""
        for stage_name, stage_config in self.config.items():
            logger.info(f"Loading datasets for stage: {stage_name}")
            
            # Load individual datasets for this stage
            stage_datasets: List[UnifiedDataset] = []  # Fix 5: More specific type
            for dataset_name in stage_config['datasets']:
                data_path: str = stage_config['data_paths'][dataset_name]  # Fix 6: Type annotation
                
                # Create base dataset
                base_dataset = ImageDataset(
                    data_dir=data_path,
                    image_size=stage_config.get('image_size', 384),
                    use_radimagenet=stage_config.get('use_radimagenet', True),
                    training=(stage_name != 'validation')  # No training transforms for validation
                )
                
                # Apply unified class mapping
                unified_dataset = UnifiedDataset(base_dataset)
                stage_datasets.append(unified_dataset)
                
                # Log dataset usage
                self.data_isolation_log.append({
                    'stage': stage_name,
                    'dataset': dataset_name,
                    'path': data_path,
                    'samples': len(unified_dataset),
                    'usage': 'training' if stage_name != 'validation' else 'evaluation_only'
                })
            
            # Combine datasets for this stage if multiple
            combined_dataset: Union[UnifiedDataset, ConcatDataset]  # Fix 7: Proper type annotation
            if len(stage_datasets) == 1:
                combined_dataset = stage_datasets[0]
            else:
                combined_dataset = ConcatDataset(stage_datasets)
            
            self.stage_datasets[stage_name] = combined_dataset
            logger.info(f"✓ Stage '{stage_name}' loaded: {len(combined_dataset)} samples")
    
    def get_stage_dataloaders(self, 
                             stage: Union[TrainingStage, str],
                             batch_size: int = 32,
                             num_workers: int = 4) -> Dict[str, DataLoader]:  # Fix 8: More specific return type
        """
        Get DataLoaders for a specific training stage.
        
        Args:
            stage: Training stage (base_training, fine_tuning, validation)
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            
        Returns:
            Dict with DataLoaders for train/val/test splits as appropriate
        """
        if isinstance(stage, TrainingStage):
            stage_name = stage.value
        else:
            stage_name = stage
            
        if stage_name not in self.stage_datasets:
            raise ValueError(f"Stage '{stage_name}' not found. Available: {list(self.stage_datasets.keys())}")
        
        dataset = self.stage_datasets[stage_name]
        stage_config = self.config[stage_name]
        split_ratios: Dict[str, float] = stage_config['split_ratio']  # Fix 9: Type annotation
        
        dataloaders: Dict[str, DataLoader] = {}  # Fix 10: Type annotation
        
        # Validation stage - test only
        if stage_name == 'validation':
            dataloaders['test'] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            return dataloaders
        
        # Training stages - create train/val splits
        train_ratio = split_ratios.get('train', 0.8)
        val_ratio = split_ratios.get('val', 0.2)
        
        # Create stratified splits to maintain class balance
        total_size = len(dataset)  # Dataset already implements __len__
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        # Get labels for stratified splitting
        labels: List[int] = []  # Fix 12: Type annotation
        if hasattr(dataset, 'get_class_distribution'):
            # For unified datasets
            for i in range(len(dataset)):
                sample = dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    _, label = sample
                    labels.append(int(label))
                else:
                    raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
        else:
            # For concat datasets - need to extract labels
            for i in range(len(dataset)):
                sample = dataset[i]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    _, label = sample
                    labels.append(int(label))
                else:
                    raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
        
        # Stratified split to maintain class balance
        train_indices: List[int]  # Fix 15: Type annotation
        val_indices: List[int]    # Fix 16: Type annotation
        try:
            train_indices, val_indices = train_test_split(
                list(range(total_size)),  # Fix 17: Explicit list conversion
                test_size=val_size,
                stratify=labels,
                random_state=42
            )
        except ValueError:
            # Fallback to random split if stratification fails
            logger.warning(f"Stratified split failed for {stage_name}, using random split")
            train_indices, val_indices = train_test_split(
                list(range(total_size)),  # Fix 18: Explicit list conversion
                test_size=val_size,
                random_state=42
            )
        
        # Create subset datasets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # Create DataLoaders
        dataloaders['train'] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # For stable batch norm in training
        )
        
        dataloaders['val'] = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Created DataLoaders for {stage_name}: "
                   f"train={len(train_subset)}, val={len(val_subset)}")
        
        return dataloaders
    
    def get_data_isolation_report(self) -> pd.DataFrame:
        """Generate report showing data usage across stages."""
        return pd.DataFrame(self.data_isolation_log)
    
    def validate_cross_stage_isolation(self) -> Dict[str, Union[bool, List[str]]]:  # Fix 19: More specific return type
        """
        Validate that validation datasets were never used for training.
        Critical for FDA compliance and unbiased evaluation.
        """
        validation_datasets: Set[str] = set()  # Fix 20: Type annotation
        training_datasets: Set[str] = set()    # Fix 21: Type annotation
        
        for log_entry in self.data_isolation_log:
            if log_entry['stage'] == 'validation':
                validation_datasets.add(log_entry['dataset'])
            elif log_entry['usage'] == 'training':
                training_datasets.add(log_entry['dataset'])
        
        # Check for contamination
        contamination = validation_datasets.intersection(training_datasets)
        
        return {
            'isolation_valid': len(contamination) == 0,
            'contaminated_datasets': list(contamination),
            'validation_datasets': list(validation_datasets),
            'training_datasets': list(training_datasets)
        }

# Factory function for easy medical AI workflow setup
def create_medical_ai_datasets(
    base_training_path: str = "data/processed/ebasaran-kaggale",
    fine_tuning_path: str = "data/processed/uci-kaggle", 
    validation_path: str = "data/processed/vanak-figshare",
    image_size: int = 384
) -> StageBasedDatasetManager:
    """
    Factory function for standard medical AI multi-dataset setup.
    
    This configuration follows medical AI best practices:
    - Base training on largest, most comprehensive dataset
    - Fine-tuning on different institutional source
    - Validation on completely external dataset
    """
    config: Dict[str, Dict[str, Any]] = {  # Fix 22: Type annotation
        'base_training': {
            'datasets': ['ebasaran_kaggle'],
            'data_paths': {'ebasaran_kaggle': base_training_path},
            'split_ratio': {'train': 0.8, 'val': 0.2},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'aggressive'
        },
        'fine_tuning': {
            'datasets': ['uci_kaggle'],
            'data_paths': {'uci_kaggle': fine_tuning_path},
            'split_ratio': {'train': 0.9, 'val': 0.1},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'conservative'
        },
        'validation': {
            'datasets': ['vanak_figshare'],
            'data_paths': {'vanak_figshare': validation_path},
            'split_ratio': {'test': 1.0},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'none'
        }
    }
    
    return StageBasedDatasetManager(config)

# Usage example
if __name__ == "__main__":
    # Create stage-based dataset manager
    dataset_manager = create_medical_ai_datasets()
    
    # Phase 1: Base training
    train_loaders = dataset_manager.get_stage_dataloaders('base_training', batch_size=16)
    train_dataset = train_loaders['train'].dataset
    val_dataset = train_loaders['val'].dataset
    train_len = len(cast(Sized, train_dataset)) if hasattr(train_dataset, '__len__') else 'Unknown'
    val_len = len(cast(Sized, val_dataset)) if hasattr(val_dataset, '__len__') else 'Unknown'
    print(f"Base training - Train: {train_len}, Val: {val_len}")
    
    # Phase 2: Fine-tuning  
    finetune_loaders = dataset_manager.get_stage_dataloaders('fine_tuning', batch_size=8)
    ft_train_dataset = finetune_loaders['train'].dataset
    ft_val_dataset = finetune_loaders['val'].dataset
    ft_train_len = len(cast(Sized, ft_train_dataset)) if hasattr(ft_train_dataset, '__len__') else 'Unknown'
    ft_val_len = len(cast(Sized, ft_val_dataset)) if hasattr(ft_val_dataset, '__len__') else 'Unknown'
    print(f"Fine-tuning - Train: {ft_train_len}, Val: {ft_val_len}")
    
    # Phase 3: Final validation
    val_loaders = dataset_manager.get_stage_dataloaders('validation', batch_size=32)
    test_dataset = val_loaders['test'].dataset
    test_len = len(cast(Sized, test_dataset)) if hasattr(test_dataset, '__len__') else 'Unknown'
    print(f"Final validation - Test: {test_len}")
    
    # Validate data isolation
    isolation_report = dataset_manager.validate_cross_stage_isolation()
    print(f"Data isolation valid: {isolation_report['isolation_valid']}")
