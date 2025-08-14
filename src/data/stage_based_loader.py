# File: src/data/stage_based_loader.py

from typing import Dict, List, Tuple, Optional, Union, cast, Any, TypeVar, Generic, Protocol, Set
from collections.abc import Sized
from pathlib import Path
import logging
from enum import Enum

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .loader import ImageDataset
from .class_mapping import UnifiedDataset
from ..core.classes import create_unified_classes

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BinaryScreeningDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    Wrapper dataset for binary screening classification.
    
    Converts multi-class otitis dataset to binary classification:
    - Class 0: Normal (Normal Tympanic Membrane)
    - Class 1: Pathological (all other conditions)
    
    This supports the dual-architecture approach where Stage 1 performs
    high-sensitivity binary screening followed by Stage 2 multi-class diagnosis.
    """
    
    def __init__(self, base_dataset: Dataset[Tuple[torch.Tensor, int]]):
        """
        Initialize binary screening dataset wrapper.
        
        Args:
            base_dataset: Base multi-class dataset (UnifiedDataset or ConcatDataset)
        """
        self.base_dataset: Dataset[Tuple[torch.Tensor, int]] = base_dataset
        
        # Class mapping for binary classification
        # Assuming unified class indices where 0 = Normal, 1-8 = Various pathological conditions
        self.class_mapping = {
            0: 0,  # Normal -> Normal (0)
            1: 1,  # AOM -> Pathological (1)
            2: 1,  # Cerumen/Earwax -> Pathological (1)
            3: 1,  # Chronic Suppurative OM -> Pathological (1)
            4: 1,  # Otitis Externa -> Pathological (1)
            5: 1,  # Tympanoskleros/Myringosclerosis -> Pathological (1)
            6: 1,  # Tympanostomy Tubes -> Pathological (1)
            7: 1,  # Pseudo Membranes -> Pathological (1)
            8: 1,  # Foreign Objects -> Pathological (1)
        }
        
        logger.info(f"Created BinaryScreeningDataset with {len(self)} samples")
        self._log_class_distribution()
    
    def __len__(self) -> int:
        """Return total number of samples."""
        if hasattr(self.base_dataset, '__len__'):
            return len(self.base_dataset)  # type: ignore
        else:
            raise TypeError(f"Dataset {type(self.base_dataset)} does not support len()")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item with binary label conversion.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image, binary_label) where binary_label is 0=Normal, 1=Pathological
        """
        # Get original sample
        sample = self.base_dataset[idx]
        
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image, original_label = sample[0], sample[1]
            
            # Convert to int if needed
            if isinstance(original_label, torch.Tensor):
                original_label = original_label.item()
            original_label = int(original_label)
            
            # Map to binary label
            binary_label = self.class_mapping.get(original_label, 1)  # Default to pathological if unknown
            
            return image, binary_label
        else:
            raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
    
    def _log_class_distribution(self):
        """Log the binary class distribution for validation."""
        normal_count = 0
        pathological_count = 0
        
        # Sample a subset for efficiency with large datasets
        sample_size = min(1000, len(self))
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in indices:
            _, label = self.__getitem__(idx)
            if label == 0:
                normal_count += 1
            else:
                pathological_count += 1
        
        # Scale counts to full dataset
        scale_factor = len(self) / sample_size
        estimated_normal = int(normal_count * scale_factor)
        estimated_pathological = int(pathological_count * scale_factor)
        
        logger.info(f"Binary class distribution estimate:")
        logger.info(f"  Normal: {estimated_normal} ({estimated_normal/len(self)*100:.1f}%)")
        logger.info(f"  Pathological: {estimated_pathological} ({estimated_pathological/len(self)*100:.1f}%)")
    
    def get_binary_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for binary classification to handle imbalance.
        
        Returns:
            Class weights tensor [weight_normal, weight_pathological]
        """
        normal_count = 0
        pathological_count = 0
        
        # Count actual distribution
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            if label == 0:
                normal_count += 1
            else:
                pathological_count += 1
        
        total_samples = normal_count + pathological_count
        
        # Calculate inverse frequency weights
        weight_normal = total_samples / (2 * normal_count) if normal_count > 0 else 1.0
        weight_pathological = total_samples / (2 * pathological_count) if pathological_count > 0 else 1.0
        
        weights = torch.tensor([weight_normal, weight_pathological], dtype=torch.float32)
        
        logger.info(f"Binary class weights: Normal={weight_normal:.3f}, Pathological={weight_pathological:.3f}")
        
        return weights


class PathologyOnlyDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    Wrapper dataset for multi-class diagnostic classification (pathological cases only).
    
    Filters out Normal cases (class 0) and remaps pathological classes to 0-7 range:
    - Original class 1 (Earwax) -> Diagnostic class 1  
    - Original class 2 (AOM) -> Diagnostic class 0
    - Original class 3 (Chronic Suppurative OM) -> Diagnostic class 2
    - Original class 4 (Otitis Externa) -> Diagnostic class 3
    - Original class 5 (Tympanoskleros) -> Diagnostic class 4
    - Original class 6 (Ear Ventilation) -> Diagnostic class 5
    - Original class 7 (Pseudo Membranes) -> Diagnostic class 6
    - Original class 8 (Foreign Bodies) -> Diagnostic class 7
    
    This supports Stage 2 of the dual-architecture approach where only pathological
    cases flagged by Stage 1 binary screening are processed for specific diagnosis.
    """
    
    def __init__(self, base_dataset: Dataset[Tuple[torch.Tensor, int]]):
        """
        Initialize pathology-only dataset wrapper.
        
        Args:
            base_dataset: Base multi-class dataset (UnifiedDataset or ConcatDataset)
        """
        self.base_dataset: Dataset[Tuple[torch.Tensor, int]] = base_dataset
        
        # Create mapping from unified classes to diagnostic classes
        # Unified: 0=Normal, 1=Earwax, 2=AOM, 3=Chronic, 4=Otitis Externa, 
        #          5=Tympanoskleros, 6=Ear Ventilation, 7=Pseudo Membranes, 8=Foreign Bodies
        # Diagnostic: 0=AOM, 1=Earwax, 2=Chronic, 3=Otitis Externa,
        #             4=Tympanoskleros, 5=Ear Ventilation, 6=Pseudo Membranes, 7=Foreign Bodies
        self.unified_to_diagnostic = {
            1: 1,  # Earwax -> 1
            2: 0,  # AOM -> 0 (most common pathology first)
            3: 2,  # Chronic Suppurative OM -> 2
            4: 3,  # Otitis Externa -> 3
            5: 4,  # Tympanoskleros -> 4
            6: 5,  # Ear Ventilation -> 5
            7: 6,  # Pseudo Membranes -> 6
            8: 7,  # Foreign Bodies -> 7
        }
        
        # Filter out normal cases and create index mapping
        self.pathology_indices = []
        self._build_pathology_index()
        
        logger.info(f"Created PathologyOnlyDataset with {len(self)} pathological samples")
        self._log_pathology_distribution()
    
    def _build_pathology_index(self):
        """Build index of pathological cases (excluding Normal class 0)."""
        if not hasattr(self.base_dataset, '__len__'):
            raise TypeError(f"Dataset {type(self.base_dataset)} does not support len()")
        
        base_len = len(self.base_dataset)  # type: ignore
        for i in range(base_len):
            sample = self.base_dataset[i]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                _, original_label = sample[0], sample[1]
                
                # Convert to int if needed
                if isinstance(original_label, torch.Tensor):
                    original_label = original_label.item()
                original_label = int(original_label)
                
                # Only include pathological cases (not Normal class 0)
                if original_label in self.unified_to_diagnostic:
                    self.pathology_indices.append(i)
    
    def __len__(self) -> int:
        """Return number of pathological samples."""
        return len(self.pathology_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get pathological sample with diagnostic label conversion.
        
        Args:
            idx: Index in pathology dataset
            
        Returns:
            Tuple of (image, diagnostic_label) where diagnostic_label is 0-7 for pathological classes
        """
        # Get original index
        original_idx = self.pathology_indices[idx]
        
        # Get original sample
        sample = self.base_dataset[original_idx]
        
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            image, original_label = sample[0], sample[1]
            
            # Convert to int if needed
            if isinstance(original_label, torch.Tensor):
                original_label = original_label.item()
            original_label = int(original_label)
            
            # Map to diagnostic label
            diagnostic_label = self.unified_to_diagnostic.get(original_label)
            if diagnostic_label is None:
                raise ValueError(f"Unexpected pathological class: {original_label}")
            
            return image, diagnostic_label
        else:
            raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
    
    def _log_pathology_distribution(self):
        """Log the pathology class distribution for validation."""
        class_counts = {}
        pathology_class_names = {
            0: 'Acute_Otitis_Media',
            1: 'Earwax_Cerumen_Impaction',
            2: 'Chronic_Suppurative_Otitis_Media',
            3: 'Otitis_Externa',
            4: 'Tympanoskleros_Myringosclerosis',
            5: 'Ear_Ventilation_Tube',
            6: 'Pseudo_Membranes',
            7: 'Foreign_Bodies'
        }
        
        # Count all pathological samples for accurate distribution
        for idx in range(len(self)):
            _, label = self.__getitem__(idx)
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self)
        
        logger.info(f"Pathology class distribution:")
        for class_idx in range(8):  # 8 pathological classes
            count = class_counts.get(class_idx, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            class_name = pathology_class_names.get(class_idx, f"Class_{class_idx}")
            logger.info(f"  {class_idx}: {class_name}: {count} ({percentage:.1f}%)")
        
        # Identify rare classes for special handling
        rare_classes = [class_idx for class_idx, count in class_counts.items() if count < 20]
        if rare_classes:
            rare_names = [pathology_class_names.get(cls, f"Class_{cls}") for cls in rare_classes]
            logger.warning(f"Rare pathology classes detected (< 20 samples): {rare_names}")
            logger.info("Consider aggressive augmentation for rare classes during training")
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for pathological classification to handle imbalance.
        
        Returns:
            Class weights tensor [8] for the 8 pathological classes
        """
        class_counts = {}
        
        # Count actual distribution
        for i in range(len(self)):
            _, label = self.__getitem__(i)
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total_samples = len(self)
        num_classes = 8
        
        # Calculate inverse frequency weights with smoothing
        weights = torch.ones(num_classes, dtype=torch.float32)
        
        for class_idx in range(num_classes):
            count = class_counts.get(class_idx, 1)  # Avoid division by zero
            # Inverse frequency with smoothing factor
            weight = total_samples / (num_classes * count)
            weights[class_idx] = weight
        
        # Apply additional weighting for extremely rare classes
        rare_class_boost = {
            6: 1.5,  # Pseudo Membranes - boost by 1.5x
            7: 2.0,  # Foreign Bodies - boost by 2.0x
        }
        
        for class_idx, boost in rare_class_boost.items():
            weights[class_idx] *= boost
        
        logger.info(f"Pathology class weights: {weights.tolist()}")
        
        return weights


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
                    image_size=stage_config.get('image_size', 500),
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
    
    def get_binary_screening_dataloaders(self, 
                                       stage: Union[TrainingStage, str],
                                       batch_size: int = 32,
                                       num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Get DataLoaders specifically for binary screening (Normal vs Pathological).
        
        Converts multi-class labels to binary classification:
        - Class 0: Normal (Normal Tympanic Membrane)
        - Class 1: Pathological (all other 8 conditions)
        
        Args:
            stage: Training stage (base_training, fine_tuning, validation)
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            
        Returns:
            Dict with DataLoaders for binary screening training
        """
        if isinstance(stage, TrainingStage):
            stage_name = stage.value
        else:
            stage_name = stage
            
        if stage_name not in self.stage_datasets:
            raise ValueError(f"Stage '{stage_name}' not found. Available: {list(self.stage_datasets.keys())}")
        
        # Get base dataset
        base_dataset = self.stage_datasets[stage_name]
        
        # Create binary wrapper dataset
        binary_dataset = BinaryScreeningDataset(base_dataset)
        
        stage_config = self.config[stage_name]
        split_ratios: Dict[str, float] = stage_config['split_ratio']
        
        dataloaders: Dict[str, DataLoader] = {}
        
        # Validation stage - test only
        if stage_name == 'validation':
            dataloaders['test'] = DataLoader(
                binary_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            return dataloaders
        
        # Training stages - create train/val splits with binary class balancing
        train_ratio = split_ratios.get('train', 0.8)
        val_ratio = split_ratios.get('val', 0.2)
        
        total_size = len(binary_dataset)
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        # Get binary labels for stratified splitting
        binary_labels: List[int] = []
        for i in range(len(binary_dataset)):
            sample = binary_dataset[i]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                _, label = sample
                binary_labels.append(int(label))
            else:
                raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
        
        # Stratified split to maintain binary class balance
        try:
            train_indices, val_indices = train_test_split(
                list(range(total_size)),
                test_size=val_size,
                stratify=binary_labels,
                random_state=42
            )
        except ValueError:
            logger.warning(f"Binary stratified split failed for {stage_name}, using random split")
            train_indices, val_indices = train_test_split(
                list(range(total_size)),
                test_size=val_size,
                random_state=42
            )
        
        # Create subset datasets
        train_subset = Subset(binary_dataset, train_indices)
        val_subset = Subset(binary_dataset, val_indices)
        
        # Create DataLoaders
        dataloaders['train'] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        dataloaders['val'] = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Log binary class distribution
        train_binary_labels = [binary_labels[i] for i in train_indices]
        val_binary_labels = [binary_labels[i] for i in val_indices]
        
        train_normal = sum(1 for l in train_binary_labels if l == 0)
        train_pathological = sum(1 for l in train_binary_labels if l == 1)
        val_normal = sum(1 for l in val_binary_labels if l == 0)
        val_pathological = sum(1 for l in val_binary_labels if l == 1)
        
        logger.info(f"Binary Screening DataLoaders for {stage_name}:")
        logger.info(f"  Train: {len(train_subset)} samples (Normal: {train_normal}, Pathological: {train_pathological})")
        logger.info(f"  Val: {len(val_subset)} samples (Normal: {val_normal}, Pathological: {val_pathological})")
        
        return dataloaders
    
    def get_diagnostic_dataloaders(self, 
                                 stage: Union[TrainingStage, str],
                                 batch_size: int = 16,
                                 num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Get DataLoaders specifically for multi-class diagnostic model (pathological cases only).
        
        Filters out Normal cases (class 0) and provides only pathological cases for Stage 2
        multi-class diagnosis. Converts unified taxonomy labels to pathology-only labels (0-7).
        
        Args:
            stage: Training stage (base_training, fine_tuning, validation)
            batch_size: Batch size for DataLoaders (smaller for diagnostic model)
            num_workers: Number of worker processes
            
        Returns:
            Dict with DataLoaders for pathological cases only
        """
        if isinstance(stage, TrainingStage):
            stage_name = stage.value
        else:
            stage_name = stage
            
        if stage_name not in self.stage_datasets:
            raise ValueError(f"Stage '{stage_name}' not found. Available: {list(self.stage_datasets.keys())}")
        
        # Get base dataset
        base_dataset = self.stage_datasets[stage_name]
        
        # Create pathology-only dataset by filtering out Normal cases
        pathology_dataset = PathologyOnlyDataset(base_dataset)
        
        stage_config = self.config[stage_name]
        split_ratios: Dict[str, float] = stage_config['split_ratio']
        
        dataloaders: Dict[str, DataLoader] = {}
        
        # Validation stage - test only
        if stage_name == 'validation':
            dataloaders['test'] = DataLoader(
                pathology_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            return dataloaders
        
        # Training stages - create train/val splits with pathology class balancing
        train_ratio = split_ratios.get('train', 0.8)
        val_ratio = split_ratios.get('val', 0.2)
        
        total_size = len(pathology_dataset)
        if total_size == 0:
            logger.warning(f"No pathological cases found in {stage_name} dataset")
            return {}
        
        train_size = int(train_ratio * total_size)
        val_size = total_size - train_size
        
        # Get pathology labels for stratified splitting
        pathology_labels: List[int] = []
        for i in range(len(pathology_dataset)):
            sample = pathology_dataset[i]
            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                _, label = sample
                pathology_labels.append(int(label))
            else:
                raise ValueError(f"Expected tuple/list with 2+ elements, got {type(sample)}")
        
        # Stratified split to maintain pathology class balance
        try:
            train_indices, val_indices = train_test_split(
                list(range(total_size)),
                test_size=val_size,
                stratify=pathology_labels,
                random_state=42
            )
        except ValueError:
            logger.warning(f"Pathology stratified split failed for {stage_name}, using random split")
            train_indices, val_indices = train_test_split(
                list(range(total_size)),
                test_size=val_size,
                random_state=42
            )
        
        # Create subset datasets
        train_subset = Subset(pathology_dataset, train_indices)
        val_subset = Subset(pathology_dataset, val_indices)
        
        # Create DataLoaders with smaller batch size for diagnostic model
        dataloaders['train'] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        dataloaders['val'] = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Log pathology class distribution
        train_pathology_labels = [pathology_labels[i] for i in train_indices]
        val_pathology_labels = [pathology_labels[i] for i in val_indices]
        
        # Count per-class distribution
        from collections import Counter
        train_distribution = Counter(train_pathology_labels)
        val_distribution = Counter(val_pathology_labels)
        
        logger.info(f"Diagnostic (Pathology-Only) DataLoaders for {stage_name}:")
        logger.info(f"  Train: {len(train_subset)} samples")
        logger.info(f"  Val: {len(val_subset)} samples")
        logger.info(f"  Train class distribution: {dict(train_distribution)}")
        logger.info(f"  Val class distribution: {dict(val_distribution)}")
        
        return dataloaders
    
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
    image_size: int = 500,
    dual_architecture: bool = True
) -> StageBasedDatasetManager:
    """
    Factory function for standard medical AI multi-dataset setup.
    
    This configuration follows medical AI best practices:
    - Base training on largest, most comprehensive dataset
    - Fine-tuning on different institutional source
    - Validation on completely external dataset
    - Dual architecture support for binary screening + multi-class diagnosis
    
    Args:
        base_training_path: Path to base training dataset (ebasaran-kaggle)
        fine_tuning_path: Path to fine-tuning dataset (uci-kaggle)
        validation_path: Path to validation dataset (vanak-figshare)
        image_size: Target image size for preprocessing
        dual_architecture: Enable dual architecture data routing support
    """
    config: Dict[str, Dict[str, Any]] = {  # Fix 22: Type annotation
        'base_training': {
            'datasets': ['ebasaran_kaggle'],
            'data_paths': {'ebasaran_kaggle': base_training_path},
            'split_ratio': {'train': 0.8, 'val': 0.2},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'aggressive',
            'dual_architecture': dual_architecture
        },
        'fine_tuning': {
            'datasets': ['uci_kaggle'],
            'data_paths': {'uci_kaggle': fine_tuning_path},
            'split_ratio': {'train': 0.9, 'val': 0.1},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'conservative',
            'dual_architecture': dual_architecture
        },
        'validation': {
            'datasets': ['vanak_figshare'],
            'data_paths': {'vanak_figshare': validation_path},
            'split_ratio': {'test': 1.0},
            'image_size': image_size,
            'use_radimagenet': True,
            'augmentation_strength': 'none',
            'dual_architecture': dual_architecture
        }
    }
    
    return StageBasedDatasetManager(config)

# Usage example
if __name__ == "__main__":
    # Create stage-based dataset manager with dual architecture support
    dataset_manager = create_medical_ai_datasets(dual_architecture=True)
    
    # Phase 1: Binary Screening Model Training (Stage 1 of dual architecture)
    print("=== Binary Screening Model Training ===")
    binary_train_loaders = dataset_manager.get_binary_screening_dataloaders('base_training', batch_size=32)
    binary_train_dataset = binary_train_loaders['train'].dataset
    binary_val_dataset = binary_train_loaders['val'].dataset
    train_len = len(cast(Sized, binary_train_dataset)) if hasattr(binary_train_dataset, '__len__') else 'Unknown'
    val_len = len(cast(Sized, binary_val_dataset)) if hasattr(binary_val_dataset, '__len__') else 'Unknown'
    print(f"Binary Screening Base Training - Train: {train_len}, Val: {val_len}")
    
    # Binary screening fine-tuning
    binary_ft_loaders = dataset_manager.get_binary_screening_dataloaders('fine_tuning', batch_size=32)
    ft_train_dataset = binary_ft_loaders['train'].dataset
    ft_val_dataset = binary_ft_loaders['val'].dataset
    ft_train_len = len(cast(Sized, ft_train_dataset)) if hasattr(ft_train_dataset, '__len__') else 'Unknown'
    ft_val_len = len(cast(Sized, ft_val_dataset)) if hasattr(ft_val_dataset, '__len__') else 'Unknown'
    print(f"Binary Screening Fine-tuning - Train: {ft_train_len}, Val: {ft_val_len}")
    
    # Binary screening validation
    binary_val_loaders = dataset_manager.get_binary_screening_dataloaders('validation', batch_size=32)
    test_dataset = binary_val_loaders['test'].dataset
    test_len = len(cast(Sized, test_dataset)) if hasattr(test_dataset, '__len__') else 'Unknown'
    print(f"Binary Screening Validation - Test: {test_len}")
    
    # Phase 2: Multi-class Diagnostic Model Training (Stage 2 of dual architecture - pathology only)
    print("\n=== Multi-class Diagnostic Model Training (Pathology-Only) ===")
    diagnostic_train_loaders = dataset_manager.get_diagnostic_dataloaders('base_training', batch_size=16)
    diag_train_dataset = diagnostic_train_loaders['train'].dataset
    diag_val_dataset = diagnostic_train_loaders['val'].dataset
    diag_train_len = len(cast(Sized, diag_train_dataset)) if hasattr(diag_train_dataset, '__len__') else 'Unknown'
    diag_val_len = len(cast(Sized, diag_val_dataset)) if hasattr(diag_val_dataset, '__len__') else 'Unknown'
    print(f"Diagnostic Base Training - Train: {diag_train_len}, Val: {diag_val_len}")
    
    # Diagnostic fine-tuning
    diagnostic_ft_loaders = dataset_manager.get_diagnostic_dataloaders('fine_tuning', batch_size=16)
    diag_ft_train_dataset = diagnostic_ft_loaders['train'].dataset
    diag_ft_val_dataset = diagnostic_ft_loaders['val'].dataset
    diag_ft_train_len = len(cast(Sized, diag_ft_train_dataset)) if hasattr(diag_ft_train_dataset, '__len__') else 'Unknown'
    diag_ft_val_len = len(cast(Sized, diag_ft_val_dataset)) if hasattr(diag_ft_val_dataset, '__len__') else 'Unknown'
    print(f"Diagnostic Fine-tuning - Train: {diag_ft_train_len}, Val: {diag_ft_val_len}")
    
    # Diagnostic validation
    diagnostic_val_loaders = dataset_manager.get_diagnostic_dataloaders('validation', batch_size=16)
    diag_test_dataset = diagnostic_val_loaders['test'].dataset
    diag_test_len = len(cast(Sized, diag_test_dataset)) if hasattr(diag_test_dataset, '__len__') else 'Unknown'
    print(f"Diagnostic Validation - Test: {diag_test_len}")
    
    # Validate data isolation for FDA compliance
    print("\n=== Data Isolation Validation ===")
    isolation_report = dataset_manager.validate_cross_stage_isolation()
    print(f"Data isolation valid: {isolation_report['isolation_valid']}")
    if not isolation_report['isolation_valid']:
        print(f"Contaminated datasets: {isolation_report['contaminated_datasets']}")
    
    # Generate data isolation report
    isolation_df = dataset_manager.get_data_isolation_report()
    print(f"\nDataset usage summary:")
    print(isolation_df.groupby(['stage', 'usage']).size())
