"""
Dual Model Checkpoint Management for Medical AI System

This module implements comprehensive checkpoint management for the dual-architecture
otitis classification system, providing coordinated saving, loading, and versioning
of both binary screening and multi-class diagnostic models.

Key Features:
- Coordinated dual model checkpoint management
- Medical AI compliance with versioning and audit trails
- Automatic best model tracking with clinical metrics
- Checkpoint compression and storage optimization
- Clinical deployment checkpoint validation
- Cross-model compatibility checking
- FDA-compliant model versioning and documentation

Checkpoint Types:
1. Training Checkpoints: Regular training state saves
2. Best Model Checkpoints: Best performing models by clinical metrics
3. Deployment Checkpoints: Production-ready model packages
4. Backup Checkpoints: Safety backups with full training state

Unix Philosophy: Single responsibility - checkpoint management and model persistence
"""

import logging
import warnings
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from pathlib import Path
from datetime import datetime
import shutil
import zipfile

import torch
import torch.nn as nn
import numpy as np

from ..models.binary_screening import BinaryScreeningModel
from ..models.multiclass_diagnostic import MultiClassDiagnosticModel

logger = logging.getLogger(__name__)


class CheckpointType:
    """Checkpoint type definitions for different use cases."""
    TRAINING = "training"
    BEST_MODEL = "best_model"
    DEPLOYMENT = "deployment"
    BACKUP = "backup"
    VALIDATION = "validation"


class ModelType:
    """Model type definitions for dual architecture."""
    BINARY_SCREENING = "binary_screening"
    MULTICLASS_DIAGNOSTIC = "multiclass_diagnostic"
    DUAL_SYSTEM = "dual_system"


class CheckpointMetadata:
    """Metadata container for checkpoint information."""
    
    def __init__(self,
                 model_type: str,
                 checkpoint_type: str,
                 epoch: int,
                 metrics: Dict[str, float],
                 timestamp: Optional[datetime] = None,
                 version: str = "1.0.0",
                 clinical_validated: bool = False,
                 notes: str = ""):
        """
        Initialize checkpoint metadata.
        
        Args:
            model_type: Type of model (binary_screening, multiclass_diagnostic, dual_system)
            checkpoint_type: Type of checkpoint (training, best_model, deployment, etc.)
            epoch: Training epoch when checkpoint was created
            metrics: Performance metrics at checkpoint time
            timestamp: Checkpoint creation timestamp
            version: Model version
            clinical_validated: Whether checkpoint passed clinical validation
            notes: Additional notes or comments
        """
        self.model_type = model_type
        self.checkpoint_type = checkpoint_type
        self.epoch = epoch
        self.metrics = metrics
        self.timestamp = timestamp or datetime.now()
        self.version = version
        self.clinical_validated = clinical_validated
        self.notes = notes
        
        # Generate unique identifier
        self.checkpoint_id = self._generate_checkpoint_id()
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint identifier."""
        content = f"{self.model_type}_{self.checkpoint_type}_{self.epoch}_{self.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'model_type': self.model_type,
            'checkpoint_type': self.checkpoint_type,
            'epoch': self.epoch,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'clinical_validated': self.clinical_validated,
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary."""
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(
            model_type=data['model_type'],
            checkpoint_type=data['checkpoint_type'],
            epoch=data['epoch'],
            metrics=data['metrics'],
            timestamp=timestamp,
            version=data['version'],
            clinical_validated=data['clinical_validated'],
            notes=data['notes']
        )


class DualModelCheckpointManager:
    """
    Comprehensive checkpoint management for dual-architecture medical AI system.
    
    Handles coordinated saving and loading of both binary screening and multi-class
    diagnostic models with medical compliance and clinical validation tracking.
    """
    
    def __init__(self,
                 checkpoint_dir: Path,
                 max_checkpoints: int = 10,
                 compression: bool = True,
                 clinical_validation: bool = True):
        """
        Initialize dual model checkpoint manager.
        
        Args:
            checkpoint_dir: Base directory for storing checkpoints
            max_checkpoints: Maximum number of checkpoints to keep per model type
            compression: Whether to compress checkpoint files
            clinical_validation: Whether to perform clinical validation checks
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.clinical_validation = clinical_validation
        
        # Create directory structure
        self._setup_directory_structure()
        
        # Checkpoint tracking
        self.checkpoint_registry: Dict[str, List[CheckpointMetadata]] = {
            ModelType.BINARY_SCREENING: [],
            ModelType.MULTICLASS_DIAGNOSTIC: [],
            ModelType.DUAL_SYSTEM: []
        }
        
        # Load existing checkpoint registry
        self._load_checkpoint_registry()
        
        logger.info(f"Initialized dual model checkpoint manager at {self.checkpoint_dir}")
        logger.info(f"Compression: {self.compression}, Clinical validation: {self.clinical_validation}")
    
    def _setup_directory_structure(self) -> None:
        """Create necessary directory structure for checkpoints."""
        directories = [
            self.checkpoint_dir,
            self.checkpoint_dir / "binary_screening",
            self.checkpoint_dir / "multiclass_diagnostic", 
            self.checkpoint_dir / "dual_system",
            self.checkpoint_dir / "deployment",
            self.checkpoint_dir / "backups",
            self.checkpoint_dir / "metadata"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Checkpoint directory structure created")
    
    def _load_checkpoint_registry(self) -> None:
        """Load existing checkpoint registry from disk."""
        registry_file = self.checkpoint_dir / "metadata" / "checkpoint_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_type, checkpoints in data.items():
                    self.checkpoint_registry[model_type] = [
                        CheckpointMetadata.from_dict(cp) for cp in checkpoints
                    ]
                
                logger.info(f"Loaded checkpoint registry with {sum(len(cps) for cps in self.checkpoint_registry.values())} checkpoints")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint registry: {e}")
                logger.info("Starting with empty registry")
    
    def _save_checkpoint_registry(self) -> None:
        """Save checkpoint registry to disk."""
        registry_file = self.checkpoint_dir / "metadata" / "checkpoint_registry.json"
        
        data = {
            model_type: [cp.to_dict() for cp in checkpoints]
            for model_type, checkpoints in self.checkpoint_registry.items()
        }
        
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug("Checkpoint registry saved")
    
    def save_binary_screening_checkpoint(self,
                                       model: BinaryScreeningModel,
                                       optimizer: torch.optim.Optimizer,
                                       epoch: int,
                                       metrics: Dict[str, float],
                                       checkpoint_type: str = CheckpointType.TRAINING,
                                       notes: str = "") -> str:
        """
        Save binary screening model checkpoint.
        
        Args:
            model: Binary screening model to save
            optimizer: Optimizer state to save
            epoch: Current training epoch
            metrics: Performance metrics
            checkpoint_type: Type of checkpoint
            notes: Additional notes
            
        Returns:
            Checkpoint ID
        """
        metadata = CheckpointMetadata(
            model_type=ModelType.BINARY_SCREENING,
            checkpoint_type=checkpoint_type,
            epoch=epoch,
            metrics=metrics,
            notes=notes
        )
        
        # Clinical validation check
        if self.clinical_validation:
            metadata.clinical_validated = self._validate_binary_screening_metrics(metrics)
        
        # Create checkpoint data
        checkpoint_data = {
            'metadata': metadata.to_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_config': {
                'num_classes': 2,
                'color_feature_fusion': getattr(model, 'color_feature_fusion', True),
                'confidence_threshold': getattr(model, 'confidence_threshold', 0.5)
            }
        }
        
        # Save checkpoint
        checkpoint_path = self._save_checkpoint_file(
            checkpoint_data, 
            ModelType.BINARY_SCREENING, 
            metadata.checkpoint_id
        )
        
        # Update registry
        self.checkpoint_registry[ModelType.BINARY_SCREENING].append(metadata)
        self._cleanup_old_checkpoints(ModelType.BINARY_SCREENING)
        self._save_checkpoint_registry()
        
        logger.info(f"Saved binary screening checkpoint: {metadata.checkpoint_id} "
                   f"(epoch {epoch}, {checkpoint_type})")
        
        return metadata.checkpoint_id
    
    def save_multiclass_diagnostic_checkpoint(self,
                                            model: MultiClassDiagnosticModel,
                                            optimizer: torch.optim.Optimizer,
                                            epoch: int,
                                            metrics: Dict[str, float],
                                            checkpoint_type: str = CheckpointType.TRAINING,
                                            notes: str = "") -> str:
        """
        Save multi-class diagnostic model checkpoint.
        
        Args:
            model: Multi-class diagnostic model to save
            optimizer: Optimizer state to save
            epoch: Current training epoch
            metrics: Performance metrics
            checkpoint_type: Type of checkpoint
            notes: Additional notes
            
        Returns:
            Checkpoint ID
        """
        metadata = CheckpointMetadata(
            model_type=ModelType.MULTICLASS_DIAGNOSTIC,
            checkpoint_type=checkpoint_type,
            epoch=epoch,
            metrics=metrics,
            notes=notes
        )
        
        # Clinical validation check
        if self.clinical_validation:
            metadata.clinical_validated = self._validate_multiclass_diagnostic_metrics(metrics)
        
        # Create checkpoint data
        checkpoint_data = {
            'metadata': metadata.to_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'model_config': {
                'num_classes': 8,
                'color_feature_fusion': getattr(model, 'color_feature_fusion', True),
                'attention_mechanism': getattr(model, 'attention_mechanism', True),
                'focal_loss_config': getattr(model, 'focal_loss_config', {})
            }
        }
        
        # Save checkpoint
        checkpoint_path = self._save_checkpoint_file(
            checkpoint_data,
            ModelType.MULTICLASS_DIAGNOSTIC,
            metadata.checkpoint_id
        )
        
        # Update registry
        self.checkpoint_registry[ModelType.MULTICLASS_DIAGNOSTIC].append(metadata)
        self._cleanup_old_checkpoints(ModelType.MULTICLASS_DIAGNOSTIC)
        self._save_checkpoint_registry()
        
        logger.info(f"Saved multi-class diagnostic checkpoint: {metadata.checkpoint_id} "
                   f"(epoch {epoch}, {checkpoint_type})")
        
        return metadata.checkpoint_id
    
    def save_dual_system_checkpoint(self,
                                   binary_model: BinaryScreeningModel,
                                   multiclass_model: MultiClassDiagnosticModel,
                                   binary_optimizer: torch.optim.Optimizer,
                                   multiclass_optimizer: torch.optim.Optimizer,
                                   epoch: int,
                                   binary_metrics: Dict[str, float],
                                   multiclass_metrics: Dict[str, float],
                                   integration_metrics: Dict[str, float],
                                   checkpoint_type: str = CheckpointType.TRAINING,
                                   notes: str = "") -> str:
        """
        Save coordinated dual system checkpoint.
        
        Args:
            binary_model: Binary screening model
            multiclass_model: Multi-class diagnostic model
            binary_optimizer: Binary model optimizer
            multiclass_optimizer: Multi-class model optimizer
            epoch: Current training epoch
            binary_metrics: Binary model metrics
            multiclass_metrics: Multi-class model metrics
            integration_metrics: Dual system integration metrics
            checkpoint_type: Type of checkpoint
            notes: Additional notes
            
        Returns:
            Checkpoint ID
        """
        # Combine metrics
        combined_metrics = {
            'binary': binary_metrics,
            'multiclass': multiclass_metrics,
            'integration': integration_metrics
        }
        
        metadata = CheckpointMetadata(
            model_type=ModelType.DUAL_SYSTEM,
            checkpoint_type=checkpoint_type,
            epoch=epoch,
            metrics=combined_metrics,
            notes=notes
        )
        
        # Clinical validation check
        if self.clinical_validation:
            metadata.clinical_validated = (
                self._validate_binary_screening_metrics(binary_metrics) and
                self._validate_multiclass_diagnostic_metrics(multiclass_metrics) and
                self._validate_integration_metrics(integration_metrics)
            )
        
        # Create checkpoint data
        checkpoint_data = {
            'metadata': metadata.to_dict(),
            'binary_model_state_dict': binary_model.state_dict(),
            'multiclass_model_state_dict': multiclass_model.state_dict(),
            'binary_optimizer_state_dict': binary_optimizer.state_dict(),
            'multiclass_optimizer_state_dict': multiclass_optimizer.state_dict(),
            'epoch': epoch,
            'binary_metrics': binary_metrics,
            'multiclass_metrics': multiclass_metrics,
            'integration_metrics': integration_metrics,
            'dual_system_config': {
                'binary_config': {
                    'num_classes': 2,
                    'color_feature_fusion': getattr(binary_model, 'color_feature_fusion', True),
                    'confidence_threshold': getattr(binary_model, 'confidence_threshold', 0.5)
                },
                'multiclass_config': {
                    'num_classes': 8,
                    'color_feature_fusion': getattr(multiclass_model, 'color_feature_fusion', True),
                    'attention_mechanism': getattr(multiclass_model, 'attention_mechanism', True)
                }
            }
        }
        
        # Save checkpoint
        checkpoint_path = self._save_checkpoint_file(
            checkpoint_data,
            ModelType.DUAL_SYSTEM,
            metadata.checkpoint_id
        )
        
        # Update registry
        self.checkpoint_registry[ModelType.DUAL_SYSTEM].append(metadata)
        self._cleanup_old_checkpoints(ModelType.DUAL_SYSTEM)
        self._save_checkpoint_registry()
        
        logger.info(f"Saved dual system checkpoint: {metadata.checkpoint_id} "
                   f"(epoch {epoch}, {checkpoint_type})")
        
        return metadata.checkpoint_id
    
    def load_binary_screening_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load binary screening model checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier to load
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = self._find_checkpoint_file(ModelType.BINARY_SCREENING, checkpoint_id)
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Binary screening checkpoint {checkpoint_id} not found")
        
        checkpoint_data = self._load_checkpoint_file(checkpoint_path)
        
        logger.info(f"Loaded binary screening checkpoint: {checkpoint_id}")
        return checkpoint_data
    
    def load_multiclass_diagnostic_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load multi-class diagnostic model checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier to load
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = self._find_checkpoint_file(ModelType.MULTICLASS_DIAGNOSTIC, checkpoint_id)
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Multi-class diagnostic checkpoint {checkpoint_id} not found")
        
        checkpoint_data = self._load_checkpoint_file(checkpoint_path)
        
        logger.info(f"Loaded multi-class diagnostic checkpoint: {checkpoint_id}")
        return checkpoint_data
    
    def load_dual_system_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Load dual system checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier to load
            
        Returns:
            Checkpoint data dictionary
        """
        checkpoint_path = self._find_checkpoint_file(ModelType.DUAL_SYSTEM, checkpoint_id)
        
        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"Dual system checkpoint {checkpoint_id} not found")
        
        checkpoint_data = self._load_checkpoint_file(checkpoint_path)
        
        logger.info(f"Loaded dual system checkpoint: {checkpoint_id}")
        return checkpoint_data
    
    def get_best_checkpoint(self, model_type: str, metric: str = 'accuracy') -> Optional[CheckpointMetadata]:
        """
        Get best checkpoint by specified metric.
        
        Args:
            model_type: Type of model to search
            metric: Metric to optimize for
            
        Returns:
            Best checkpoint metadata or None if no checkpoints found
        """
        if model_type not in self.checkpoint_registry:
            return None
        
        checkpoints = self.checkpoint_registry[model_type]
        if not checkpoints:
            return None
        
        # Filter validated checkpoints if clinical validation is enabled
        if self.clinical_validation:
            checkpoints = [cp for cp in checkpoints if cp.clinical_validated]
        
        if not checkpoints:
            return None
        
        # Find best checkpoint by metric
        best_checkpoint = None
        best_value = -float('inf')
        
        for checkpoint in checkpoints:
            if model_type == ModelType.DUAL_SYSTEM:
                # For dual system, check integration metrics
                value = checkpoint.metrics.get('integration', {}).get(metric, -float('inf'))
            else:
                value = checkpoint.metrics.get(metric, -float('inf'))
            
            if value > best_value:
                best_value = value
                best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def create_deployment_package(self,
                                binary_checkpoint_id: str,
                                multiclass_checkpoint_id: str,
                                package_name: str,
                                version: str = "1.0.0") -> Path:
        """
        Create deployment package with both models.
        
        Args:
            binary_checkpoint_id: Binary screening checkpoint ID
            multiclass_checkpoint_id: Multi-class diagnostic checkpoint ID
            package_name: Name for deployment package
            version: Package version
            
        Returns:
            Path to deployment package
        """
        # Load checkpoints
        binary_checkpoint = self.load_binary_screening_checkpoint(binary_checkpoint_id)
        multiclass_checkpoint = self.load_multiclass_diagnostic_checkpoint(multiclass_checkpoint_id)
        
        # Validate compatibility
        self._validate_model_compatibility(binary_checkpoint, multiclass_checkpoint)
        
        # Create deployment package
        deployment_dir = self.checkpoint_dir / "deployment"
        package_dir = deployment_dir / f"{package_name}_v{version}"
        package_dir.mkdir(exist_ok=True)
        
        # Package contents
        package_data = {
            'package_info': {
                'name': package_name,
                'version': version,
                'created': datetime.now().isoformat(),
                'binary_checkpoint_id': binary_checkpoint_id,
                'multiclass_checkpoint_id': multiclass_checkpoint_id
            },
            'binary_model': {
                'state_dict': binary_checkpoint['model_state_dict'],
                'config': binary_checkpoint['model_config'],
                'metrics': binary_checkpoint['metrics']
            },
            'multiclass_model': {
                'state_dict': multiclass_checkpoint['model_state_dict'],
                'config': multiclass_checkpoint['model_config'],
                'metrics': multiclass_checkpoint['metrics']
            }
        }
        
        # Save package
        package_file = package_dir / "deployment_package.pth"
        torch.save(package_data, package_file)
        
        # Create documentation
        self._create_deployment_documentation(package_dir, package_data)
        
        # Create compressed archive
        if self.compression:
            archive_path = deployment_dir / f"{package_name}_v{version}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(package_dir))
            
            logger.info(f"Created deployment package: {archive_path}")
            return archive_path
        else:
            logger.info(f"Created deployment package: {package_dir}")
            return package_dir
    
    def _save_checkpoint_file(self, checkpoint_data: Dict[str, Any], model_type: str, checkpoint_id: str) -> Path:
        """Save checkpoint data to file."""
        model_dir = self.checkpoint_dir / model_type
        filename = f"{checkpoint_id}.pth"
        
        if self.compression:
            filename += ".gz"
        
        checkpoint_path = model_dir / filename
        
        if self.compression:
            import gzip
            with gzip.open(checkpoint_path, 'wb') as f:
                torch.save(checkpoint_data, f)
        else:
            torch.save(checkpoint_data, checkpoint_path)
        
        return checkpoint_path
    
    def _load_checkpoint_file(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint data from file."""
        if checkpoint_path.suffix == '.gz':
            import gzip
            with gzip.open(checkpoint_path, 'rb') as f:
                return torch.load(f, map_location='cpu')
        else:
            return torch.load(checkpoint_path, map_location='cpu')
    
    def _find_checkpoint_file(self, model_type: str, checkpoint_id: str) -> Optional[Path]:
        """Find checkpoint file by ID."""
        model_dir = self.checkpoint_dir / model_type
        
        # Try both compressed and uncompressed
        for suffix in ['.pth', '.pth.gz']:
            checkpoint_path = model_dir / f"{checkpoint_id}{suffix}"
            if checkpoint_path.exists():
                return checkpoint_path
        
        return None
    
    def _cleanup_old_checkpoints(self, model_type: str) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.checkpoint_registry[model_type]
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep most recent
        checkpoints.sort(key=lambda cp: cp.timestamp, reverse=True)
        
        # Remove old checkpoints
        old_checkpoints = checkpoints[self.max_checkpoints:]
        for old_checkpoint in old_checkpoints:
            checkpoint_path = self._find_checkpoint_file(model_type, old_checkpoint.checkpoint_id)
            if checkpoint_path and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint.checkpoint_id}")
        
        # Update registry
        self.checkpoint_registry[model_type] = checkpoints[:self.max_checkpoints]
    
    def _validate_binary_screening_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate binary screening metrics for clinical deployment."""
        required_metrics = ['sensitivity', 'specificity', 'accuracy']
        
        for metric in required_metrics:
            if metric not in metrics:
                logger.warning(f"Missing required metric: {metric}")
                return False
        
        # Clinical thresholds
        if metrics['sensitivity'] < 0.95:  # High sensitivity requirement
            logger.warning(f"Sensitivity too low: {metrics['sensitivity']:.3f} < 0.95")
            return False
        
        if metrics['specificity'] < 0.85:  # Reasonable specificity
            logger.warning(f"Specificity too low: {metrics['specificity']:.3f} < 0.85")
            return False
        
        return True
    
    def _validate_multiclass_diagnostic_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate multi-class diagnostic metrics for clinical deployment."""
        required_metrics = ['balanced_accuracy', 'accuracy']
        
        for metric in required_metrics:
            if metric not in metrics:
                logger.warning(f"Missing required metric: {metric}")
                return False
        
        # Clinical thresholds
        if metrics['balanced_accuracy'] < 0.80:  # Balanced accuracy for rare classes
            logger.warning(f"Balanced accuracy too low: {metrics['balanced_accuracy']:.3f} < 0.80")
            return False
        
        return True
    
    def _validate_integration_metrics(self, metrics: Dict[str, float]) -> bool:
        """Validate dual system integration metrics."""
        required_metrics = ['overall_accuracy', 'end_to_end_sensitivity']
        
        for metric in required_metrics:
            if metric not in metrics:
                logger.warning(f"Missing required integration metric: {metric}")
                return False
        
        # Integration thresholds
        if metrics['overall_accuracy'] < 0.85:
            logger.warning(f"Overall accuracy too low: {metrics['overall_accuracy']:.3f} < 0.85")
            return False
        
        return True
    
    def _validate_model_compatibility(self, binary_checkpoint: Dict, multiclass_checkpoint: Dict) -> None:
        """Validate that models are compatible for deployment."""
        # Check color feature compatibility
        binary_color = binary_checkpoint.get('model_config', {}).get('color_feature_fusion', True)
        multiclass_color = multiclass_checkpoint.get('model_config', {}).get('color_feature_fusion', True)
        
        if binary_color != multiclass_color:
            raise ValueError("Models have incompatible color feature configurations")
        
        logger.info("Model compatibility validated")
    
    def _create_deployment_documentation(self, package_dir: Path, package_data: Dict) -> None:
        """Create deployment documentation."""
        doc_content = f"""
# Otitis Classifier Deployment Package

## Package Information
- Name: {package_data['package_info']['name']}
- Version: {package_data['package_info']['version']}
- Created: {package_data['package_info']['created']}

## Models Included

### Binary Screening Model
- Checkpoint ID: {package_data['package_info']['binary_checkpoint_id']}
- Performance: {package_data['binary_model']['metrics']}
- Configuration: {package_data['binary_model']['config']}

### Multi-Class Diagnostic Model
- Checkpoint ID: {package_data['package_info']['multiclass_checkpoint_id']}
- Performance: {package_data['multiclass_model']['metrics']}
- Configuration: {package_data['multiclass_model']['config']}

## Usage
Load models using the provided checkpoint data and configurations.
Follow clinical deployment guidelines for medical AI systems.
"""
        
        doc_path = package_dir / "README.md"
        with open(doc_path, 'w') as f:
            f.write(doc_content)
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints."""
        summary = {}
        
        for model_type, checkpoints in self.checkpoint_registry.items():
            summary[model_type] = {
                'total_checkpoints': len(checkpoints),
                'validated_checkpoints': sum(1 for cp in checkpoints if cp.clinical_validated),
                'latest_checkpoint': checkpoints[-1].checkpoint_id if checkpoints else None,
                'best_checkpoints': {}
            }
            
            # Find best checkpoints by key metrics
            for metric in ['accuracy', 'sensitivity', 'specificity', 'balanced_accuracy']:
                best = self.get_best_checkpoint(model_type, metric)
                if best:
                    summary[model_type]['best_checkpoints'][metric] = best.checkpoint_id
        
        return summary


def create_checkpoint_manager(
    checkpoint_dir: Path,
    max_checkpoints: int = 10,
    compression: bool = True,
    clinical_validation: bool = True
) -> DualModelCheckpointManager:
    """
    Create a dual model checkpoint manager with medical AI configuration.
    
    Args:
        checkpoint_dir: Directory for storing checkpoints
        max_checkpoints: Maximum checkpoints to keep per model
        compression: Whether to compress checkpoint files
        clinical_validation: Whether to perform clinical validation
        
    Returns:
        Configured checkpoint manager
    """
    manager = DualModelCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        compression=compression,
        clinical_validation=clinical_validation
    )
    
    logger.info("Created dual model checkpoint manager for medical AI compliance")
    return manager