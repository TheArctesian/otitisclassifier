"""
Medical AI Training Orchestrator for Dual-Architecture System

This module provides the main training orchestration interface that coordinates
all components of the dual-architecture medical AI training pipeline.

Key Features:
- Complete dual-architecture training workflow orchestration
- Integration of curriculum learning, progressive unfreezing, and checkpoint management
- Cross-model validation and clinical compliance monitoring
- FDA-compliant training pipeline with comprehensive audit trails
- Medical domain-specific training optimizations
- Production-ready training infrastructure for clinical deployment

Training Pipeline:
1. Data preparation and validation
2. Stage 1: Binary screening model training with progressive unfreezing
3. Stage 2: Multi-class diagnostic model training with curriculum learning
4. Stage 3: Integrated validation and clinical compliance testing
5. Deployment package creation and documentation

Unix Philosophy: Single responsibility - training orchestration and coordination
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .curriculum_learning import (
    CurriculumLearningOrchestrator,
    CurriculumLearningConfig,
    create_curriculum_learning_orchestrator
)
from .progressive_unfreezing import (
    ProgressiveUnfreezer,
    ProgressiveUnfreezingConfig,
    UnfreezingStrategy,
    create_progressive_unfreezer
)
from .checkpoint_manager import (
    DualModelCheckpointManager,
    CheckpointType,
    create_checkpoint_manager
)
from .cross_model_validation import (
    CrossModelValidator,
    ValidationProtocol,
    create_cross_model_validator
)

from ..models.binary_screening import BinaryScreeningModel, create_binary_screening_model
from ..models.multiclass_diagnostic import MultiClassDiagnosticModel, create_multiclass_diagnostic_model
from ..data.stage_based_loader import create_medical_ai_datasets

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Comprehensive training configuration for dual-architecture system."""
    
    def __init__(self,
                 # General training parameters
                 experiment_name: str = "dual_architecture_training",
                 base_output_dir: Path = Path("training_outputs"),
                 device: Optional[torch.device] = None,
                 random_seed: int = 42,
                 
                 # Data parameters
                 image_size: int = 500,
                 batch_sizes: Dict[str, int] = None,
                 
                 # Curriculum learning parameters
                 curriculum_config: Optional[CurriculumLearningConfig] = None,
                 
                 # Progressive unfreezing parameters
                 unfreezing_strategy: UnfreezingStrategy = UnfreezingStrategy.HYBRID,
                 
                 # Checkpoint management parameters
                 max_checkpoints: int = 10,
                 checkpoint_compression: bool = True,
                 
                 # Clinical validation parameters
                 clinical_validation: bool = True,
                 clinical_thresholds: Optional[Dict[str, float]] = None,
                 
                 # Advanced training parameters
                 mixed_precision: bool = True,
                 gradient_clipping: float = 1.0,
                 early_stopping_patience: int = 15):
        """
        Initialize comprehensive training configuration.
        
        Args:
            experiment_name: Name for the training experiment
            base_output_dir: Base directory for training outputs
            device: Training device (auto-detected if None)
            random_seed: Random seed for reproducibility
            image_size: Input image size
            batch_sizes: Batch sizes for different training stages
            curriculum_config: Curriculum learning configuration
            unfreezing_strategy: Progressive unfreezing strategy
            max_checkpoints: Maximum checkpoints to keep
            checkpoint_compression: Whether to compress checkpoints
            clinical_validation: Whether to perform clinical validation
            clinical_thresholds: Clinical performance thresholds
            mixed_precision: Whether to use mixed precision training
            gradient_clipping: Gradient clipping threshold
            early_stopping_patience: Early stopping patience
        """
        self.experiment_name = experiment_name
        self.base_output_dir = Path(base_output_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_seed = random_seed
        
        # Data configuration
        self.image_size = image_size
        self.batch_sizes = batch_sizes or {
            'binary_train': 32,
            'binary_val': 32,
            'multiclass_train': 16,
            'multiclass_val': 16,
            'integration_test': 32
        }
        
        # Training component configurations
        self.curriculum_config = curriculum_config or CurriculumLearningConfig()
        self.unfreezing_strategy = unfreezing_strategy
        
        # Infrastructure configuration
        self.max_checkpoints = max_checkpoints
        self.checkpoint_compression = checkpoint_compression
        self.clinical_validation = clinical_validation
        self.clinical_thresholds = clinical_thresholds or {
            'binary_sensitivity': 0.98,
            'binary_specificity': 0.90,
            'multiclass_balanced_accuracy': 0.85,
            'integration_accuracy': 0.85,
            'end_to_end_sensitivity': 0.95
        }
        
        # Advanced training parameters
        self.mixed_precision = mixed_precision
        self.gradient_clipping = gradient_clipping
        self.early_stopping_patience = early_stopping_patience
        
        # Create experiment directory
        self.experiment_dir = self.base_output_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up subdirectories
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.validation_dir = self.experiment_dir / "validation"
        self.deployment_dir = self.experiment_dir / "deployment"
        
        for directory in [self.checkpoint_dir, self.logs_dir, self.validation_dir, self.deployment_dir]:
            directory.mkdir(exist_ok=True)


class MedicalAITrainingOrchestrator:
    """
    Comprehensive training orchestrator for dual-architecture medical AI system.
    
    Coordinates all aspects of training including curriculum learning, progressive unfreezing,
    checkpoint management, and clinical validation for FDA-compliant medical AI development.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize medical AI training orchestrator.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
        
        # Initialize training components
        self.curriculum_orchestrator: Optional[CurriculumLearningOrchestrator] = None
        self.binary_unfreezer: Optional[ProgressiveUnfreezer] = None
        self.multiclass_unfreezer: Optional[ProgressiveUnfreezer] = None
        self.checkpoint_manager: Optional[DualModelCheckpointManager] = None
        self.validator: Optional[CrossModelValidator] = None
        
        # Training state
        self.training_started = False
        self.training_completed = False
        self.validation_completed = False
        
        # Performance tracking
        self.training_history: Dict[str, List[Dict[str, Any]]] = {
            'binary_training': [],
            'multiclass_training': [],
            'validation_results': []
        }
        
        logger.info(f"Initialized Medical AI Training Orchestrator")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Output directory: {config.experiment_dir}")
        logger.info(f"Device: {config.device}")
    
    def setup_training_infrastructure(self,
                                    base_training_path: Path,
                                    fine_tuning_path: Path,
                                    validation_path: Path) -> None:
        """
        Set up complete training infrastructure including data, models, and management components.
        
        Args:
            base_training_path: Path to base training dataset
            fine_tuning_path: Path to fine-tuning dataset
            validation_path: Path to validation dataset
        """
        logger.info("=== Setting up Training Infrastructure ===")
        
        # 1. Set up data loading
        logger.info("Setting up medical AI datasets...")
        self.dataset_manager = create_medical_ai_datasets(
            base_training_path=str(base_training_path),
            fine_tuning_path=str(fine_tuning_path),
            validation_path=str(validation_path),
            image_size=self.config.image_size,
            dual_architecture=True
        )
        
        # 2. Set up curriculum learning orchestrator
        logger.info("Setting up curriculum learning orchestrator...")
        self.curriculum_orchestrator = create_curriculum_learning_orchestrator(
            config=self.config.curriculum_config,
            device=self.config.device,
            checkpoint_dir=self.config.checkpoint_dir
        )
        
        # 3. Set up checkpoint manager
        logger.info("Setting up checkpoint manager...")
        self.checkpoint_manager = create_checkpoint_manager(
            checkpoint_dir=self.config.checkpoint_dir,
            max_checkpoints=self.config.max_checkpoints,
            compression=self.config.checkpoint_compression,
            clinical_validation=self.config.clinical_validation
        )
        
        logger.info("Training infrastructure setup complete")
    
    def train_binary_screening_stage(self) -> Dict[str, Any]:
        """
        Execute Stage 1: Binary screening model training with progressive unfreezing.
        
        Returns:
            Training results and metrics
        """
        logger.info("=== Starting Stage 1: Binary Screening Training ===")
        
        if not self.curriculum_orchestrator:
            raise RuntimeError("Training infrastructure not set up. Call setup_training_infrastructure() first.")
        
        # Set up binary screening model
        self.curriculum_orchestrator.setup_stage_1_binary_screening()
        binary_model = self.curriculum_orchestrator.binary_model
        
        # Set up progressive unfreezing for binary model
        self.binary_unfreezer = create_progressive_unfreezer(
            model=binary_model,
            strategy=self.config.unfreezing_strategy,
            total_epochs=self.config.curriculum_config.binary_epochs,
            device=self.config.device
        )
        
        # Get data loaders for binary training
        binary_loaders = self.dataset_manager.get_binary_screening_dataloaders(
            batch_size=self.config.batch_sizes['binary_train']
        )
        
        train_loader = binary_loaders['train']
        val_loader = binary_loaders['val']
        
        # Train binary screening model
        binary_metrics = self.curriculum_orchestrator.train_binary_screening_stage(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Save best binary model checkpoint
        if self.checkpoint_manager:
            checkpoint_id = self.checkpoint_manager.save_binary_screening_checkpoint(
                model=binary_model,
                optimizer=torch.optim.AdamW(binary_model.parameters()),  # Dummy optimizer for checkpoint
                epoch=self.config.curriculum_config.binary_epochs,
                metrics=binary_metrics,
                checkpoint_type=CheckpointType.BEST_MODEL,
                notes="Best binary screening model from Stage 1 training"
            )
            logger.info(f"Saved binary screening checkpoint: {checkpoint_id}")
        
        # Update training history
        self.training_history['binary_training'] = self.curriculum_orchestrator.get_training_history()['binary_screening']
        
        logger.info("=== Stage 1: Binary Screening Training Complete ===")
        return {
            'stage': 'binary_screening',
            'metrics': binary_metrics,
            'checkpoint_id': checkpoint_id if self.checkpoint_manager else None,
            'training_history': self.training_history['binary_training']
        }
    
    def train_multiclass_diagnostic_stage(self) -> Dict[str, Any]:
        """
        Execute Stage 2: Multi-class diagnostic model training with curriculum learning.
        
        Returns:
            Training results and metrics
        """
        logger.info("=== Starting Stage 2: Multi-Class Diagnostic Training ===")
        
        if not self.curriculum_orchestrator:
            raise RuntimeError("Training infrastructure not set up. Call setup_training_infrastructure() first.")
        
        # Set up multi-class diagnostic model
        self.curriculum_orchestrator.setup_stage_2_multiclass_diagnostic()
        multiclass_model = self.curriculum_orchestrator.multiclass_model
        
        # Set up progressive unfreezing for multi-class model
        self.multiclass_unfreezer = create_progressive_unfreezer(
            model=multiclass_model,
            strategy=self.config.unfreezing_strategy,
            total_epochs=self.config.curriculum_config.multiclass_epochs,
            device=self.config.device
        )
        
        # Get data loaders for multi-class training
        multiclass_loaders = self.dataset_manager.get_diagnostic_dataloaders(
            batch_size=self.config.batch_sizes['multiclass_train']
        )
        
        train_loader = multiclass_loaders['train']
        val_loader = multiclass_loaders['val']
        
        # Train multi-class diagnostic model
        multiclass_metrics = self.curriculum_orchestrator.train_multiclass_diagnostic_stage(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Save best multi-class model checkpoint
        if self.checkpoint_manager:
            checkpoint_id = self.checkpoint_manager.save_multiclass_diagnostic_checkpoint(
                model=multiclass_model,
                optimizer=torch.optim.AdamW(multiclass_model.parameters()),  # Dummy optimizer for checkpoint
                epoch=self.config.curriculum_config.multiclass_epochs,
                metrics=multiclass_metrics,
                checkpoint_type=CheckpointType.BEST_MODEL,
                notes="Best multi-class diagnostic model from Stage 2 training"
            )
            logger.info(f"Saved multi-class diagnostic checkpoint: {checkpoint_id}")
        
        # Update training history
        self.training_history['multiclass_training'] = self.curriculum_orchestrator.get_training_history()['multiclass_diagnostic']
        
        logger.info("=== Stage 2: Multi-Class Diagnostic Training Complete ===")
        return {
            'stage': 'multiclass_diagnostic',
            'metrics': multiclass_metrics,
            'checkpoint_id': checkpoint_id if self.checkpoint_manager else None,
            'training_history': self.training_history['multiclass_training']
        }
    
    def validate_integrated_system(self) -> Dict[str, Any]:
        """
        Execute Stage 3: Integrated validation and clinical compliance testing.
        
        Returns:
            Validation results and compliance status
        """
        logger.info("=== Starting Stage 3: Integrated Validation ===")
        
        if not self.curriculum_orchestrator or not self.curriculum_orchestrator.binary_model or not self.curriculum_orchestrator.multiclass_model:
            raise RuntimeError("Models not trained. Complete stages 1 and 2 first.")
        
        # Set up cross-model validator
        self.validator = create_cross_model_validator(
            binary_model=self.curriculum_orchestrator.binary_model,
            multiclass_model=self.curriculum_orchestrator.multiclass_model,
            device=self.config.device,
            clinical_thresholds=self.config.clinical_thresholds
        )
        
        # Get validation data loaders
        validation_loaders = self.dataset_manager.get_stage_dataloaders('validation', batch_size=self.config.batch_sizes['integration_test'])
        integration_loader = validation_loaders['test']
        
        # Prepare individual model validation data
        binary_val_loaders = self.dataset_manager.get_binary_screening_dataloaders(batch_size=self.config.batch_sizes['binary_val'])
        multiclass_val_loaders = self.dataset_manager.get_diagnostic_dataloaders(batch_size=self.config.batch_sizes['multiclass_val'])
        
        # Create class mapping for validation
        class_mapping = {
            0: "Normal Tympanic Membrane",
            1: "Acute Otitis Media",
            2: "Cerumen Impaction",
            3: "Chronic Suppurative Otitis Media",
            4: "Otitis Externa",
            5: "Tympanoskleros",
            6: "Ear Ventilation Tubes",
            7: "Pseudo Membranes",
            8: "Foreign Bodies"
        }
        
        # Run comprehensive validation suite
        validation_results = self.validator.run_full_validation_suite(
            binary_dataloader=binary_val_loaders['val'],
            multiclass_dataloader=multiclass_val_loaders['val'],
            integration_dataloader=integration_loader,
            consistency_dataloaders={'validation': integration_loader},  # Can add more datasets
            safety_dataloader=integration_loader,
            class_mapping=class_mapping
        )
        
        # Update validation history
        self.training_history['validation_results'] = validation_results
        self.validation_completed = True
        
        # Check clinical compliance
        clinical_compliance = self._assess_clinical_compliance(validation_results)
        
        logger.info("=== Stage 3: Integrated Validation Complete ===")
        return {
            'stage': 'integrated_validation',
            'validation_results': validation_results,
            'clinical_compliance': clinical_compliance,
            'validation_summary': self.validator.get_validation_summary()
        }
    
    def create_deployment_package(self, package_name: str, version: str = "1.0.0") -> Path:
        """
        Create production-ready deployment package with both models.
        
        Args:
            package_name: Name for the deployment package
            version: Package version
            
        Returns:
            Path to deployment package
        """
        logger.info("=== Creating Deployment Package ===")
        
        if not self.validation_completed:
            logger.warning("Creating deployment package without completing validation")
        
        if not self.checkpoint_manager:
            raise RuntimeError("Checkpoint manager not available")
        
        # Get best checkpoints
        best_binary = self.checkpoint_manager.get_best_checkpoint('binary_screening', 'sensitivity')
        best_multiclass = self.checkpoint_manager.get_best_checkpoint('multiclass_diagnostic', 'balanced_accuracy')
        
        if not best_binary or not best_multiclass:
            raise RuntimeError("Best model checkpoints not found")
        
        # Create deployment package
        package_path = self.checkpoint_manager.create_deployment_package(
            binary_checkpoint_id=best_binary.checkpoint_id,
            multiclass_checkpoint_id=best_multiclass.checkpoint_id,
            package_name=package_name,
            version=version
        )
        
        # Create deployment documentation
        self._create_deployment_documentation(package_path, package_name, version)
        
        logger.info(f"Deployment package created: {package_path}")
        return package_path
    
    def run_complete_training_pipeline(self,
                                     base_training_path: Path,
                                     fine_tuning_path: Path,
                                     validation_path: Path,
                                     package_name: str = "otitis_classifier") -> Dict[str, Any]:
        """
        Execute the complete dual-architecture training pipeline.
        
        Args:
            base_training_path: Path to base training dataset
            fine_tuning_path: Path to fine-tuning dataset
            validation_path: Path to validation dataset
            package_name: Name for deployment package
            
        Returns:
            Complete training results
        """
        logger.info("=== Starting Complete Medical AI Training Pipeline ===")
        
        pipeline_results = {}
        
        try:
            # Stage 0: Infrastructure setup
            self.setup_training_infrastructure(
                base_training_path=base_training_path,
                fine_tuning_path=fine_tuning_path,
                validation_path=validation_path
            )
            
            # Stage 1: Binary screening training
            binary_results = self.train_binary_screening_stage()
            pipeline_results['binary_training'] = binary_results
            
            # Stage 2: Multi-class diagnostic training
            multiclass_results = self.train_multiclass_diagnostic_stage()
            pipeline_results['multiclass_training'] = multiclass_results
            
            # Stage 3: Integrated validation
            validation_results = self.validate_integrated_system()
            pipeline_results['validation'] = validation_results
            
            # Stage 4: Deployment package creation
            deployment_package = self.create_deployment_package(package_name)
            pipeline_results['deployment'] = {
                'package_path': str(deployment_package),
                'package_name': package_name
            }
            
            # Mark training as completed
            self.training_completed = True
            
            # Save complete training summary
            self._save_training_summary(pipeline_results)
            
            logger.info("=== Complete Medical AI Training Pipeline Finished Successfully ===")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        
        return pipeline_results
    
    def _assess_clinical_compliance(self, validation_results: Dict[str, Any]) -> Dict[str, bool]:
        """Assess clinical compliance based on validation results."""
        compliance = {}
        
        for protocol_name, result in validation_results.items():
            if hasattr(result, 'clinical_validated'):
                compliance[protocol_name] = result.clinical_validated
            else:
                compliance[protocol_name] = False
        
        # Overall compliance
        compliance['overall_compliant'] = all(compliance.values())
        
        return compliance
    
    def _create_deployment_documentation(self, package_path: Path, package_name: str, version: str) -> None:
        """Create comprehensive deployment documentation."""
        if isinstance(package_path, Path) and package_path.is_dir():
            doc_dir = package_path
        else:
            # If it's a zip file, create documentation in the parent directory
            doc_dir = package_path.parent
        
        # Training summary documentation
        training_summary = {
            'package_info': {
                'name': package_name,
                'version': version,
                'created': datetime.now().isoformat(),
                'training_completed': self.training_completed,
                'validation_completed': self.validation_completed
            },
            'training_configuration': {
                'experiment_name': self.config.experiment_name,
                'image_size': self.config.image_size,
                'batch_sizes': self.config.batch_sizes,
                'clinical_thresholds': self.config.clinical_thresholds
            },
            'performance_summary': self._get_performance_summary()
        }
        
        summary_path = doc_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        logger.info(f"Deployment documentation created at {doc_dir}")
    
    def _save_training_summary(self, pipeline_results: Dict[str, Any]) -> None:
        """Save complete training summary."""
        summary_path = self.config.experiment_dir / "training_summary.json"
        
        summary_data = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'device': str(self.config.device),
                'random_seed': self.config.random_seed
            },
            'configuration': {
                'image_size': self.config.image_size,
                'batch_sizes': self.config.batch_sizes,
                'clinical_thresholds': self.config.clinical_thresholds,
                'unfreezing_strategy': self.config.unfreezing_strategy.value
            },
            'results': pipeline_results,
            'training_history': self.training_history
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Training summary saved to {summary_path}")
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model performance."""
        summary = {}
        
        if self.training_history['binary_training']:
            last_binary = self.training_history['binary_training'][-1]
            summary['binary_screening'] = {
                'final_epoch': last_binary.get('epoch', 'unknown'),
                'val_sensitivity': last_binary.get('val_sensitivity', 'unknown'),
                'val_specificity': last_binary.get('val_specificity', 'unknown'),
                'val_accuracy': last_binary.get('val_accuracy', 'unknown')
            }
        
        if self.training_history['multiclass_training']:
            last_multiclass = self.training_history['multiclass_training'][-1]
            summary['multiclass_diagnostic'] = {
                'final_epoch': last_multiclass.get('epoch', 'unknown'),
                'val_balanced_accuracy': last_multiclass.get('val_balanced_accuracy', 'unknown'),
                'val_accuracy': last_multiclass.get('val_accuracy', 'unknown')
            }
        
        return summary
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and progress."""
        return {
            'experiment_name': self.config.experiment_name,
            'experiment_dir': str(self.config.experiment_dir),
            'training_started': self.training_started,
            'training_completed': self.training_completed,
            'validation_completed': self.validation_completed,
            'components_initialized': {
                'curriculum_orchestrator': self.curriculum_orchestrator is not None,
                'checkpoint_manager': self.checkpoint_manager is not None,
                'validator': self.validator is not None
            },
            'training_history_length': {
                'binary': len(self.training_history['binary_training']),
                'multiclass': len(self.training_history['multiclass_training']),
                'validation': len(self.training_history['validation_results'])
            }
        }


def create_training_orchestrator(
    experiment_name: str = "dual_architecture_medical_ai",
    base_output_dir: Path = Path("training_outputs"),
    clinical_validation: bool = True,
    device: Optional[torch.device] = None
) -> MedicalAITrainingOrchestrator:
    """
    Create a medical AI training orchestrator with optimized configuration.
    
    Args:
        experiment_name: Name for the training experiment
        base_output_dir: Base directory for training outputs
        clinical_validation: Whether to perform clinical validation
        device: Training device (auto-detected if None)
        
    Returns:
        Configured medical AI training orchestrator
    """
    config = TrainingConfig(
        experiment_name=experiment_name,
        base_output_dir=base_output_dir,
        clinical_validation=clinical_validation,
        device=device
    )
    
    orchestrator = MedicalAITrainingOrchestrator(config)
    
    logger.info("Created medical AI training orchestrator for dual-architecture system")
    return orchestrator