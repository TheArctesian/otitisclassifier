"""
Progressive Unfreezing Scheduler for Medical Domain Adaptation

This module implements sophisticated progressive unfreezing strategies for medical AI,
providing gradual adaptation from ImageNet features to medical domain specificity
with clinical safety validation and optimal layer unfreezing schedules.

Key Features:
- Medical domain-aware layer unfreezing schedules
- Differential learning rates for backbone vs classifier layers
- Clinical performance monitoring during unfreezing
- Adaptive unfreezing based on convergence patterns
- EfficientNet-specific unfreezing strategies for medical imaging
- Cross-model coordination for dual architecture systems

Medical Domain Adaptation Strategy:
1. Start with frozen backbone, train classifier only (medical-specific features)
2. Unfreeze top layers gradually (fine-grained medical features)
3. Unfreeze middle layers with reduced learning rates (general features)
4. Fine-tune entire network with careful learning rate scheduling
5. Validate clinical performance at each unfreezing stage

Unix Philosophy: Single responsibility - progressive layer unfreezing with medical safety
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class UnfreezingStrategy(Enum):
    """Different progressive unfreezing strategies for medical domain adaptation."""
    LINEAR_PROGRESSION = "linear_progression"
    PERFORMANCE_BASED = "performance_based"
    LAYER_GROUP_BASED = "layer_group_based"
    MEDICAL_OPTIMIZED = "medical_optimized"
    DUAL_ARCHITECTURE_COORDINATED = "dual_architecture_coordinated"


class LayerType(Enum):
    """Types of layers in neural networks for targeted unfreezing."""
    STEM = auto()  # Initial convolutional layers
    BACKBONE_EARLY = auto()  # Early backbone layers
    BACKBONE_MIDDLE = auto()  # Middle backbone layers
    BACKBONE_LATE = auto()  # Late backbone layers
    FEATURE_EXTRACTION = auto()  # Feature extraction layers
    ATTENTION = auto()  # Attention mechanisms
    CLASSIFIER = auto()  # Final classification layers
    BATCH_NORM = auto()  # Batch normalization layers
    DROPOUT = auto()  # Dropout layers


@dataclass
class UnfreezingSchedule:
    """Schedule for progressive layer unfreezing."""
    
    # Unfreezing timing
    epoch: int
    layer_groups: List[str]
    learning_rate: float
    weight_decay: Optional[float] = None
    
    # Clinical validation
    validation_required: bool = True
    minimum_performance_threshold: float = 0.85
    clinical_safety_check: bool = True
    
    # Layer-specific parameters
    layer_specific_lr: Optional[Dict[str, float]] = None
    freeze_batch_norm: bool = False
    adaptive_lr_scaling: bool = True


@dataclass
class MedicalDomainAdaptationConfig:
    """Configuration for medical domain adaptation through progressive unfreezing."""
    
    # Unfreezing strategy
    strategy: UnfreezingStrategy = UnfreezingStrategy.MEDICAL_OPTIMIZED
    start_frozen: bool = True
    
    # Learning rate management
    base_learning_rate: float = 1e-4
    classifier_lr_multiplier: float = 10.0  # Higher LR for medical-specific classifier
    backbone_lr_multiplier: float = 0.1   # Lower LR for pretrained backbone
    layer_lr_decay: float = 0.95  # Decrease LR for earlier layers
    
    # Timing parameters
    initial_frozen_epochs: int = 5
    unfreezing_interval: int = 3
    final_finetuning_epochs: int = 10
    
    # Performance-based adaptation
    performance_based_unfreezing: bool = True
    convergence_patience: int = 3
    minimum_improvement_threshold: float = 0.01
    
    # Clinical safety
    clinical_validation_frequency: int = 2
    safety_performance_threshold: float = 0.90
    rollback_on_performance_drop: bool = True
    performance_drop_threshold: float = 0.05
    
    # Architecture-specific
    preserve_batch_norm_stats: bool = True
    unfreeze_batch_norm_gradually: bool = True
    attention_priority_unfreezing: bool = True
    
    # Dual architecture coordination
    coordinate_dual_models: bool = True
    synchronize_unfreezing_stages: bool = True
    cross_model_validation: bool = True


class EfficientNetLayerAnalyzer:
    """
    Analyzes EfficientNet architecture for optimal progressive unfreezing.
    
    Identifies layer groups, dependencies, and optimal unfreezing order
    specifically for EfficientNet variants used in medical imaging.
    """
    
    def __init__(self, model: nn.Module, model_variant: str = "efficientnet_b3"):
        """
        Initialize EfficientNet layer analyzer.
        
        Args:
            model: EfficientNet model to analyze
            model_variant: EfficientNet variant (b3, b4, etc.)
        """
        self.model = model
        self.model_variant = model_variant
        self.layer_groups = {}
        self.unfreezing_order = []
        
        # Analyze model architecture
        self._analyze_architecture()
        
        logger.info(f"Initialized EfficientNetLayerAnalyzer for {model_variant}")
        logger.info(f"Identified {len(self.layer_groups)} layer groups")
    
    def _analyze_architecture(self) -> None:
        """Analyze EfficientNet architecture and identify layer groups."""
        self.layer_groups = {
            'classifier': [],
            'global_pool': [],
            'conv_head': [],
            'blocks_late': [],
            'blocks_middle': [],
            'blocks_early': [],
            'conv_stem': [],
            'batch_norm': [],
            'attention': []
        }
        
        # Iterate through model modules
        for name, module in self.model.named_modules():
            if 'classifier' in name:
                self.layer_groups['classifier'].append(name)
            elif 'global_pool' in name or 'avgpool' in name:
                self.layer_groups['global_pool'].append(name)
            elif 'conv_head' in name:
                self.layer_groups['conv_head'].append(name)
            elif 'blocks' in name:
                # Determine if early, middle, or late based on block number
                if any(f'blocks.{i}' in name for i in range(0, 2)):
                    self.layer_groups['blocks_early'].append(name)
                elif any(f'blocks.{i}' in name for i in range(2, 5)):
                    self.layer_groups['blocks_middle'].append(name)
                else:
                    self.layer_groups['blocks_late'].append(name)
            elif 'conv_stem' in name:
                self.layer_groups['conv_stem'].append(name)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                self.layer_groups['batch_norm'].append(name)
            elif 'attn' in name or 'se' in name:  # Squeeze-and-excitation attention
                self.layer_groups['attention'].append(name)
        
        # Define optimal unfreezing order for medical domain adaptation
        self.unfreezing_order = [
            'classifier',      # Start with classifier (medical-specific)
            'global_pool',     # Global pooling layer
            'conv_head',       # Convolutional head
            'attention',       # Attention mechanisms (important for medical features)
            'blocks_late',     # Late blocks (high-level features)
            'blocks_middle',   # Middle blocks (mid-level features)
            'blocks_early',    # Early blocks (low-level features)
            'conv_stem'        # Stem convolution (basic features)
        ]
        
        # Filter out empty groups
        self.unfreezing_order = [group for group in self.unfreezing_order 
                                if self.layer_groups[group]]
    
    def get_layer_parameters(self, layer_group: str) -> List[torch.nn.Parameter]:
        """Get parameters for a specific layer group."""
        if layer_group not in self.layer_groups:
            logger.warning(f"Layer group '{layer_group}' not found")
            return []
        
        parameters = []
        layer_names = self.layer_groups[layer_group]
        
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                parameters.append(param)
        
        return parameters
    
    def freeze_layer_group(self, layer_group: str) -> int:
        """Freeze parameters in a specific layer group."""
        parameters = self.get_layer_parameters(layer_group)
        frozen_count = 0
        
        for param in parameters:
            param.requires_grad = False
            frozen_count += 1
        
        if frozen_count > 0:
            logger.info(f"Frozen {frozen_count} parameters in {layer_group}")
        
        return frozen_count
    
    def unfreeze_layer_group(self, layer_group: str) -> int:
        """Unfreeze parameters in a specific layer group."""
        parameters = self.get_layer_parameters(layer_group)
        unfrozen_count = 0
        
        for param in parameters:
            param.requires_grad = True
            unfrozen_count += 1
        
        if unfrozen_count > 0:
            logger.info(f"Unfrozen {frozen_count} parameters in {layer_group}")
        
        return unfrozen_count
    
    def get_unfreezing_schedule(self, 
                               config: MedicalDomainAdaptationConfig) -> List[UnfreezingSchedule]:
        """Generate progressive unfreezing schedule for medical domain adaptation."""
        schedule = []
        current_epoch = config.initial_frozen_epochs
        
        # Start with classifier only
        schedule.append(UnfreezingSchedule(
            epoch=0,
            layer_groups=['classifier'],
            learning_rate=config.base_learning_rate * config.classifier_lr_multiplier,
            validation_required=True,
            clinical_safety_check=True
        ))
        
        # Progressive unfreezing of other layer groups
        for i, layer_group in enumerate(self.unfreezing_order[1:]):  # Skip classifier
            # Calculate learning rate with decay for earlier layers
            lr_multiplier = config.backbone_lr_multiplier * (config.layer_lr_decay ** i)
            learning_rate = config.base_learning_rate * lr_multiplier
            
            schedule.append(UnfreezingSchedule(
                epoch=current_epoch,
                layer_groups=[layer_group],
                learning_rate=learning_rate,
                validation_required=True,
                clinical_safety_check=True,
                layer_specific_lr={layer_group: learning_rate}
            ))
            
            current_epoch += config.unfreezing_interval
        
        # Final fine-tuning stage
        schedule.append(UnfreezingSchedule(
            epoch=current_epoch,
            layer_groups=list(self.layer_groups.keys()),
            learning_rate=config.base_learning_rate * 0.1,  # Lower LR for final tuning
            validation_required=True,
            clinical_safety_check=True
        ))
        
        return schedule


class MedicalDomainAdapter:
    """
    Medical domain adaptation controller for progressive unfreezing.
    
    Manages domain adaptation from ImageNet to medical imaging with
    clinical performance monitoring and safety validation.
    """
    
    def __init__(self, config: MedicalDomainAdaptationConfig):
        """
        Initialize medical domain adapter.
        
        Args:
            config: Medical domain adaptation configuration
        """
        self.config = config
        self.adaptation_history = []
        self.performance_tracking = []
        self.rollback_points = {}
        
        logger.info("Initialized MedicalDomainAdapter for clinical AI adaptation")
    
    def should_adapt_domain(self, 
                           current_metrics: Dict[str, float],
                           epoch: int) -> bool:
        """
        Determine if domain adaptation (unfreezing) should proceed.
        
        Args:
            current_metrics: Current model performance metrics
            epoch: Current training epoch
            
        Returns:
            True if domain adaptation should proceed
        """
        if not self.config.performance_based_unfreezing:
            return True
        
        # Check convergence
        if len(self.performance_tracking) < self.config.convergence_patience:
            self.performance_tracking.append(current_metrics)
            return False
        
        # Check for improvement in recent epochs
        recent_metrics = self.performance_tracking[-self.config.convergence_patience:]
        recent_accuracies = [m.get('accuracy', 0.0) for m in recent_metrics]
        
        improvement = max(recent_accuracies) - min(recent_accuracies)
        
        # If little improvement, ready for next adaptation stage
        adaptation_ready = improvement < self.config.minimum_improvement_threshold
        
        if adaptation_ready:
            logger.info(f"Domain adaptation ready - improvement plateau detected: {improvement:.4f}")
        
        return adaptation_ready
    
    def validate_adaptation_safety(self, 
                                 current_metrics: Dict[str, float],
                                 previous_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Validate that domain adaptation maintains clinical safety.
        
        Args:
            current_metrics: Current model performance metrics
            previous_metrics: Previous performance metrics for comparison
            
        Returns:
            True if adaptation is clinically safe
        """
        # Check absolute performance thresholds
        sensitivity = current_metrics.get('sensitivity', 0.0)
        accuracy = current_metrics.get('accuracy', 0.0)
        
        safety_checks = []
        
        # Clinical safety thresholds
        safety_checks.append(sensitivity >= self.config.safety_performance_threshold)
        safety_checks.append(accuracy >= self.config.safety_performance_threshold)
        
        # Performance drop check
        if previous_metrics is not None:
            prev_sensitivity = previous_metrics.get('sensitivity', 0.0)
            prev_accuracy = previous_metrics.get('accuracy', 0.0)
            
            sensitivity_drop = prev_sensitivity - sensitivity
            accuracy_drop = prev_accuracy - accuracy
            
            safety_checks.append(sensitivity_drop <= self.config.performance_drop_threshold)
            safety_checks.append(accuracy_drop <= self.config.performance_drop_threshold)
        
        safety_valid = all(safety_checks)
        
        if not safety_valid:
            logger.warning("⚠️ Domain adaptation safety validation failed")
            logger.warning(f"  Sensitivity: {sensitivity:.4f} (threshold: {self.config.safety_performance_threshold})")
            logger.warning(f"  Accuracy: {accuracy:.4f} (threshold: {self.config.safety_performance_threshold})")
        
        return safety_valid
    
    def create_rollback_point(self, 
                             model: nn.Module, 
                             optimizer: optim.Optimizer, 
                             metrics: Dict[str, float],
                             epoch: int) -> str:
        """
        Create rollback point for domain adaptation safety.
        
        Args:
            model: Model to save state for
            optimizer: Optimizer to save state for
            metrics: Current performance metrics
            epoch: Current epoch
            
        Returns:
            Rollback point identifier
        """
        rollback_id = f"epoch_{epoch}_acc_{metrics.get('accuracy', 0):.4f}"
        
        self.rollback_points[rollback_id] = {
            'model_state': model.state_dict().copy(),
            'optimizer_state': optimizer.state_dict().copy(),
            'metrics': metrics.copy(),
            'epoch': epoch
        }
        
        logger.info(f"Created rollback point: {rollback_id}")
        return rollback_id
    
    def rollback_adaptation(self, 
                           model: nn.Module,
                           optimizer: optim.Optimizer,
                           rollback_id: str) -> bool:
        """
        Rollback domain adaptation to a previous safe state.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            rollback_id: Rollback point identifier
            
        Returns:
            True if rollback successful
        """
        if rollback_id not in self.rollback_points:
            logger.error(f"Rollback point {rollback_id} not found")
            return False
        
        rollback_data = self.rollback_points[rollback_id]
        
        try:
            model.load_state_dict(rollback_data['model_state'])
            optimizer.load_state_dict(rollback_data['optimizer_state'])
            
            logger.info(f"🔄 Rolled back to {rollback_id}")
            logger.info(f"  Restored metrics: {rollback_data['metrics']}")
            
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False


class ProgressiveUnfreezer:
    """
    Progressive unfreezing scheduler for dual-architecture medical AI system.
    
    Implements sophisticated progressive unfreezing strategies with medical domain
    adaptation, clinical safety validation, and cross-model coordination for
    dual architecture systems.
    
    Key Features:
    - EfficientNet-specific layer analysis and unfreezing
    - Medical domain adaptation with clinical safety monitoring
    - Performance-based adaptive unfreezing
    - Differential learning rates for optimal convergence
    - Cross-model coordination for dual architecture systems
    - Clinical validation at each unfreezing stage
    """
    
    def __init__(self,
                 binary_model: nn.Module,
                 multiclass_model: nn.Module,
                 config: MedicalDomainAdaptationConfig,
                 binary_optimizer: Optional[optim.Optimizer] = None,
                 multiclass_optimizer: Optional[optim.Optimizer] = None):
        """
        Initialize progressive unfreezer for dual architecture models.
        
        Args:
            binary_model: Binary screening model
            multiclass_model: Multi-class diagnostic model
            config: Medical domain adaptation configuration
            binary_optimizer: Optional binary model optimizer
            multiclass_optimizer: Optional multiclass model optimizer
        """
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.config = config
        self.binary_optimizer = binary_optimizer
        self.multiclass_optimizer = multiclass_optimizer
        
        # Architecture analyzers
        self.binary_analyzer = EfficientNetLayerAnalyzer(binary_model, "efficientnet_b3")
        self.multiclass_analyzer = EfficientNetLayerAnalyzer(multiclass_model, "efficientnet_b4")
        
        # Domain adaptation controllers
        self.binary_adapter = MedicalDomainAdapter(config)
        self.multiclass_adapter = MedicalDomainAdapter(config)
        
        # Unfreezing schedules
        self.binary_schedule = self.binary_analyzer.get_unfreezing_schedule(config)
        self.multiclass_schedule = self.multiclass_analyzer.get_unfreezing_schedule(config)
        
        # State tracking
        self.current_stage = 0
        self.unfreezing_history = []
        self.performance_history = []
        
        # Initially freeze models if configured
        if config.start_frozen:
            self._freeze_all_layers()
        
        logger.info("Initialized ProgressiveUnfreezer for dual-architecture medical AI")
        logger.info(f"Binary model schedule: {len(self.binary_schedule)} stages")
        logger.info(f"Multiclass model schedule: {len(self.multiclass_schedule)} stages")
    
    def _freeze_all_layers(self) -> None:
        """Freeze all layers in both models except classifiers."""
        # Freeze binary model
        for name, param in self.binary_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        # Freeze multiclass model
        for name, param in self.multiclass_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        
        logger.info("🥶 Froze all layers except classifiers for initial training")
    
    def should_unfreeze_next_stage(self,
                                  binary_metrics: Dict[str, float],
                                  multiclass_metrics: Optional[Dict[str, float]] = None,
                                  epoch: int = 0) -> bool:
        """
        Determine if next unfreezing stage should be triggered.
        
        Args:
            binary_metrics: Binary model performance metrics
            multiclass_metrics: Optional multiclass model metrics
            epoch: Current training epoch
            
        Returns:
            True if next stage should be unfrozen
        """
        if self.current_stage >= len(self.binary_schedule):
            return False  # All stages completed
        
        current_schedule = self.binary_schedule[self.current_stage]
        
        # Check epoch-based timing
        epoch_ready = epoch >= current_schedule.epoch
        
        # Check performance-based readiness for binary model
        binary_ready = self.binary_adapter.should_adapt_domain(binary_metrics, epoch)
        
        # Check multiclass model if provided
        multiclass_ready = True
        if multiclass_metrics is not None and self.config.coordinate_dual_models:
            multiclass_ready = self.multiclass_adapter.should_adapt_domain(multiclass_metrics, epoch)
        
        # Clinical safety validation
        binary_safe = self.binary_adapter.validate_adaptation_safety(binary_metrics)
        multiclass_safe = True
        if multiclass_metrics is not None:
            multiclass_safe = self.multiclass_adapter.validate_adaptation_safety(multiclass_metrics)
        
        # All conditions must be met
        ready_to_unfreeze = (
            epoch_ready and 
            binary_ready and 
            multiclass_ready and 
            binary_safe and 
            multiclass_safe
        )
        
        if ready_to_unfreeze:
            logger.info(f"✅ Ready to unfreeze stage {self.current_stage}")
        else:
            conditions = [
                f"epoch: {epoch_ready}",
                f"binary_perf: {binary_ready}",
                f"multiclass_perf: {multiclass_ready}",
                f"binary_safe: {binary_safe}",
                f"multiclass_safe: {multiclass_safe}"
            ]
            logger.info(f"⏳ Unfreezing delayed - conditions: {conditions}")
        
        return ready_to_unfreeze
    
    def unfreeze_next_stage(self, epoch: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Unfreeze the next stage of layers for both models.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Tuple of (success, unfreezing_info)
        """
        if self.current_stage >= len(self.binary_schedule):
            return False, {'message': 'All stages already unfrozen'}
        
        binary_stage = self.binary_schedule[self.current_stage]
        multiclass_stage = self.multiclass_schedule[min(self.current_stage, 
                                                       len(self.multiclass_schedule) - 1)]
        
        try:
            # Create rollback points before unfreezing
            if self.config.rollback_on_performance_drop:
                binary_rollback = self.binary_adapter.create_rollback_point(
                    self.binary_model, self.binary_optimizer, {}, epoch
                )
                multiclass_rollback = self.multiclass_adapter.create_rollback_point(
                    self.multiclass_model, self.multiclass_optimizer, {}, epoch
                )
            
            # Unfreeze binary model layers
            binary_unfrozen = 0
            for layer_group in binary_stage.layer_groups:
                binary_unfrozen += self.binary_analyzer.unfreeze_layer_group(layer_group)
            
            # Unfreeze multiclass model layers
            multiclass_unfrozen = 0
            for layer_group in multiclass_stage.layer_groups:
                multiclass_unfrozen += self.multiclass_analyzer.unfreeze_layer_group(layer_group)
            
            # Update optimizers with new learning rates
            self._update_optimizer_lr(self.binary_optimizer, binary_stage)
            self._update_optimizer_lr(self.multiclass_optimizer, multiclass_stage)
            
            # Record unfreezing event
            unfreezing_info = {
                'stage': self.current_stage,
                'epoch': epoch,
                'binary_layers': binary_stage.layer_groups,
                'multiclass_layers': multiclass_stage.layer_groups,
                'binary_unfrozen_params': binary_unfrozen,
                'multiclass_unfrozen_params': multiclass_unfrozen,
                'binary_lr': binary_stage.learning_rate,
                'multiclass_lr': multiclass_stage.learning_rate
            }
            
            self.unfreezing_history.append(unfreezing_info)
            self.current_stage += 1
            
            logger.info(f"🔓 Unfrozen stage {self.current_stage - 1}")
            logger.info(f"  Binary: {binary_unfrozen} parameters in {binary_stage.layer_groups}")
            logger.info(f"  Multiclass: {multiclass_unfrozen} parameters in {multiclass_stage.layer_groups}")
            
            return True, unfreezing_info
        
        except Exception as e:
            logger.error(f"Unfreezing failed: {e}")
            return False, {'error': str(e)}
    
    def _update_optimizer_lr(self, 
                            optimizer: Optional[optim.Optimizer], 
                            schedule: UnfreezingSchedule) -> None:
        """Update optimizer learning rates for unfrozen layers."""
        if optimizer is None:
            return
        
        # Update learning rate for unfrozen parameter groups
        for param_group in optimizer.param_groups:
            param_group['lr'] = schedule.learning_rate
        
        logger.info(f"Updated optimizer learning rate to {schedule.learning_rate}")
    
    def get_unfrozen_parameters(self, model: nn.Module) -> List[torch.nn.Parameter]:
        """Get list of currently unfrozen parameters."""
        unfrozen_params = []
        
        for param in model.parameters():
            if param.requires_grad:
                unfrozen_params.append(param)
        
        return unfrozen_params
    
    def get_unfreezing_status(self) -> Dict[str, Any]:
        """Get current unfreezing status for both models."""
        binary_unfrozen = sum(1 for p in self.binary_model.parameters() if p.requires_grad)
        binary_total = sum(1 for p in self.binary_model.parameters())
        
        multiclass_unfrozen = sum(1 for p in self.multiclass_model.parameters() if p.requires_grad)
        multiclass_total = sum(1 for p in self.multiclass_model.parameters())
        
        return {
            'current_stage': self.current_stage,
            'total_stages': len(self.binary_schedule),
            'binary_model': {
                'unfrozen_parameters': binary_unfrozen,
                'total_parameters': binary_total,
                'unfrozen_percentage': binary_unfrozen / binary_total * 100
            },
            'multiclass_model': {
                'unfrozen_parameters': multiclass_unfrozen,
                'total_parameters': multiclass_total,
                'unfrozen_percentage': multiclass_unfrozen / multiclass_total * 100
            },
            'unfreezing_history': self.unfreezing_history,
            'schedule_binary': [
                {
                    'stage': i,
                    'epoch': stage.epoch,
                    'layers': stage.layer_groups,
                    'lr': stage.learning_rate
                }
                for i, stage in enumerate(self.binary_schedule)
            ],
            'schedule_multiclass': [
                {
                    'stage': i,
                    'epoch': stage.epoch,
                    'layers': stage.layer_groups,
                    'lr': stage.learning_rate
                }
                for i, stage in enumerate(self.multiclass_schedule)
            ]
        }
    
    def validate_clinical_safety_after_unfreezing(self,
                                                 binary_metrics: Dict[str, float],
                                                 multiclass_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Validate clinical safety after unfreezing operation.
        
        Args:
            binary_metrics: Binary model metrics after unfreezing
            multiclass_metrics: Optional multiclass model metrics
            
        Returns:
            True if clinical safety is maintained
        """
        # Validate binary model safety
        binary_safe = self.binary_adapter.validate_adaptation_safety(binary_metrics)
        
        # Validate multiclass model safety if provided
        multiclass_safe = True
        if multiclass_metrics is not None:
            multiclass_safe = self.multiclass_adapter.validate_adaptation_safety(multiclass_metrics)
        
        overall_safe = binary_safe and multiclass_safe
        
        if not overall_safe:
            logger.warning("⚠️ Clinical safety validation failed after unfreezing")
            
            # Consider rollback if configured
            if self.config.rollback_on_performance_drop:
                logger.info("Consider rollback to previous safe state")
        
        return overall_safe
    
    def emergency_rollback(self) -> bool:
        """
        Perform emergency rollback for both models to last safe state.
        
        Returns:
            True if rollback successful
        """
        logger.warning("🚨 Performing emergency rollback for clinical safety")
        
        rollback_success = True
        
        # Rollback binary model
        if (self.binary_adapter.rollback_points and 
            self.binary_optimizer is not None):
            latest_rollback = list(self.binary_adapter.rollback_points.keys())[-1]
            binary_success = self.binary_adapter.rollback_adaptation(
                self.binary_model, self.binary_optimizer, latest_rollback
            )
            rollback_success &= binary_success
        
        # Rollback multiclass model
        if (self.multiclass_adapter.rollback_points and 
            self.multiclass_optimizer is not None):
            latest_rollback = list(self.multiclass_adapter.rollback_points.keys())[-1]
            multiclass_success = self.multiclass_adapter.rollback_adaptation(
                self.multiclass_model, self.multiclass_optimizer, latest_rollback
            )
            rollback_success &= multiclass_success
        
        if rollback_success:
            # Adjust current stage
            self.current_stage = max(0, self.current_stage - 1)
            logger.info("✅ Emergency rollback completed successfully")
        else:
            logger.error("❌ Emergency rollback failed")
        
        return rollback_success


# Factory function for easy unfreezer creation
def create_progressive_unfreezer(
    binary_model: nn.Module,
    multiclass_model: nn.Module,
    binary_optimizer: Optional[optim.Optimizer] = None,
    multiclass_optimizer: Optional[optim.Optimizer] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> ProgressiveUnfreezer:
    """
    Factory function to create progressive unfreezer with medical AI defaults.
    
    Args:
        binary_model: Binary screening model
        multiclass_model: Multi-class diagnostic model
        binary_optimizer: Optional binary model optimizer
        multiclass_optimizer: Optional multiclass model optimizer
        config_overrides: Optional configuration overrides
        
    Returns:
        Configured ProgressiveUnfreezer instance
    """
    # Default medical AI configuration
    config = MedicalDomainAdaptationConfig(
        strategy=UnfreezingStrategy.MEDICAL_OPTIMIZED,
        start_frozen=True,
        performance_based_unfreezing=True,
        clinical_validation_frequency=2,
        safety_performance_threshold=0.90,
        rollback_on_performance_drop=True,
        coordinate_dual_models=True
    )
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    unfreezer = ProgressiveUnfreezer(
        binary_model=binary_model,
        multiclass_model=multiclass_model,
        config=config,
        binary_optimizer=binary_optimizer,
        multiclass_optimizer=multiclass_optimizer
    )
    
    logger.info("Created ProgressiveUnfreezer for dual-architecture medical AI")
    return unfreezer