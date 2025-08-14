"""
Progressive Unfreezing with Medical Domain Adaptation

This module implements progressive layer unfreezing strategies specifically designed
for medical imaging domain adaptation, supporting the dual-architecture otitis
classification system's transfer learning requirements.

Key Features:
- Medical domain-aware progressive unfreezing schedules
- RadImageNet to general medical imaging adaptation
- Layer-wise learning rate scheduling for medical feature preservation
- Clinical safety-aware unfreezing with performance monitoring
- Dual-model coordination for consistent feature learning

Progressive Unfreezing Strategies:
1. Conservative Medical Unfreezing: Gradual unfreezing preserving medical features
2. Aggressive Fine-tuning: Rapid adaptation for domain-specific features
3. Hybrid Approach: Balanced unfreezing for optimal performance

Unix Philosophy: Single responsibility - progressive unfreezing orchestration
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class UnfreezingStrategy(Enum):
    """Progressive unfreezing strategies for medical domain adaptation."""
    CONSERVATIVE = "conservative"  # Gradual unfreezing preserving medical features
    AGGRESSIVE = "aggressive"      # Rapid adaptation for domain-specific features
    HYBRID = "hybrid"             # Balanced approach
    CUSTOM = "custom"             # User-defined schedule


class LayerGroup(Enum):
    """Layer groups for progressive unfreezing in medical models."""
    BACKBONE_EARLY = "backbone_early"      # Early feature extraction layers
    BACKBONE_MIDDLE = "backbone_middle"    # Mid-level feature layers
    BACKBONE_LATE = "backbone_late"        # High-level feature layers
    CLASSIFIER_HEAD = "classifier_head"    # Final classification layers
    COLOR_FEATURES = "color_features"      # Color feature extraction components
    ATTENTION = "attention"               # Attention mechanisms


class ProgressiveUnfreezingConfig:
    """Configuration for progressive unfreezing parameters."""
    
    def __init__(self,
                 strategy: UnfreezingStrategy = UnfreezingStrategy.HYBRID,
                 total_epochs: int = 100,
                 warmup_epochs: int = 5,
                 conservative_epochs: int = 30,
                 aggressive_epochs: int = 20,
                 base_lr: float = 1e-4,
                 frozen_lr_multiplier: float = 0.1,
                 unfrozen_lr_multiplier: float = 1.0,
                 layer_lr_decay: float = 0.8,
                 performance_threshold: float = 0.02,
                 safety_check_interval: int = 5,
                 min_improvement_threshold: float = 0.001):
        """
        Initialize progressive unfreezing configuration.
        
        Args:
            strategy: Unfreezing strategy to use
            total_epochs: Total training epochs
            warmup_epochs: Initial warmup epochs with all layers frozen
            conservative_epochs: Epochs for conservative unfreezing
            aggressive_epochs: Epochs for aggressive unfreezing
            base_lr: Base learning rate
            frozen_lr_multiplier: LR multiplier for frozen layers
            unfrozen_lr_multiplier: LR multiplier for unfrozen layers
            layer_lr_decay: LR decay factor for deeper layers
            performance_threshold: Performance improvement threshold for unfreezing
            safety_check_interval: Epochs between safety checks
            min_improvement_threshold: Minimum improvement to continue unfreezing
        """
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.conservative_epochs = conservative_epochs
        self.aggressive_epochs = aggressive_epochs
        
        self.base_lr = base_lr
        self.frozen_lr_multiplier = frozen_lr_multiplier
        self.unfrozen_lr_multiplier = unfrozen_lr_multiplier
        self.layer_lr_decay = layer_lr_decay
        
        self.performance_threshold = performance_threshold
        self.safety_check_interval = safety_check_interval
        self.min_improvement_threshold = min_improvement_threshold


class ProgressiveUnfreezer:
    """
    Implements progressive unfreezing for medical domain adaptation.
    
    Manages gradual layer unfreezing with medical-specific strategies and
    performance monitoring for clinical safety.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: ProgressiveUnfreezingConfig,
                 device: Optional[torch.device] = None):
        """
        Initialize progressive unfreezer.
        
        Args:
            model: Model to apply progressive unfreezing to
            config: Unfreezing configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Layer groups mapping
        self.layer_groups: Dict[LayerGroup, List[str]] = {}
        self._identify_layer_groups()
        
        # Unfreezing state
        self.current_epoch = 0
        self.unfrozen_groups: List[LayerGroup] = []
        self.performance_history: List[float] = []
        self.unfreezing_schedule: Dict[int, List[LayerGroup]] = {}
        
        # Generate unfreezing schedule
        self._generate_unfreezing_schedule()
        
        logger.info(f"Initialized progressive unfreezer with {self.config.strategy.value} strategy")
        logger.info(f"Layer groups identified: {list(self.layer_groups.keys())}")
    
    def _identify_layer_groups(self) -> None:
        """Identify layer groups in the model for progressive unfreezing."""
        self.layer_groups = {
            LayerGroup.BACKBONE_EARLY: [],
            LayerGroup.BACKBONE_MIDDLE: [],
            LayerGroup.BACKBONE_LATE: [],
            LayerGroup.CLASSIFIER_HEAD: [],
            LayerGroup.COLOR_FEATURES: [],
            LayerGroup.ATTENTION: []
        }
        
        # Identify layers based on common naming patterns
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ['color', 'lab', 'feature_extract']):
                self.layer_groups[LayerGroup.COLOR_FEATURES].append(name)
            elif any(x in name.lower() for x in ['attention', 'attn']):
                self.layer_groups[LayerGroup.ATTENTION].append(name)
            elif any(x in name.lower() for x in ['classifier', 'head', 'fc']):
                self.layer_groups[LayerGroup.CLASSIFIER_HEAD].append(name)
            elif any(x in name.lower() for x in ['efficientnet', 'backbone', 'features']):
                # Determine backbone layer depth
                if any(x in name.lower() for x in ['block_0', 'block_1', 'block_2', 'conv_stem']):
                    self.layer_groups[LayerGroup.BACKBONE_EARLY].append(name)
                elif any(x in name.lower() for x in ['block_3', 'block_4', 'block_5']):
                    self.layer_groups[LayerGroup.BACKBONE_MIDDLE].append(name)
                else:
                    self.layer_groups[LayerGroup.BACKBONE_LATE].append(name)
        
        # Log layer group contents
        for group, layers in self.layer_groups.items():
            if layers:
                logger.info(f"{group.value}: {len(layers)} layers")
    
    def _generate_unfreezing_schedule(self) -> None:
        """Generate epoch-based unfreezing schedule based on strategy."""
        if self.config.strategy == UnfreezingStrategy.CONSERVATIVE:
            self._generate_conservative_schedule()
        elif self.config.strategy == UnfreezingStrategy.AGGRESSIVE:
            self._generate_aggressive_schedule()
        elif self.config.strategy == UnfreezingStrategy.HYBRID:
            self._generate_hybrid_schedule()
        else:
            raise ValueError(f"Unsupported unfreezing strategy: {self.config.strategy}")
        
        logger.info("Unfreezing schedule generated:")
        for epoch, groups in self.unfreezing_schedule.items():
            logger.info(f"  Epoch {epoch}: Unfreeze {[g.value for g in groups]}")
    
    def _generate_conservative_schedule(self) -> None:
        """Generate conservative unfreezing schedule for medical domain preservation."""
        # Conservative approach: Unfreeze from top to bottom slowly
        schedule_points = [
            (self.config.warmup_epochs, [LayerGroup.CLASSIFIER_HEAD]),
            (self.config.warmup_epochs + 10, [LayerGroup.COLOR_FEATURES]),
            (self.config.warmup_epochs + 20, [LayerGroup.ATTENTION]),
            (self.config.warmup_epochs + 30, [LayerGroup.BACKBONE_LATE]),
            (self.config.warmup_epochs + 45, [LayerGroup.BACKBONE_MIDDLE]),
            (self.config.warmup_epochs + 60, [LayerGroup.BACKBONE_EARLY])
        ]
        
        for epoch, groups in schedule_points:
            if epoch < self.config.total_epochs:
                self.unfreezing_schedule[epoch] = groups
    
    def _generate_aggressive_schedule(self) -> None:
        """Generate aggressive unfreezing schedule for rapid adaptation."""
        # Aggressive approach: Unfreeze quickly for domain adaptation
        schedule_points = [
            (self.config.warmup_epochs, [LayerGroup.CLASSIFIER_HEAD, LayerGroup.COLOR_FEATURES]),
            (self.config.warmup_epochs + 5, [LayerGroup.ATTENTION, LayerGroup.BACKBONE_LATE]),
            (self.config.warmup_epochs + 10, [LayerGroup.BACKBONE_MIDDLE]),
            (self.config.warmup_epochs + 15, [LayerGroup.BACKBONE_EARLY])
        ]
        
        for epoch, groups in schedule_points:
            if epoch < self.config.total_epochs:
                self.unfreezing_schedule[epoch] = groups
    
    def _generate_hybrid_schedule(self) -> None:
        """Generate hybrid unfreezing schedule balancing preservation and adaptation."""
        # Hybrid approach: Balanced unfreezing
        schedule_points = [
            (self.config.warmup_epochs, [LayerGroup.CLASSIFIER_HEAD]),
            (self.config.warmup_epochs + 8, [LayerGroup.COLOR_FEATURES, LayerGroup.ATTENTION]),
            (self.config.warmup_epochs + 18, [LayerGroup.BACKBONE_LATE]),
            (self.config.warmup_epochs + 30, [LayerGroup.BACKBONE_MIDDLE]),
            (self.config.warmup_epochs + 45, [LayerGroup.BACKBONE_EARLY])
        ]
        
        for epoch, groups in schedule_points:
            if epoch < self.config.total_epochs:
                self.unfreezing_schedule[epoch] = groups
    
    def update_epoch(self, epoch: int, performance_metric: float) -> bool:
        """
        Update current epoch and check for unfreezing triggers.
        
        Args:
            epoch: Current training epoch
            performance_metric: Current performance metric for safety checking
            
        Returns:
            True if unfreezing occurred, False otherwise
        """
        self.current_epoch = epoch
        self.performance_history.append(performance_metric)
        
        # Check if this epoch triggers unfreezing
        if epoch in self.unfreezing_schedule:
            groups_to_unfreeze = self.unfreezing_schedule[epoch]
            
            # Safety check: Ensure performance is improving
            if self._performance_safety_check():
                self._unfreeze_layer_groups(groups_to_unfreeze)
                return True
            else:
                logger.warning(f"Performance safety check failed at epoch {epoch}. "
                             f"Delaying unfreezing of {[g.value for g in groups_to_unfreeze]}")
                # Reschedule for next safety check interval
                next_epoch = epoch + self.config.safety_check_interval
                if next_epoch < self.config.total_epochs:
                    self.unfreezing_schedule[next_epoch] = groups_to_unfreeze
        
        return False
    
    def _performance_safety_check(self) -> bool:
        """
        Check if performance is improving enough to continue unfreezing.
        
        Returns:
            True if safe to unfreeze, False otherwise
        """
        if len(self.performance_history) < self.config.safety_check_interval:
            return True  # Not enough history, proceed
        
        # Check recent performance trend
        recent_performance = self.performance_history[-self.config.safety_check_interval:]
        performance_improvement = recent_performance[-1] - recent_performance[0]
        
        if performance_improvement >= self.config.min_improvement_threshold:
            logger.info(f"Performance safety check passed: improvement = {performance_improvement:.4f}")
            return True
        else:
            logger.warning(f"Performance safety check failed: improvement = {performance_improvement:.4f} "
                         f"< threshold = {self.config.min_improvement_threshold:.4f}")
            return False
    
    def _unfreeze_layer_groups(self, groups: List[LayerGroup]) -> None:
        """
        Unfreeze specified layer groups.
        
        Args:
            groups: List of layer groups to unfreeze
        """
        for group in groups:
            if group not in self.unfrozen_groups:
                self._unfreeze_group(group)
                self.unfrozen_groups.append(group)
                logger.info(f"✓ Unfroze layer group: {group.value}")
    
    def _unfreeze_group(self, group: LayerGroup) -> None:
        """
        Unfreeze a specific layer group.
        
        Args:
            group: Layer group to unfreeze
        """
        if group not in self.layer_groups:
            logger.warning(f"Layer group {group.value} not found in model")
            return
        
        for layer_name in self.layer_groups[group]:
            # Find and unfreeze the layer
            module = self._get_module_by_name(layer_name)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """
        Get module by name from the model.
        
        Args:
            name: Module name
            
        Returns:
            Module if found, None otherwise
        """
        try:
            parts = name.split('.')
            module = self.model
            for part in parts:
                module = getattr(module, part)
            return module
        except AttributeError:
            logger.warning(f"Module {name} not found in model")
            return None
    
    def create_optimizer_with_layer_groups(self, base_lr: float) -> optim.Optimizer:
        """
        Create optimizer with different learning rates for different layer groups.
        
        Args:
            base_lr: Base learning rate
            
        Returns:
            Configured optimizer with layer-specific learning rates
        """
        param_groups = []
        
        # Group parameters by unfreezing status and layer group
        for group in LayerGroup:
            if group in self.layer_groups and self.layer_groups[group]:
                group_params = []
                for layer_name in self.layer_groups[group]:
                    module = self._get_module_by_name(layer_name)
                    if module is not None:
                        group_params.extend(list(module.parameters()))
                
                if group_params:
                    # Determine learning rate based on group and frozen status
                    if group in self.unfrozen_groups:
                        lr = base_lr * self.config.unfrozen_lr_multiplier
                        # Apply layer depth decay
                        if group == LayerGroup.BACKBONE_EARLY:
                            lr *= self.config.layer_lr_decay ** 2
                        elif group == LayerGroup.BACKBONE_MIDDLE:
                            lr *= self.config.layer_lr_decay
                    else:
                        lr = base_lr * self.config.frozen_lr_multiplier
                    
                    param_groups.append({
                        'params': group_params,
                        'lr': lr,
                        'group_name': group.value
                    })
        
        # Add any remaining parameters
        all_grouped_params = set()
        for group in param_groups:
            all_grouped_params.update(id(p) for p in group['params'])
        
        remaining_params = [p for p in self.model.parameters() 
                          if id(p) not in all_grouped_params]
        
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': base_lr,
                'group_name': 'remaining'
            })
        
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
        
        logger.info(f"Created optimizer with {len(param_groups)} parameter groups:")
        for group in param_groups:
            logger.info(f"  {group['group_name']}: {len(group['params'])} params, lr={group['lr']:.2e}")
        
        return optimizer
    
    def get_unfreezing_status(self) -> Dict[str, Any]:
        """
        Get current unfreezing status and statistics.
        
        Returns:
            Dictionary with unfreezing status information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'current_epoch': self.current_epoch,
            'unfrozen_groups': [g.value for g in self.unfrozen_groups],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'performance_history': self.performance_history[-10:],  # Last 10 epochs
            'next_unfreeze_epoch': min([e for e in self.unfreezing_schedule.keys() 
                                      if e > self.current_epoch], default=None)
        }
    
    def freeze_all_layers(self) -> None:
        """Freeze all model parameters (useful for initialization)."""
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.unfrozen_groups = []
        logger.info("All model layers frozen")
    
    def unfreeze_all_layers(self) -> None:
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.unfrozen_groups = list(LayerGroup)
        logger.info("All model layers unfrozen")


def create_progressive_unfreezer(
    model: nn.Module,
    strategy: UnfreezingStrategy = UnfreezingStrategy.HYBRID,
    total_epochs: int = 100,
    device: Optional[torch.device] = None
) -> ProgressiveUnfreezer:
    """
    Create a progressive unfreezer with medical AI optimized configuration.
    
    Args:
        model: Model to apply progressive unfreezing to
        strategy: Unfreezing strategy to use
        total_epochs: Total training epochs
        device: Training device
        
    Returns:
        Configured progressive unfreezer
    """
    config = ProgressiveUnfreezingConfig(
        strategy=strategy,
        total_epochs=total_epochs
    )
    
    unfreezer = ProgressiveUnfreezer(
        model=model,
        config=config,
        device=device
    )
    
    logger.info(f"Created progressive unfreezer for medical domain adaptation with {strategy.value} strategy")
    return unfreezer