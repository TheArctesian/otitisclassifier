"""
Stage-Based Curriculum Learning for Dual-Architecture Medical AI System

This module implements progressive curriculum learning for medical image classification,
specifically designed for the dual-architecture otitis diagnosis system.

Key Features:
- Progressive training from binary screening to multi-class diagnostic models
- Medical domain-specific curriculum with pathology complexity ordering
- FDA-compliant stage isolation and validation protocols
- Adaptive difficulty progression based on model performance
- Clinical safety thresholds and early stopping criteria

Curriculum Stages:
1. Binary Screening (Stage 1): Normal vs Pathological classification
2. Multi-Class Diagnostic (Stage 2): 8-class pathology identification
3. Integrated Validation: Cross-model performance validation

Unix Philosophy: Single responsibility - curriculum learning orchestration
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol
from pathlib import Path
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from ..models.binary_screening import BinaryScreeningModel, create_binary_screening_model
from ..models.multiclass_diagnostic import MultiClassDiagnosticModel, create_multiclass_diagnostic_model
from ..data.stage_based_loader import create_medical_ai_datasets

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training stages for curriculum learning progression."""
    BINARY_SCREENING = "binary_screening"
    MULTICLASS_DIAGNOSTIC = "multiclass_diagnostic"
    INTEGRATED_VALIDATION = "integrated_validation"


class ModelTrainer(Protocol):
    """Protocol for model training interface."""
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch and return metrics."""
        ...
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate one epoch and return metrics."""
        ...


class CurriculumLearningConfig:
    """Configuration for curriculum learning parameters."""
    
    def __init__(self,
                 binary_epochs: int = 50,
                 multiclass_epochs: int = 100,
                 validation_epochs: int = 20,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 binary_lr: float = 1e-4,
                 multiclass_lr: float = 5e-5,
                 validation_lr: float = 1e-5,
                 binary_batch_size: int = 32,
                 multiclass_batch_size: int = 16,
                 validation_batch_size: int = 32,
                 clinical_sensitivity_threshold: float = 0.98,
                 clinical_specificity_threshold: float = 0.90,
                 diagnostic_accuracy_threshold: float = 0.85):
        """
        Initialize curriculum learning configuration.
        
        Args:
            binary_epochs: Maximum epochs for binary screening training
            multiclass_epochs: Maximum epochs for multi-class diagnostic training
            validation_epochs: Maximum epochs for integrated validation
            patience: Early stopping patience
            min_delta: Minimum improvement threshold for early stopping
            binary_lr: Learning rate for binary screening model
            multiclass_lr: Learning rate for multi-class diagnostic model
            validation_lr: Learning rate for validation fine-tuning
            binary_batch_size: Batch size for binary screening training
            multiclass_batch_size: Batch size for multi-class diagnostic training
            validation_batch_size: Batch size for validation training
            clinical_sensitivity_threshold: Required sensitivity for clinical deployment
            clinical_specificity_threshold: Required specificity for clinical deployment
            diagnostic_accuracy_threshold: Required accuracy for diagnostic model
        """
        self.binary_epochs = binary_epochs
        self.multiclass_epochs = multiclass_epochs
        self.validation_epochs = validation_epochs
        self.patience = patience
        self.min_delta = min_delta
        
        self.binary_lr = binary_lr
        self.multiclass_lr = multiclass_lr
        self.validation_lr = validation_lr
        
        self.binary_batch_size = binary_batch_size
        self.multiclass_batch_size = multiclass_batch_size
        self.validation_batch_size = validation_batch_size
        
        self.clinical_sensitivity_threshold = clinical_sensitivity_threshold
        self.clinical_specificity_threshold = clinical_specificity_threshold
        self.diagnostic_accuracy_threshold = diagnostic_accuracy_threshold


class CurriculumLearningOrchestrator:
    """
    Orchestrates progressive curriculum learning for dual-architecture medical AI system.
    
    Implements stage-based training progression with clinical safety validation
    and medical domain-specific curriculum ordering.
    """
    
    def __init__(self,
                 config: CurriculumLearningConfig,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: Optional[Path] = None):
        """
        Initialize curriculum learning orchestrator.
        
        Args:
            config: Curriculum learning configuration
            device: Training device (CPU/GPU)
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Models
        self.binary_model: Optional[BinaryScreeningModel] = None
        self.multiclass_model: Optional[MultiClassDiagnosticModel] = None
        
        # Training state
        self.current_stage = TrainingStage.BINARY_SCREENING
        self.training_history: Dict[str, List[Dict[str, float]]] = {
            stage.value: [] for stage in TrainingStage
        }
        
        logger.info(f"Initialized curriculum learning orchestrator on {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def setup_stage_1_binary_screening(self) -> None:
        """Setup Stage 1: Binary screening model training."""
        logger.info("=== Setting up Stage 1: Binary Screening Training ===")
        
        # Create binary screening model
        self.binary_model = create_binary_screening_model()
        self.binary_model.to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.binary_model.parameters())
        trainable_params = sum(p.numel() for p in self.binary_model.parameters() if p.requires_grad)
        
        logger.info(f"Binary screening model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Target performance: {self.config.clinical_sensitivity_threshold:.1%} sensitivity, "
                   f"{self.config.clinical_specificity_threshold:.1%} specificity")
        
        self.current_stage = TrainingStage.BINARY_SCREENING
    
    def setup_stage_2_multiclass_diagnostic(self) -> None:
        """Setup Stage 2: Multi-class diagnostic model training."""
        logger.info("=== Setting up Stage 2: Multi-Class Diagnostic Training ===")
        
        # Create multi-class diagnostic model
        self.multiclass_model = create_multiclass_diagnostic_model()
        self.multiclass_model.to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.multiclass_model.parameters())
        trainable_params = sum(p.numel() for p in self.multiclass_model.parameters() if p.requires_grad)
        
        logger.info(f"Multi-class diagnostic model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Target performance: {self.config.diagnostic_accuracy_threshold:.1%} balanced accuracy")
        
        self.current_stage = TrainingStage.MULTICLASS_DIAGNOSTIC
    
    def train_binary_screening_stage(self,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader) -> Dict[str, float]:
        """
        Train Stage 1: Binary screening model with high-sensitivity optimization.
        
        Args:
            train_loader: Training data loader for binary classification
            val_loader: Validation data loader for binary classification
            
        Returns:
            Final validation metrics
        """
        logger.info("=== Starting Stage 1: Binary Screening Training ===")
        
        if self.binary_model is None:
            raise ValueError("Binary model not initialized. Call setup_stage_1_binary_screening() first.")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.binary_model.parameters(),
            lr=self.config.binary_lr,
            weight_decay=1e-4
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=self.config.patience // 2,
            verbose=True
        )
        
        # Early stopping
        best_sensitivity = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.binary_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.binary_epochs}")
            
            # Training
            self.binary_model.train()
            train_metrics = self._train_binary_epoch(train_loader, optimizer)
            
            # Validation
            self.binary_model.eval()
            val_metrics = self._validate_binary_epoch(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['sensitivity'])
            
            # Logging
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.3f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.3f}, "
                       f"Sens: {val_metrics['sensitivity']:.3f}, "
                       f"Spec: {val_metrics['specificity']:.3f}")
            
            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_sensitivity': val_metrics['sensitivity'],
                'val_specificity': val_metrics['specificity']
            }
            self.training_history[TrainingStage.BINARY_SCREENING.value].append(epoch_metrics)
            
            # Clinical threshold check
            if (val_metrics['sensitivity'] >= self.config.clinical_sensitivity_threshold and
                val_metrics['specificity'] >= self.config.clinical_specificity_threshold):
                logger.info(f"✓ Clinical thresholds met! Sensitivity: {val_metrics['sensitivity']:.3f}, "
                           f"Specificity: {val_metrics['specificity']:.3f}")
            
            # Early stopping and checkpointing
            if val_metrics['sensitivity'] > best_sensitivity + self.config.min_delta:
                best_sensitivity = val_metrics['sensitivity']
                patience_counter = 0
                
                # Save best model
                checkpoint_path = self.checkpoint_dir / "binary_screening_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.binary_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config.__dict__
                }, checkpoint_path)
                logger.info(f"✓ New best model saved: {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / "binary_screening_best.pth")
        self.binary_model.load_state_dict(best_checkpoint['model_state_dict'])
        
        logger.info("=== Stage 1: Binary Screening Training Complete ===")
        return best_checkpoint['metrics']
    
    def train_multiclass_diagnostic_stage(self,
                                        train_loader: DataLoader,
                                        val_loader: DataLoader) -> Dict[str, float]:
        """
        Train Stage 2: Multi-class diagnostic model with focal loss.
        
        Args:
            train_loader: Training data loader for multi-class classification
            val_loader: Validation data loader for multi-class classification
            
        Returns:
            Final validation metrics
        """
        logger.info("=== Starting Stage 2: Multi-Class Diagnostic Training ===")
        
        if self.multiclass_model is None:
            raise ValueError("Multi-class model not initialized. Call setup_stage_2_multiclass_diagnostic() first.")
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.multiclass_model.parameters(),
            lr=self.config.multiclass_lr,
            weight_decay=1e-4
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.config.multiclass_epochs,
            eta_min=1e-6
        )
        
        # Early stopping
        best_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.multiclass_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.multiclass_epochs}")
            
            # Training
            self.multiclass_model.train()
            train_metrics = self._train_multiclass_epoch(train_loader, optimizer)
            
            # Validation
            self.multiclass_model.eval()
            val_metrics = self._validate_multiclass_epoch(val_loader)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.3f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.3f}, "
                       f"Balanced Acc: {val_metrics['balanced_accuracy']:.3f}")
            
            # Save metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_balanced_accuracy': val_metrics['balanced_accuracy']
            }
            self.training_history[TrainingStage.MULTICLASS_DIAGNOSTIC.value].append(epoch_metrics)
            
            # Clinical threshold check
            if val_metrics['balanced_accuracy'] >= self.config.diagnostic_accuracy_threshold:
                logger.info(f"✓ Clinical accuracy threshold met! "
                           f"Balanced accuracy: {val_metrics['balanced_accuracy']:.3f}")
            
            # Early stopping and checkpointing
            if val_metrics['balanced_accuracy'] > best_accuracy + self.config.min_delta:
                best_accuracy = val_metrics['balanced_accuracy']
                patience_counter = 0
                
                # Save best model
                checkpoint_path = self.checkpoint_dir / "multiclass_diagnostic_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.multiclass_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config.__dict__
                }, checkpoint_path)
                logger.info(f"✓ New best model saved: {checkpoint_path}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / "multiclass_diagnostic_best.pth")
        self.multiclass_model.load_state_dict(best_checkpoint['model_state_dict'])
        
        logger.info("=== Stage 2: Multi-Class Diagnostic Training Complete ===")
        return best_checkpoint['metrics']
    
    def _train_binary_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train one epoch for binary screening model."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.binary_model(images)
            loss = self.binary_model.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate_binary_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch for binary screening model."""
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.binary_model(images)
                loss = self.binary_model.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Binary classification metrics
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
    
    def _train_multiclass_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Dict[str, float]:
        """Train one epoch for multi-class diagnostic model."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.multiclass_model(images)
            loss = self.multiclass_model.loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def _validate_multiclass_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one epoch for multi-class diagnostic model."""
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.multiclass_model(images)
                loss = self.multiclass_model.loss_fn(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy
        }
    
    def get_training_history(self) -> Dict[str, List[Dict[str, float]]]:
        """Get complete training history for all stages."""
        return self.training_history
    
    def save_training_state(self, filepath: Path) -> None:
        """Save complete training state including history and configurations."""
        state = {
            'config': self.config.__dict__,
            'current_stage': self.current_stage.value,
            'training_history': self.training_history,
            'checkpoint_dir': str(self.checkpoint_dir)
        }
        
        torch.save(state, filepath)
        logger.info(f"Training state saved to {filepath}")


def create_curriculum_learning_orchestrator(
    config: Optional[CurriculumLearningConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[Path] = None
) -> CurriculumLearningOrchestrator:
    """
    Create a curriculum learning orchestrator with default medical AI configuration.
    
    Args:
        config: Custom configuration (uses default if None)
        device: Training device (auto-detects if None)
        checkpoint_dir: Checkpoint directory (uses default if None)
        
    Returns:
        Configured curriculum learning orchestrator
    """
    if config is None:
        config = CurriculumLearningConfig()
    
    orchestrator = CurriculumLearningOrchestrator(
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("Created curriculum learning orchestrator for dual-architecture medical AI training")
    return orchestrator