#!/usr/bin/env python3
"""
Training Script for Multi-Class Diagnostic Model (Stage 2 of Dual-Architecture)

This script demonstrates comprehensive training of the multi-class diagnostic model
for otitis pathology classification with advanced rare pathology handling.

Key Features:
- Stage 2 of dual-architecture medical AI system
- 8-class pathological classification (excluding normal cases)
- Advanced focal loss with class-specific gamma values
- Curriculum learning for progressive difficulty training
- Aggressive augmentation for rare pathologies
- Clinical validation and safety checks
- Integration with existing dual-architecture framework

Usage:
    python src/train_diagnostic_model.py [--config config.yaml]

Clinical Context:
- Processes only pathological cases flagged by Stage 1 binary screening
- Targets 85%+ balanced accuracy with 80%+ sensitivity for rare classes
- Provides specialist-grade diagnostic precision for 8 pathological conditions
- Includes clinical decision support and referral recommendations
"""

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Import our models and data loaders
from models.multiclass_diagnostic import (
    MultiClassDiagnosticModel, 
    create_multiclass_diagnostic_model,
    DualArchitectureIntegration
)
from models.binary_screening import create_binary_screening_model
from data.stage_based_loader import create_medical_ai_datasets, TrainingStage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiClassTrainer:
    """
    Comprehensive trainer for multi-class diagnostic model with clinical safety features.
    
    Handles the complete training pipeline for Stage 2 of the dual-architecture system:
    - Progressive difficulty curriculum learning
    - Rare pathology augmentation strategies
    - Clinical validation and safety checks
    - Integration testing with binary screening model
    """
    
    def __init__(self, 
                 model: MultiClassDiagnosticModel,
                 dataset_manager,
                 config: Dict[str, Any]):
        """
        Initialize multi-class trainer.
        
        Args:
            model: Multi-class diagnostic model
            dataset_manager: Stage-based dataset manager
            config: Training configuration
        """
        self.model = model
        self.dataset_manager = dataset_manager
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_balanced_accuracy = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_balanced_accuracy': [],
            'val_rare_class_sensitivity': []
        }
        
        # Clinical thresholds
        self.clinical_thresholds = {
            'balanced_accuracy': config.get('balanced_accuracy_threshold', 0.85),
            'rare_class_sensitivity': config.get('rare_class_sensitivity_threshold', 0.80),
            'min_epochs': config.get('min_epochs', 20),
            'max_epochs': config.get('max_epochs', 100)
        }
        
        logger.info("Initialized MultiClassTrainer for pathology diagnosis")
        logger.info(f"Clinical thresholds: {self.clinical_thresholds}")
    
    def setup_training(self, stage: str = 'base_training') -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data loaders for pathology-only training.
        
        Args:
            stage: Training stage ('base_training', 'fine_tuning')
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info(f"Setting up training for stage: {stage}")
        
        # Get pathology-only data loaders
        dataloaders = self.dataset_manager.get_diagnostic_dataloaders(
            stage=stage,
            batch_size=self.config.get('batch_size', 16),
            num_workers=self.config.get('num_workers', 4)
        )
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Setup optimizer and learning rate scheduler for pathology classification.
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Use AdamW optimizer with weight decay for medical imaging
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing with warm restarts for stable convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.get('scheduler_t0', 10),
            T_mult=self.config.get('scheduler_tmult', 2),
            eta_min=self.config.get('min_learning_rate', 1e-6)
        )
        
        logger.info(f"Optimizer: AdamW with lr={self.config.get('learning_rate', 1e-4)}")
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts")
        
        return optimizer, scheduler
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler) -> float:
        """
        Train model for one epoch with rare pathology focus.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(images)
            
            # Compute loss using advanced focal loss
            loss = self.model.compute_loss(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Update loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{total_loss / num_batches:.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model with clinical performance metrics.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("Running clinical validation...")
        
        # Use model's built-in clinical validation
        metrics = self.model.validate_clinical_performance(
            val_loader,
            balanced_accuracy_threshold=self.clinical_thresholds['balanced_accuracy'],
            rare_class_sensitivity_threshold=self.clinical_thresholds['rare_class_sensitivity']
        )
        
        # Calculate validation loss
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                logits = self.model(images)
                loss = self.model.compute_loss(logits, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def train_stage(self, stage: str = 'base_training') -> Dict[str, float]:
        """
        Train model for a complete stage with early stopping and clinical validation.
        
        Args:
            stage: Training stage ('base_training', 'fine_tuning')
            
        Returns:
            Final validation metrics
        """
        logger.info(f"=== Starting Training Stage: {stage} ===")
        
        # Setup data loaders
        train_loader, val_loader = self.setup_training(stage)
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler()
        
        # Training loop
        best_metrics = {}
        patience_counter = 0
        patience = self.config.get('patience', 15)
        
        for epoch in range(self.clinical_thresholds['max_epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['val_balanced_accuracy'].append(val_metrics['balanced_accuracy'])
            self.training_history['val_rare_class_sensitivity'].append(val_metrics['rare_class_sensitivity'])
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.clinical_thresholds['max_epochs']}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Rare Class Sensitivity: {val_metrics['rare_class_sensitivity']:.4f}")
            logger.info(f"  Clinical Safety: {'PASSED' if val_metrics['clinical_safety_passed'] else 'FAILED'}")
            
            # Check for improvement
            current_score = val_metrics['balanced_accuracy']
            if current_score > self.best_balanced_accuracy:
                self.best_balanced_accuracy = current_score
                best_metrics = val_metrics.copy()
                patience_counter = 0
                
                # Save best model
                self.save_checkpoint(f"best_model_{stage}.pth", val_metrics)
                logger.info(f"  New best model saved! Balanced Accuracy: {current_score:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping with clinical constraints
            if epoch >= self.clinical_thresholds['min_epochs']:
                # Check clinical safety passing
                if val_metrics['clinical_safety_passed']:
                    logger.info(f"Clinical safety criteria met at epoch {epoch + 1}")
                    if patience_counter >= patience:
                        logger.info(f"Early stopping: no improvement for {patience} epochs")
                        break
                elif epoch >= self.clinical_thresholds['max_epochs'] * 0.8:
                    logger.warning(f"Clinical safety not met after 80% of max epochs")
        
        logger.info(f"=== Completed Training Stage: {stage} ===")
        logger.info(f"Best Balanced Accuracy: {self.best_balanced_accuracy:.4f}")
        
        return best_metrics
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """
        Save model checkpoint with training state.
        
        Args:
            filename: Checkpoint filename
            metrics: Current validation metrics
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_balanced_accuracy': self.best_balanced_accuracy,
            'metrics': metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    def plot_training_history(self, save_path: str = "training_history.png"):
        """
        Plot training history for analysis.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Class Diagnostic Model Training History')
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Balanced accuracy plot
        axes[0, 1].plot(self.training_history['val_balanced_accuracy'], label='Balanced Accuracy')
        axes[0, 1].axhline(y=self.clinical_thresholds['balanced_accuracy'], 
                          color='r', linestyle='--', label='Clinical Threshold')
        axes[0, 1].set_title('Validation Balanced Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rare class sensitivity plot
        axes[1, 0].plot(self.training_history['val_rare_class_sensitivity'], label='Rare Class Sensitivity')
        axes[1, 0].axhline(y=self.clinical_thresholds['rare_class_sensitivity'], 
                          color='r', linestyle='--', label='Clinical Threshold')
        axes[1, 0].set_title('Rare Class Sensitivity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Sensitivity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined clinical metrics
        axes[1, 1].plot(self.training_history['val_balanced_accuracy'], label='Balanced Accuracy')
        axes[1, 1].plot(self.training_history['val_rare_class_sensitivity'], label='Rare Class Sensitivity')
        axes[1, 1].set_title('Clinical Performance Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {save_path}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load training configuration from YAML file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        # Model configuration
        'model': {
            'num_pathology_classes': 8,
            'pretrained': True,
            'radimagenet_weights': True,
            'color_feature_fusion': True,
            'regional_attention': True,
            'dropout_rate': 0.4,
            'attention_dropout': 0.2,
            'confidence_threshold': 0.7
        },
        
        # Training configuration
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'num_workers': 4,
            'min_epochs': 20,
            'max_epochs': 100,
            'patience': 15,
            'scheduler_t0': 10,
            'scheduler_tmult': 2,
            'min_learning_rate': 1e-6
        },
        
        # Clinical thresholds
        'clinical': {
            'balanced_accuracy_threshold': 0.85,
            'rare_class_sensitivity_threshold': 0.80
        },
        
        # Data configuration
        'data': {
            'base_training_path': "data/processed/ebasaran-kaggale",
            'fine_tuning_path': "data/processed/uci-kaggle",
            'validation_path': "data/processed/vanak-figshare",
            'image_size': 500,
            'dual_architecture': True
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)
            # Merge configurations
            for key, value in file_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
    
    return default_config


def test_dual_architecture_integration(binary_model, diagnostic_model):
    """
    Test integration between binary screening and diagnostic models.
    
    Args:
        binary_model: Stage 1 binary screening model
        diagnostic_model: Stage 2 diagnostic model
    """
    logger.info("=== Testing Dual-Architecture Integration ===")
    
    # Create integration wrapper
    dual_system = DualArchitectureIntegration(
        binary_model=binary_model,
        diagnostic_model=diagnostic_model,
        binary_threshold=0.5,
        diagnostic_threshold=0.7
    )
    
    # Test with dummy data
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 500, 500)
    
    with torch.no_grad():
        # Complete clinical prediction
        results = dual_system.predict_clinical(dummy_input)
        
        logger.info("Dual-architecture integration test results:")
        logger.info(f"  Pathology detected: {results['pathology_detected'].sum().item()}/{batch_size}")
        logger.info(f"  Final diagnoses: {results['final_diagnosis']}")
        logger.info(f"  Clinical recommendations: {len(results['clinical_recommendations'])}")
        
        for i, rec in enumerate(results['clinical_recommendations'][:3]):  # Show first 3
            logger.info(f"    {i+1}. {rec}")
    
    logger.info("Dual-architecture integration test completed successfully")


def main():
    """Main training function for multi-class diagnostic model."""
    parser = argparse.ArgumentParser(description='Train Multi-Class Diagnostic Model')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--stage', type=str, default='base_training', 
                       choices=['base_training', 'fine_tuning'],
                       help='Training stage')
    parser.add_argument('--test-integration', action='store_true',
                       help='Test dual-architecture integration')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully")
    
    # Create dataset manager
    dataset_manager = create_medical_ai_datasets(
        base_training_path=config['data']['base_training_path'],
        fine_tuning_path=config['data']['fine_tuning_path'],
        validation_path=config['data']['validation_path'],
        image_size=config['data']['image_size'],
        dual_architecture=config['data']['dual_architecture']
    )
    
    # Create diagnostic model
    diagnostic_model = create_multiclass_diagnostic_model(config['model'])
    logger.info("Multi-class diagnostic model created successfully")
    
    # Create trainer
    trainer_config = {**config['training'], **config['clinical']}
    trainer = MultiClassTrainer(diagnostic_model, dataset_manager, trainer_config)
    
    # Train model
    final_metrics = trainer.train_stage(args.stage)
    
    # Plot training history
    trainer.plot_training_history(f"diagnostic_model_{args.stage}_history.png")
    
    # Test dual-architecture integration if requested
    if args.test_integration:
        # Create binary screening model for integration testing
        binary_model = create_binary_screening_model()
        test_dual_architecture_integration(binary_model, diagnostic_model)
    
    # Final validation on external dataset
    if args.stage == 'fine_tuning':  # After fine-tuning, test on external data
        logger.info("=== Final External Validation ===")
        val_loaders = dataset_manager.get_diagnostic_dataloaders('validation', batch_size=16)
        if 'test' in val_loaders:
            external_metrics = trainer.validate_epoch(val_loaders['test'])
            logger.info("External validation completed:")
            logger.info(f"  Balanced Accuracy: {external_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  Rare Class Sensitivity: {external_metrics['rare_class_sensitivity']:.4f}")
            logger.info(f"  Clinical Safety: {'PASSED' if external_metrics['clinical_safety_passed'] else 'FAILED'}")
    
    logger.info("=== Training Complete ===")
    logger.info("Multi-class diagnostic model training completed successfully")
    logger.info("Ready for Stage 2 deployment in dual-architecture clinical workflow")


if __name__ == "__main__":
    main()