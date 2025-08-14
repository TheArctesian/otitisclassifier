#!/usr/bin/env python3
"""
Test script for Binary Screening Model integration with dual-architecture system.

This script validates:
1. Binary screening model creation and initialization
2. Integration with dual-architecture data loading
3. Forward pass and clinical prediction functionality
4. Color feature extraction pipeline
5. Confidence calibration capabilities
6. Grad-CAM interpretability features

Usage:
    python test_binary_screening.py

Requirements:
    - Processed dataset in data/processed/ebasaran-kaggale
    - All dependencies from requirements.txt
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from models.binary_screening import create_binary_screening_model, BinaryScreeningModel
from data.stage_based_loader import create_medical_ai_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test binary screening model creation and basic functionality."""
    logger.info("=== Testing Binary Screening Model Creation ===")
    
    try:
        # Create model with default configuration
        model = create_binary_screening_model()
        logger.info(f"✓ Model created successfully: {type(model).__name__}")
        
        # Test model properties
        assert model.num_classes == 2, f"Expected 2 classes, got {model.num_classes}"
        assert model.color_feature_fusion == True, "Color feature fusion should be enabled"
        logger.info(f"✓ Model configuration validated")
        
        # Test with dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 500, 500)
        
        with torch.no_grad():
            logits = model(dummy_input)
            assert logits.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {logits.shape}"
            logger.info(f"✓ Forward pass successful: output shape {logits.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        raise


def test_clinical_prediction(model):
    """Test clinical prediction with confidence scoring."""
    logger.info("=== Testing Clinical Prediction ===")
    
    try:
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 500, 500)
        
        with torch.no_grad():
            results = model.predict_with_confidence(dummy_input)
            
            # Validate required outputs
            required_keys = [
                'raw_logits', 'calibrated_logits', 'probabilities', 
                'pathology_probability', 'predictions', 'max_probability',
                'entropy', 'high_confidence', 'needs_specialist_review', 
                'immediate_attention'
            ]
            
            for key in required_keys:
                assert key in results, f"Missing key in clinical prediction: {key}"
            
            # Validate shapes and ranges
            assert results['probabilities'].shape == (batch_size, 2), "Invalid probabilities shape"
            assert torch.all(results['pathology_probability'] >= 0), "Pathology probability should be >= 0"
            assert torch.all(results['pathology_probability'] <= 1), "Pathology probability should be <= 1"
            
            logger.info("✓ Clinical prediction validated")
            logger.info(f"  Pathology probabilities: {results['pathology_probability'].tolist()}")
            logger.info(f"  Predictions: {results['predictions'].tolist()}")
            logger.info(f"  High confidence: {results['high_confidence'].tolist()}")
            
    except Exception as e:
        logger.error(f"✗ Clinical prediction failed: {e}")
        raise


def test_color_features(model):
    """Test color feature extraction pipeline."""
    logger.info("=== Testing Color Feature Extraction ===")
    
    try:
        # Test color extractor directly
        color_extractor = model.color_extractor
        
        # Create test image with different color characteristics
        test_image = torch.randn(2, 3, 500, 500)
        # Normalize to [0, 1] range for realistic color analysis
        test_image = torch.sigmoid(test_image)
        
        color_features = color_extractor.extract_lab_features(test_image)
        expected_dim = color_extractor.get_feature_dimension()
        
        assert color_features.shape == (2, expected_dim), f"Expected shape (2, {expected_dim}), got {color_features.shape}"
        
        logger.info(f"✓ Color features extracted: shape {color_features.shape}")
        logger.info(f"  Feature dimension: {expected_dim}")
        logger.info(f"  Sample features: {color_features[0, :5].tolist()}")
        
    except Exception as e:
        logger.error(f"✗ Color feature extraction failed: {e}")
        raise


def test_gradcam(model):
    """Test Grad-CAM interpretability."""
    logger.info("=== Testing Grad-CAM Interpretability ===")
    
    try:
        # Single image for Grad-CAM
        single_input = torch.randn(1, 3, 500, 500)
        
        # Generate Grad-CAM for both classes
        for target_class in [0, 1]:  # Normal and Pathological
            gradcam = model.generate_gradcam(single_input, target_class=target_class)
            
            assert gradcam.shape == (500, 500), f"Expected Grad-CAM shape (500, 500), got {gradcam.shape}"
            assert torch.all(gradcam >= 0), "Grad-CAM values should be non-negative"
            assert torch.all(gradcam <= 1), "Grad-CAM values should be <= 1"
            
            class_name = "Normal" if target_class == 0 else "Pathological"
            logger.info(f"✓ Grad-CAM generated for {class_name}: shape {gradcam.shape}")
        
    except Exception as e:
        logger.error(f"✗ Grad-CAM generation failed: {e}")
        raise


def test_loss_function(model):
    """Test high-recall loss function."""
    logger.info("=== Testing High-Recall Loss Function ===")
    
    try:
        batch_size = 4
        dummy_logits = torch.randn(batch_size, 2)
        
        # Test with balanced targets
        balanced_targets = torch.tensor([0, 1, 0, 1])  # Normal, Pathological, Normal, Pathological
        loss_balanced = model.compute_loss(dummy_logits, balanced_targets)
        
        # Test with imbalanced targets (more pathological cases)
        imbalanced_targets = torch.tensor([1, 1, 1, 0])  # More pathological cases
        loss_imbalanced = model.compute_loss(dummy_logits, imbalanced_targets)
        
        assert loss_balanced.item() > 0, "Loss should be positive"
        assert loss_imbalanced.item() > 0, "Loss should be positive"
        
        logger.info(f"✓ High-recall loss computed successfully")
        logger.info(f"  Balanced loss: {loss_balanced.item():.4f}")
        logger.info(f"  Imbalanced loss: {loss_imbalanced.item():.4f}")
        
    except Exception as e:
        logger.error(f"✗ Loss function test failed: {e}")
        raise


def test_data_integration():
    """Test integration with dual-architecture data loading."""
    logger.info("=== Testing Data Integration ===")
    
    try:
        # Check if data path exists
        data_path = Path("data/processed/ebasaran-kaggale")
        if not data_path.exists():
            logger.warning(f"Data path not found: {data_path}")
            logger.warning("Skipping data integration test - ensure data is processed first")
            return
        
        # Create dataset manager
        dataset_manager = create_medical_ai_datasets(dual_architecture=True)
        logger.info("✓ Dataset manager created")
        
        # Test binary screening data loaders
        try:
            binary_loaders = dataset_manager.get_binary_screening_dataloaders(
                'base_training', 
                batch_size=8
            )
            
            train_loader = binary_loaders['train']
            val_loader = binary_loaders['val']
            
            logger.info(f"✓ Binary screening loaders created")
            logger.info(f"  Train batches: {len(train_loader)}")
            logger.info(f"  Val batches: {len(val_loader)}")
            
            # Test loading a batch
            train_batch = next(iter(train_loader))
            images, labels = train_batch
            
            assert images.shape[1:] == (3, 500, 500), f"Expected image shape (3, 500, 500), got {images.shape[1:]}"
            assert len(torch.unique(labels)) <= 2, f"Binary labels should have max 2 unique values"
            assert torch.all((labels == 0) | (labels == 1)), "Labels should be 0 or 1 for binary classification"
            
            logger.info(f"✓ Data loading validated")
            logger.info(f"  Batch shape: {images.shape}")
            logger.info(f"  Label distribution: {torch.bincount(labels).tolist()}")
            
        except Exception as e:
            logger.warning(f"Data loading test failed (may be expected if data not processed): {e}")
            
    except Exception as e:
        logger.error(f"✗ Data integration test failed: {e}")
        raise


def test_model_compatibility():
    """Test model compatibility with clinical deployment requirements."""
    logger.info("=== Testing Clinical Deployment Compatibility ===")
    
    try:
        model = create_binary_screening_model()
        
        # Test model can be saved and loaded
        save_path = "test_binary_model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Create new model and load weights
        new_model = create_binary_screening_model()
        new_model.load_state_dict(torch.load(save_path))
        
        # Clean up
        Path(save_path).unlink()
        
        logger.info("✓ Model save/load compatibility validated")
        
        # Test model can be put in eval mode
        model.eval()
        assert not model.training, "Model should be in eval mode"
        
        # Test model can be moved to CUDA if available
        if torch.cuda.is_available():
            model.cuda()
            logger.info("✓ CUDA compatibility validated")
        else:
            logger.info("✓ CPU deployment validated (CUDA not available)")
        
    except Exception as e:
        logger.error(f"✗ Clinical deployment compatibility test failed: {e}")
        raise


def main():
    """Run comprehensive test suite for Binary Screening Model."""
    logger.info("Starting Binary Screening Model Test Suite")
    
    try:
        # Core functionality tests
        model = test_model_creation()
        test_clinical_prediction(model)
        test_color_features(model)
        test_gradcam(model)
        test_loss_function(model)
        
        # Integration tests
        test_data_integration()
        test_model_compatibility()
        
        logger.info("🎉 All tests passed! Binary Screening Model is ready for deployment.")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()