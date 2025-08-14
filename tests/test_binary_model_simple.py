#!/usr/bin/env python3
"""
Simple standalone test for Binary Screening Model.

This test validates the core Binary Screening Model functionality without
requiring the full data loading infrastructure.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_imports():
    """Test that we can import the binary screening model."""
    logger.info("=== Testing Model Imports ===")
    
    try:
        from models.binary_screening import create_binary_screening_model, BinaryScreeningModel
        from models.binary_screening import ColorFeatureExtractor, TemperatureScaling, HighRecallLoss
        logger.info("✓ All binary screening model imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_color_feature_extractor():
    """Test the color feature extraction pipeline."""
    logger.info("=== Testing Color Feature Extractor ===")
    
    try:
        from models.binary_screening import ColorFeatureExtractor
        
        # Create extractor
        extractor = ColorFeatureExtractor(spatial_bins=8)
        
        # Test with dummy image
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 500, 500)
        # Normalize to [0, 1] for realistic color processing
        dummy_images = torch.sigmoid(dummy_images)
        
        # Extract features
        features = extractor.extract_lab_features(dummy_images)
        expected_dim = extractor.get_feature_dimension()
        
        assert features.shape == (batch_size, expected_dim), f"Expected shape ({batch_size}, {expected_dim}), got {features.shape}"
        
        logger.info(f"✓ Color features extracted successfully")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Feature dimension: {expected_dim}")
        logger.info(f"  Sample features: {features[0, :5].tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Color feature extraction failed: {e}")
        return False


def test_high_recall_loss():
    """Test the high-recall loss function."""
    logger.info("=== Testing High-Recall Loss Function ===")
    
    try:
        from models.binary_screening import HighRecallLoss
        
        # Create loss function
        loss_fn = HighRecallLoss(
            alpha=1.0,
            gamma=2.0,
            false_negative_penalty=10.0
        )
        
        # Test with dummy data
        batch_size = 4
        dummy_logits = torch.randn(batch_size, 2)
        dummy_targets = torch.tensor([0, 1, 0, 1])  # Binary targets
        
        # Compute loss
        loss = loss_fn(dummy_logits, dummy_targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        logger.info(f"✓ High-recall loss computed successfully")
        logger.info(f"  Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ High-recall loss test failed: {e}")
        return False


def test_temperature_scaling():
    """Test temperature scaling for confidence calibration."""
    logger.info("=== Testing Temperature Scaling ===")
    
    try:
        from models.binary_screening import TemperatureScaling
        
        # Create temperature scaling module
        temp_scaling = TemperatureScaling()
        
        # Test with dummy logits
        batch_size = 10
        dummy_logits = torch.randn(batch_size, 2)
        dummy_labels = torch.randint(0, 2, (batch_size,))
        
        # Apply temperature scaling
        scaled_logits = temp_scaling(dummy_logits)
        
        assert scaled_logits.shape == dummy_logits.shape, "Temperature scaling should preserve shape"
        
        # Test calibration (simplified)
        initial_temp = temp_scaling.temperature.item()
        temp_scaling.calibrate(dummy_logits, dummy_labels, lr=0.1, max_iter=10)
        final_temp = temp_scaling.temperature.item()
        
        logger.info(f"✓ Temperature scaling successful")
        logger.info(f"  Initial temperature: {initial_temp:.4f}")
        logger.info(f"  Calibrated temperature: {final_temp:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Temperature scaling test failed: {e}")
        return False


def test_model_creation():
    """Test binary screening model creation."""
    logger.info("=== Testing Model Creation ===")
    
    try:
        from models.binary_screening import create_binary_screening_model
        
        # Create model with clinical configuration
        clinical_config = {
            'pretrained': True,
            'radimagenet_weights': True,
            'color_feature_fusion': True,
            'dropout_rate': 0.3,
            'confidence_threshold': 0.5
        }
        
        model = create_binary_screening_model(clinical_config)
        
        # Validate model properties
        assert model.num_classes == 2, f"Expected 2 classes, got {model.num_classes}"
        assert model.color_feature_fusion == True, "Color feature fusion should be enabled"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Model created successfully")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Color feature fusion: {model.color_feature_fusion}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return None


def test_model_forward_pass(model):
    """Test model forward pass."""
    logger.info("=== Testing Model Forward Pass ===")
    
    try:
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 500, 500)
        
        with torch.no_grad():
            # Standard forward pass
            logits = model(dummy_input)
            assert logits.shape == (batch_size, 2), f"Expected logits shape ({batch_size}, 2), got {logits.shape}"
            
            # Clinical prediction with confidence
            results = model.predict_with_confidence(dummy_input)
            
            # Validate results structure
            required_keys = [
                'raw_logits', 'calibrated_logits', 'probabilities', 
                'pathology_probability', 'predictions', 'max_probability',
                'entropy', 'high_confidence', 'needs_specialist_review', 
                'immediate_attention'
            ]
            
            for key in required_keys:
                assert key in results, f"Missing key in clinical prediction: {key}"
            
            logger.info(f"✓ Forward pass successful")
            logger.info(f"  Logits shape: {logits.shape}")
            logger.info(f"  Pathology probabilities: {results['pathology_probability'].tolist()}")
            logger.info(f"  Predictions: {results['predictions'].tolist()}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return False


def test_gradcam_generation(model):
    """Test Grad-CAM generation."""
    logger.info("=== Testing Grad-CAM Generation ===")
    
    try:
        # Single image for Grad-CAM
        single_input = torch.randn(1, 3, 500, 500)
        
        # Generate Grad-CAM
        gradcam = model.generate_gradcam(single_input, target_class=1)  # Pathological class
        
        assert gradcam.shape == (500, 500), f"Expected Grad-CAM shape (500, 500), got {gradcam.shape}"
        assert torch.all(gradcam >= 0), "Grad-CAM values should be non-negative"
        assert torch.all(gradcam <= 1), "Grad-CAM values should be <= 1"
        
        logger.info(f"✓ Grad-CAM generated successfully")
        logger.info(f"  Shape: {gradcam.shape}")
        logger.info(f"  Value range: [{gradcam.min():.4f}, {gradcam.max():.4f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Grad-CAM generation failed: {e}")
        return False


def test_loss_computation(model):
    """Test loss computation."""
    logger.info("=== Testing Loss Computation ===")
    
    try:
        batch_size = 8
        dummy_logits = torch.randn(batch_size, 2)
        dummy_targets = torch.randint(0, 2, (batch_size,))
        
        # Compute loss
        loss = model.compute_loss(dummy_logits, dummy_targets)
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        logger.info(f"✓ Loss computation successful")
        logger.info(f"  Loss value: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Loss computation failed: {e}")
        return False


def test_model_save_load(model):
    """Test model save and load functionality."""
    logger.info("=== Testing Model Save/Load ===")
    
    try:
        # Save model
        save_path = "test_binary_model_temp.pt"
        torch.save(model.state_dict(), save_path)
        
        # Create new model and load weights
        from models.binary_screening import create_binary_screening_model
        
        new_model = create_binary_screening_model()
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that loaded model works
        dummy_input = torch.randn(2, 3, 500, 500)
        
        with torch.no_grad():
            original_output = model(dummy_input)
            loaded_output = new_model(dummy_input)
            
            # Should be very close (not exact due to floating point)
            diff = torch.abs(original_output - loaded_output).max()
            assert diff < 1e-5, f"Loaded model output differs too much: {diff}"
        
        # Clean up
        Path(save_path).unlink()
        
        logger.info(f"✓ Model save/load successful")
        logger.info(f"  Max difference: {diff:.2e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model save/load failed: {e}")
        return False


def main():
    """Run the complete test suite."""
    logger.info("🏥 Binary Screening Model - Simple Test Suite")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: Imports
    if test_model_imports():
        tests_passed += 1
    
    # Test 2: Color feature extractor
    if test_color_feature_extractor():
        tests_passed += 1
    
    # Test 3: High-recall loss
    if test_high_recall_loss():
        tests_passed += 1
    
    # Test 4: Temperature scaling
    if test_temperature_scaling():
        tests_passed += 1
    
    # Test 5: Model creation
    model = test_model_creation()
    if model is not None:
        tests_passed += 1
        
        # Test 6: Forward pass
        if test_model_forward_pass(model):
            tests_passed += 1
        
        # Test 7: Grad-CAM
        if test_gradcam_generation(model):
            tests_passed += 1
        
        # Test 8: Loss computation
        if test_loss_computation(model):
            tests_passed += 1
        
        # Test 9: Save/Load (bonus test)
        if test_model_save_load(model):
            logger.info("✓ Bonus test: Model save/load successful")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Binary Screening Model is working correctly.")
        logger.info("The model is ready for integration with the dual-architecture system.")
    else:
        logger.error(f"❌ {total_tests - tests_passed} tests failed. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()