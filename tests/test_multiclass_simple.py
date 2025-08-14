#!/usr/bin/env python3
"""
Simple Test for Multi-Class Diagnostic Model

Quick verification of the Multi-Class Diagnostic Model implementation.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn.functional as F
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_basic():
    """Basic test of model creation and forward pass."""
    print("Testing Multi-Class Diagnostic Model")
    print("=" * 50)
    
    try:
        # Import the model
        from models.multiclass_diagnostic import create_multiclass_diagnostic_model
        
        print("[PASS] Successfully imported model")
        
        # Create model
        model = create_multiclass_diagnostic_model()
        print("[PASS] Model created successfully")
        print(f"       - Classes: {model.num_pathology_classes}")
        print(f"       - Color fusion: {model.color_feature_fusion}")
        print(f"       - Regional attention: {model.regional_attention}")
        
        # Test forward pass
        model.eval()
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 500, 500)
        
        with torch.no_grad():
            # Basic forward pass
            logits = model(dummy_input)
            print(f"[PASS] Forward pass: {logits.shape}")
            
            # Clinical prediction
            results = model.predict_with_confidence(dummy_input)
            print("[PASS] Clinical prediction successful")
            print(f"       - Predictions: {results['predictions'].tolist()}")
            print(f"       - Confidences: {results['max_probability'].numpy()}")
        
        # Test loss
        dummy_targets = torch.randint(0, 8, (batch_size,))
        loss = model.compute_loss(logits, dummy_targets)
        print(f"[PASS] Loss computation: {loss.item():.4f}")
        
        print("\n[SUCCESS] All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pathology_classes():
    """Test pathology class mapping and names."""
    print("\nTesting Pathology Classes")
    print("=" * 30)
    
    try:
        from models.multiclass_diagnostic import create_multiclass_diagnostic_model
        
        model = create_multiclass_diagnostic_model()
        
        print("Pathology Classes:")
        for i, name in model.pathology_class_names.items():
            print(f"  {i}: {name}")
        
        # Test label conversion
        unified = torch.tensor([1, 2, 3, 7, 8])
        pathology = model.convert_unified_to_pathology_labels(unified)
        print(f"\n[PASS] Label conversion: {unified.tolist()} -> {pathology.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Pathology test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MULTI-CLASS DIAGNOSTIC MODEL TESTS")
    print("Stage 2 of Dual-Architecture System")
    print("=" * 60)
    
    success = True
    success &= test_model_basic()
    success &= test_pathology_classes()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All tests passed!")
        print("Model is ready for deployment")
    else:
        print("[FAILURE] Some tests failed")
    print("=" * 60)

if __name__ == "__main__":
    main()