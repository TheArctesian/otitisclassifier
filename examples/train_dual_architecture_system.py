#!/usr/bin/env python3
"""
Dual Architecture Medical AI Training Example

This example demonstrates how to use the unified training infrastructure
to train the complete dual-architecture otitis classification system.

Usage:
    python examples/train_dual_architecture_system.py

This script will:
1. Set up the complete training infrastructure
2. Train the binary screening model (Stage 1)
3. Train the multi-class diagnostic model (Stage 2)
4. Validate the integrated system (Stage 3)
5. Create a deployment package

Requirements:
- Processed datasets in data/processed/
- Sufficient GPU memory (recommended: 8GB+)
- Approximately 2-4 hours training time
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from training.training_orchestrator import create_training_orchestrator
from training.curriculum_learning import CurriculumLearningConfig
from training.progressive_unfreezing import UnfreezingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dual_architecture_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main training function for dual-architecture medical AI system."""
    logger.info("🏥 Starting Dual Architecture Medical AI Training")
    logger.info("=" * 60)
    
    # Configuration
    experiment_name = "otitis_dual_architecture_v1"
    
    # Data paths
    base_training_path = Path("data/processed/ebasaran-kaggale")
    fine_tuning_path = Path("data/processed/uci-kaggle") 
    validation_path = Path("data/processed/vanak-figshare")
    
    # Verify data paths exist
    for path in [base_training_path, fine_tuning_path, validation_path]:
        if not path.exists():
            logger.error(f"Dataset path not found: {path}")
            logger.error("Please ensure datasets are processed using: python src/preprocessing/image_utils.py")
            return
    
    logger.info(f"Data paths verified:")
    logger.info(f"  Base training: {base_training_path}")
    logger.info(f"  Fine-tuning: {fine_tuning_path}")
    logger.info(f"  Validation: {validation_path}")
    
    try:
        # Create training orchestrator with medical AI optimized configuration
        logger.info("🔧 Setting up training orchestrator...")
        
        orchestrator = create_training_orchestrator(
            experiment_name=experiment_name,
            base_output_dir=Path("training_outputs"),
            clinical_validation=True
        )
        
        # Customize training configuration for medical AI
        orchestrator.config.curriculum_config = CurriculumLearningConfig(
            binary_epochs=50,           # Conservative training for high sensitivity
            multiclass_epochs=100,      # Extended training for rare pathology detection
            validation_epochs=20,       # Thorough validation
            patience=15,                # Medical AI requires patience for convergence
            binary_lr=1e-4,            # Conservative learning rate for medical domain
            multiclass_lr=5e-5,        # Lower LR for complex multi-class problem
            clinical_sensitivity_threshold=0.98,  # High sensitivity requirement
            clinical_specificity_threshold=0.90,  # Reasonable specificity
            diagnostic_accuracy_threshold=0.85     # Diagnostic accuracy target
        )
        
        orchestrator.config.unfreezing_strategy = UnfreezingStrategy.HYBRID
        orchestrator.config.clinical_thresholds = {
            'binary_sensitivity': 0.98,      # Critical for clinical safety
            'binary_specificity': 0.90,      # Balance false positives
            'multiclass_balanced_accuracy': 0.85,  # Handle class imbalance
            'integration_accuracy': 0.85,    # End-to-end performance
            'end_to_end_sensitivity': 0.95   # Overall system sensitivity
        }
        
        logger.info("✅ Training orchestrator configured")
        logger.info(f"Experiment directory: {orchestrator.config.experiment_dir}")
        
        # Execute complete training pipeline
        logger.info("🚀 Starting complete dual architecture training pipeline...")
        
        pipeline_results = orchestrator.run_complete_training_pipeline(
            base_training_path=base_training_path,
            fine_tuning_path=fine_tuning_path,
            validation_path=validation_path,
            package_name="otitis_classifier_dual_v1"
        )
        
        # Display results summary
        logger.info("📊 Training Pipeline Results Summary")
        logger.info("=" * 60)
        
        # Binary screening results
        if 'binary_training' in pipeline_results:
            binary_metrics = pipeline_results['binary_training']['metrics']
            logger.info("🔍 Binary Screening Model (Stage 1):")
            logger.info(f"  ✓ Sensitivity: {binary_metrics.get('sensitivity', 'N/A'):.3f}")
            logger.info(f"  ✓ Specificity: {binary_metrics.get('specificity', 'N/A'):.3f}")
            logger.info(f"  ✓ Accuracy: {binary_metrics.get('accuracy', 'N/A'):.3f}")
        
        # Multi-class diagnostic results
        if 'multiclass_training' in pipeline_results:
            multiclass_metrics = pipeline_results['multiclass_training']['metrics']
            logger.info("🎯 Multi-Class Diagnostic Model (Stage 2):")
            logger.info(f"  ✓ Balanced Accuracy: {multiclass_metrics.get('balanced_accuracy', 'N/A'):.3f}")
            logger.info(f"  ✓ Accuracy: {multiclass_metrics.get('accuracy', 'N/A'):.3f}")
        
        # Validation results
        if 'validation' in pipeline_results:
            validation_summary = pipeline_results['validation']['validation_summary']
            compliance = pipeline_results['validation']['clinical_compliance']
            logger.info("🏥 Clinical Validation (Stage 3):")
            logger.info(f"  ✓ Protocols tested: {validation_summary.get('protocols_tested', [])}")
            logger.info(f"  ✓ Clinical compliance: {compliance.get('overall_compliant', False)}")
        
        # Deployment package
        if 'deployment' in pipeline_results:
            package_info = pipeline_results['deployment']
            logger.info("📦 Deployment Package:")
            logger.info(f"  ✓ Package: {package_info['package_name']}")
            logger.info(f"  ✓ Location: {package_info['package_path']}")
        
        # Clinical readiness assessment
        logger.info("🏥 Clinical Readiness Assessment")
        logger.info("=" * 60)
        
        clinical_ready = True
        
        # Check binary screening performance
        if 'binary_training' in pipeline_results:
            binary_metrics = pipeline_results['binary_training']['metrics']
            sensitivity = binary_metrics.get('sensitivity', 0)
            specificity = binary_metrics.get('specificity', 0)
            
            if sensitivity >= 0.98:
                logger.info("✅ Binary screening sensitivity meets clinical requirements (≥98%)")
            else:
                logger.warning(f"⚠️  Binary screening sensitivity below clinical threshold: {sensitivity:.3f} < 0.98")
                clinical_ready = False
                
            if specificity >= 0.90:
                logger.info("✅ Binary screening specificity meets clinical requirements (≥90%)")
            else:
                logger.warning(f"⚠️  Binary screening specificity below clinical threshold: {specificity:.3f} < 0.90")
                clinical_ready = False
        
        # Check multi-class diagnostic performance
        if 'multiclass_training' in pipeline_results:
            multiclass_metrics = pipeline_results['multiclass_training']['metrics']
            balanced_accuracy = multiclass_metrics.get('balanced_accuracy', 0)
            
            if balanced_accuracy >= 0.85:
                logger.info("✅ Multi-class diagnostic accuracy meets clinical requirements (≥85%)")
            else:
                logger.warning(f"⚠️  Multi-class diagnostic accuracy below threshold: {balanced_accuracy:.3f} < 0.85")
                clinical_ready = False
        
        # Check validation compliance
        if 'validation' in pipeline_results:
            compliance = pipeline_results['validation']['clinical_compliance']
            if compliance.get('overall_compliant', False):
                logger.info("✅ All validation protocols passed clinical compliance")
            else:
                logger.warning("⚠️  Some validation protocols failed clinical compliance")
                clinical_ready = False
        
        # Final assessment
        if clinical_ready:
            logger.info("🎉 SYSTEM READY FOR CLINICAL DEPLOYMENT")
            logger.info("The dual architecture model meets all clinical safety and performance requirements.")
        else:
            logger.warning("⚠️  SYSTEM REQUIRES ADDITIONAL TRAINING")
            logger.warning("Please review performance metrics and retrain if necessary.")
        
        logger.info("=" * 60)
        logger.info("🏥 Dual Architecture Medical AI Training Complete")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error("Please check the logs for detailed error information.")
        raise


if __name__ == "__main__":
    main()