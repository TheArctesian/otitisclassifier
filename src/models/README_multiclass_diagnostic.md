# Multi-Class Diagnostic Model (Stage 2 Dual-Architecture)

## Overview

The Multi-Class Diagnostic Model implements Stage 2 of the dual-architecture medical AI system for otitis classification. It processes pathological cases flagged by the binary screening model and provides specific pathology identification with enhanced clinical accuracy.

## Key Features

### Architecture
- **EfficientNet-B4 backbone** with RadImageNet transfer learning (higher capacity than B3 for diagnostic precision)
- **Regional attention mechanisms** for anatomical localization and pathology-specific feature extraction
- **LAB color feature integration** (18-dimensional features from binary screening pipeline)
- **Advanced focal loss** with class-specific gamma values for rare pathology handling

### Clinical Capabilities
- **8-class pathological classification** (excludes Normal cases handled by Stage 1)
- **Rare pathology detection** with aggressive handling for Foreign Bodies (3 samples) and Pseudo Membranes (11 samples)
- **Clinical decision support** with specialist referral recommendations
- **Regional confidence scoring** for anatomical localization
- **Grad-CAM interpretability** with pathology-specific visualization

## Pathological Classes

| Class ID | Pathology | Estimated Samples | Clinical Priority | Augmentation Strategy |
|----------|-----------|-------------------|------------------|---------------------|
| 0        | Acute Otitis Media | ~700+ | Critical | Standard |
| 1 | Earwax/Cerumen Impaction | ~400+ | Medium | Standard |
| 2 | Chronic Suppurative OM | ~80+ | High | 2x augmentation |
| 3 | Otitis Externa | ~60+ | Medium | 2x augmentation |
| 4 | Tympanoskleros/Myringosclerosis | ~35+ | Medium | 3x augmentation |
| 5 | Ear Ventilation/Tympanostomy Tubes | ~20+ | Medium | 5x augmentation |
| 6 | Pseudo Membranes | ~11 | Critical (rare) | **15x augmentation** |
| 7 | Foreign Bodies | ~3 | High (emergency) | **20x augmentation** |

## Usage

### Basic Model Creation

```python
from src.models.multiclass_diagnostic import create_multiclass_diagnostic_model

# Create model with default configuration
model = create_multiclass_diagnostic_model()

# Create model with custom configuration
config = {
    'num_pathology_classes': 8,
    'color_feature_fusion': True,
    'regional_attention': True,
    'dropout_rate': 0.4,
    'confidence_threshold': 0.7
}
model = create_multiclass_diagnostic_model(config)
```

### Clinical Prediction

```python
import torch

# Prepare input images [B, 3, 500, 500]
images = torch.randn(4, 3, 500, 500)

# Clinical prediction with confidence scoring
results = model.predict_with_confidence(images, return_attention=True)

# Access results
predictions = results['predictions']  # Class predictions (0-7 or -1 for uncertain)
probabilities = results['probabilities']  # Class probabilities
max_confidence = results['max_probability']  # Maximum confidence scores
specialist_review = results['specialist_review_needed']  # Specialist referral flags
urgent_attention = results['urgent_attention']  # Urgent case flags
attention_maps = results['attention_maps']  # Regional attention visualization
```

### Training with Data Loaders

```python
from src.data.stage_based_loader import create_medical_ai_datasets

# Create dataset manager for dual architecture
dataset_manager = create_medical_ai_datasets(
    base_training_path="data/processed/ebasaran-kaggale",
    fine_tuning_path="data/processed/uci-kaggle", 
    validation_path="data/processed/vanak-figshare",
    dual_architecture=True
)

# Get pathology-only data loaders
diagnostic_loaders = dataset_manager.get_diagnostic_dataloaders('base_training', batch_size=16)
train_loader = diagnostic_loaders['train']
val_loader = diagnostic_loaders['val']

# Training loop
model.train()
for images, pathology_labels in train_loader:
    # Forward pass
    logits = model(images)
    
    # Compute focal loss with rare class handling
    loss = model.compute_loss(logits, pathology_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
```

### Complete Training Script

```bash
# Basic training
python src/train_diagnostic_model.py --stage base_training

# Training with custom configuration
python src/train_diagnostic_model.py --config config/diagnostic_model_config.yaml --stage base_training

# Fine-tuning with integration testing
python src/train_diagnostic_model.py --stage fine_tuning --test-integration
```

## Dual-Architecture Integration

### Integration with Binary Screening

```python
from src.models.binary_screening import create_binary_screening_model
from src.models.multiclass_diagnostic import DualArchitectureIntegration

# Create both models
binary_model = create_binary_screening_model()
diagnostic_model = create_multiclass_diagnostic_model()

# Create integration wrapper
dual_system = DualArchitectureIntegration(
    binary_model=binary_model,
    diagnostic_model=diagnostic_model,
    binary_threshold=0.5,
    diagnostic_threshold=0.7
)

# Complete clinical workflow
results = dual_system.predict_clinical(images)
final_diagnosis = results['final_diagnosis']  # Unified taxonomy labels
clinical_recommendations = results['clinical_recommendations']
```

## Label Mappings

### Unified to Pathology Labels
The model converts from unified taxonomy (0-8) to pathology-only labels (0-7):

```python
# Convert unified labels to pathology labels
unified_labels = torch.tensor([1, 2, 3, 7, 8])  # Earwax, AOM, Chronic, Pseudo, Foreign
pathology_labels = model.convert_unified_to_pathology_labels(unified_labels)
# Result: tensor([1, 0, 2, 6, 7])

# Convert back to unified labels
unified_back = model.convert_pathology_to_unified_labels(pathology_labels)
# Result: tensor([1, 2, 3, 7, 8])
```

## Clinical Validation

### Performance Targets
- **Balanced Accuracy**: ≥85% across all pathological classes
- **Rare Class Sensitivity**: ≥80% for Pseudo Membranes and Foreign Bodies
- **Clinical Safety**: Conservative thresholds to minimize false negative referrals

### Validation Metrics

```python
# Clinical performance validation
metrics = model.validate_clinical_performance(
    val_loader,
    balanced_accuracy_threshold=0.85,
    rare_class_sensitivity_threshold=0.80
)

print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"Rare Class Sensitivity: {metrics['rare_class_sensitivity']:.4f}")
print(f"Clinical Safety: {'PASSED' if metrics['clinical_safety_passed'] else 'FAILED'}")
```

## Configuration

### Model Parameters
- **EfficientNet-B4**: Higher capacity backbone for diagnostic precision
- **Regional Attention**: 8 anatomical regions for pathology localization
- **Color Features**: 18 LAB color space features from preprocessing pipeline
- **Focal Loss**: Class-specific gamma values (1.5-3.5) with rare class emphasis

### Clinical Thresholds
- **Confidence Threshold**: 0.7 (conservative for medical safety)
- **Specialist Review**: <0.7 confidence requires specialist evaluation
- **Urgent Attention**: Foreign Bodies and severe infections flagged for immediate care
- **High Confidence**: >0.85 confidence enables standard care pathway

## Integration Points

### With Binary Screening Model
- Shares LAB color feature extraction pipeline (18 features)
- Processes only pathological cases flagged by binary screening
- Maintains clinical decision workflow continuity

### With Clinical Decision Support
- Provides specialist referral recommendations
- Generates anatomical localization maps
- Supports evidence-based diagnostic confidence scoring

## Files Structure

```
src/models/
├── multiclass_diagnostic.py           # Main model implementation
├── binary_screening.py               # Stage 1 binary model (dependency)
└── README_multiclass_diagnostic.md   # This documentation

src/data/
├── stage_based_loader.py             # Data loading with PathologyOnlyDataset
└── class_mapping.py                  # Class mapping utilities

src/
├── train_diagnostic_model.py         # Complete training script
└── config/
    └── diagnostic_model_config.yaml  # Training configuration example
```

## Clinical Deployment

### Safety Considerations
- Conservative confidence thresholds for medical safety
- Rare pathology emphasis prevents missed critical diagnoses  
- Regional attention provides anatomical context for clinical interpretation
- Integration with specialist referral workflows

### Performance Monitoring
- Continuous validation against clinical thresholds
- Per-class performance tracking for rare pathologies
- Clinical decision support quality metrics
- Integration testing with binary screening model

## Future Enhancements

- RadImageNet weight integration when available
- Advanced curriculum learning strategies
- Multi-modal integration with symptom assessment
- Real-time clinical feedback incorporation