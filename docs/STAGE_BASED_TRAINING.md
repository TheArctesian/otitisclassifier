# Stage-Based Training Guide

## Overview

This document outlines the stage-based training strategy for medical AI that ensures strict data isolation and FDA-compliant validation. The methodology follows medical AI best practices with progressive domain adaptation across 3 validated datasets.

## Stage-Based Dataset Architecture

### Training Stages
1. **Stage 1: Base Training** - Ebasaran-Kaggle (956 images) - Foundation model training
2. **Stage 2: Fine-Tuning** - UCI-Kaggle (~900+ images) - Domain adaptation
3. **Stage 3: External Validation** - VanAk-Figshare (~270+ images) - Unbiased evaluation
4. **Clinical Annotations** - Sumotosima-GitHub (38+ cases) - Interpretability validation

### Stage Isolation Principles

| Stage | Dataset | Usage | Data Split | Isolation Level |
|-------|---------|-------|------------|----------------|
| Stage 1 | Ebasaran-Kaggle | Base Training | 80% train, 20% val | Internal split only |
| Stage 2 | UCI-Kaggle | Fine-Tuning | 90% train, 10% val | No overlap with Stage 1 |
| Stage 3 | VanAk-Figshare | External Test | 100% test | Never used for training |
| Clinical | Sumotosima-GitHub | Validation | Expert comparison | Interpretability only |

## Stage-Based Training Strategy

### Phase 1: Data Isolation Setup
- Validate strict separation between training stages
- Apply consistent CLAHE preprocessing across all formats
- Preserve full resolution at 500×500 pixels for medical detail
- Convert all images to unified PNG format for quality preservation
- Implement contamination detection to prevent data leakage

### Phase 2: Class Mapping and Standardization
- Map dataset-specific class names to unified 9-class taxonomy
- Handle naming variations and synonyms across datasets
- Validate mapping completeness and accuracy
- Document class correspondence for reproducibility

### Phase 3: Source-Aware Data Splitting
- **Training Set**: Ebasaran-Kaggle + UCI-Kaggle (stratified combination)
- **Validation Set**: Hold-out portion from training datasets (20%)
- **External Test Set**: VanAk-Figshare (completely separate source)
- **Clinical Validation**: Sumotosima annotations for interpretation checking

### Phase 4: Quality Assurance and Validation
- Cross-dataset duplicate detection using perceptual hashing
- Image quality validation and outlier detection
- Class distribution analysis and imbalance assessment
- Processing pipeline validation and benchmarking

## Technical Implementation

### Configuration Management
- Use Hydra framework for hierarchical configuration management
- YAML-based configuration files for reproducibility
- Environment-specific overrides for development/production

### Data Processing Pipeline
```
Raw Datasets → Format Validation → Preprocessing → Class Mapping → Quality Validation → Unified Dataset
```

### Processing Scripts
- `scripts/process_all_datasets.py` - Main processing pipeline
- `scripts/create_combined_dataset.py` - Dataset combination and splitting
- `scripts/validate_data_integrity.py` - Comprehensive validation

## Class Mapping Strategy

### Unified Class Taxonomy (9 Classes)
1. **Normal_Tympanic_Membrane** - Healthy eardrum
2. **Acute_Otitis_Media** - Active middle ear infection
3. **Chronic_Otitis_Media** - Persistent middle ear pathology
4. **Cerumen_Impaction** - Earwax blockage
5. **Otitis_Externa** - Outer ear canal inflammation
6. **Myringosclerosis** - Eardrum scarring/calcification
7. **Tympanostomy_Tubes** - Surgical ventilation tubes
8. **Foreign_Bodies** - Objects in ear canal
9. **Pseudo_Membranes** - False membrane formations

### Cross-Dataset Mapping
```yaml
# Example mapping (to be completed during implementation)
ebasaran_kaggle:
  "Normal" → "Normal_Tympanic_Membrane"
  "Aom" → "Acute_Otitis_Media"
  "Chornic" → "Chronic_Otitis_Media"  # Note: original typo preserved
  "Earwax" → "Cerumen_Impaction"
  # ... complete mapping

uci_kaggle:
  "Normal" → "Normal_Tympanic_Membrane"
  "Acute Otitis Media" → "Acute_Otitis_Media"
  "Cerumen Impaction" → "Cerumen_Impaction"
  # ... complete mapping
```

## Validation Strategy

### Cross-Dataset Validation Approach
1. **Internal Validation**: Stratified splits within combined training data
2. **External Validation**: VanAk-Figshare as completely separate test set
3. **Clinical Validation**: Sumotosima expert annotations for interpretation
4. **Source-Aware Evaluation**: Performance analysis by dataset source

### Quality Metrics
- Image integrity and format consistency
- Class distribution balance across sources
- Cross-dataset duplicate detection
- Processing pipeline performance benchmarks

### Performance Expectations
| Metric | Target | Rationale |
|--------|---------|-----------|
| Overall Accuracy | >90% | Clinical decision support requirement |
| Cross-Dataset Stability | <5% variance | Generalization validation |
| Processing Speed | <1s per image | Real-time clinical workflow |
| Duplicate Rate | <1% | Data quality assurance |

## Clinical Integration Context

### Multi-Modal System Integration
- Image classification provides **40% weight** in final diagnostic decision
- Integration with symptom assessment (35%) and patient history (25%)
- Clinical decision support with confidence-based recommendations
- Safety protocols for low-confidence cases requiring human review

### Regulatory Considerations
- FDA guidelines compliance for medical AI validation
- Cross-institutional validation following medical device standards
- Documentation requirements for regulatory submission
- Clinical trial protocol alignment

## Implementation Timeline

### Week 1-2: Infrastructure Setup
- [ ] Create configuration framework
- [ ] Set up processing scripts structure
- [ ] Implement basic validation tools

### Week 3-4: Dataset Processing
- [ ] Process all 4 datasets with unified pipeline
- [ ] Implement class mapping logic
- [ ] Validate data integrity across sources

### Week 5-6: Integration and Validation
- [ ] Create combined dataset structure
- [ ] Implement cross-dataset validation
- [ ] Generate comprehensive integration report

### Week 7-8: Documentation and Testing
- [ ] Complete technical documentation
- [ ] Implement automated testing suite
- [ ] Validate clinical integration readiness

## Next Steps

1. **Complete Class Mapping**: Finalize mapping between all dataset class names
2. **Implement Processing Pipeline**: Execute unified preprocessing across all datasets
3. **Validate Integration**: Comprehensive quality assurance and validation
4. **Clinical Review**: Expert validation of integrated dataset quality
5. **Model Training**: Begin training on combined dataset with cross-validation

## Risk Mitigation

### Technical Risks
- **Data Quality**: Comprehensive validation and quality checking
- **Class Imbalance**: Differential augmentation and loss weighting
- **Domain Shift**: Cross-dataset evaluation and domain adaptation

### Clinical Risks
- **Bias Introduction**: Multi-source validation and bias testing
- **Performance Degradation**: Conservative clinical thresholds
- **Interpretability**: Expert-validated explanation generation

---

*This document should be updated as implementation progresses and new insights are gained from the integration process.*