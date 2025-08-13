# Dataset Analysis - Ear Condition Classification

## Overview

Comprehensive analysis of medical imaging datasets for ear condition classification, compiled from multiple authoritative sources with ENT specialist validation.

## Primary Dataset: Ebasaran Kaggle Collection

### Dataset Statistics
- **Total Images**: 956 otoscopic images
- **Expert Validation**: Evaluated by 3 ENT specialists
- **Quality Control**: Low-quality images removed (light, blur, motion artifacts)
- **Source**: https://www.kaggle.com/datasets/erdalbasaran/eardrum-dataset-otitis-media

### Class Distribution

| Condition | Count | Percentage | Notes |
|-----------|-------|------------|-------|
| Normal Tympanic Membrane | 535 | 55.9% | Healthy eardrums |
| Earwax (Cerumen Impaction) | 140 | 14.6% | Various degrees of impaction |
| Acute Otitis Media (AOM) | 119 | 12.4% | Infected, inflamed membranes |
| Chronic Suppurative Otitis Media | 63 | 6.6% | Persistent infection/discharge |
| Otitis Externa | 41 | 4.3% | Outer ear canal inflammation |
| Tympanoskleros (Myringosclerosis) | 28 | 2.9% | Calcium deposits on membrane |
| Ear Ventilation Tube | 16 | 1.7% | Surgical tubes in place |
| Pseudo Membranes | 11 | 1.1% | False membrane formations |
| Foreign Bodies in Ear | 3 | 0.3% | Objects lodged in ear canal |

### Class Imbalance Analysis

#### Major Classes (>10% of dataset)
- **Normal**: 535 images (55.9%) - Well represented
- **Earwax**: 140 images (14.6%) - Good representation
- **AOM**: 119 images (12.4%) - Adequate for training

#### Minor Classes (1-10% of dataset)
- **Chronic OM**: 63 images (6.6%) - May need augmentation
- **Otitis Externa**: 41 images (4.3%) - Requires careful sampling
- **Tympanoskleros**: 28 images (2.9%) - Limited, needs augmentation
- **Ventilation Tube**: 16 images (1.7%) - Very limited
- **Pseudo Membranes**: 11 images (1.1%) - Critically low

#### Critical Classes (<1% of dataset)
- **Foreign Bodies**: 3 images (0.3%) - Insufficient for reliable training

## Supplementary Datasets

### UCI Kaggle - Otoscope Data
- **Purpose**: Additional validation data
- **Format**: Digital otoscope images
- **Integration**: Cross-validation and testing

### VanAk Figshare Collection
- **Source**: https://figshare.com/articles/dataset/eardrum_zip/13648166/1
- **Content**: Eardrum images with metadata
- **Use Case**: External validation set

### Sumotosima GitHub Dataset
- **Repository**: https://github.com/anas2908/Sumotosima  
- **Format**: CSV/Excel with image metadata
- **Contribution**: Demographic and clinical data correlation

### Roboflow Digital Otoscope
- **Source**: https://universe.roboflow.com/otoscope/digital-otoscope
- **Format**: Annotated images
- **Application**: Object detection and localization tasks

## Data Quality Assessment

### Enhanced Image Quality Metrics (Production System)
- **Resolution**: Standardized to 500x500 pixels with lossless PNG compression
- **Focus**: Sharp tympanic membrane visibility
- **Color Cast Detection**: Automated detection of channel imbalances
  - **Severe Color Cast**: Ratio >1.5 (quality score penalized by 0.4)
  - **Moderate Color Cast**: Ratio >1.3 (quality score penalized by 0.2)
- **Exposure Analysis**: Comprehensive brightness assessment
  - **Overexposure**: Average pixel values >220 (penalized by 0.3)
  - **Underexposure**: Average pixel values <35 (penalized by 0.3)
  - **Extreme Exposure**: Values <20 or >235 (additional 0.2 penalty)
- **Quality Scoring**: Automated 0-1 scale scoring system
- **Processing Report**: Comprehensive JSON reports with detailed statistics

### Expert Validation Protocol
1. **Triple Review**: Each image evaluated by 3 ENT specialists
2. **Consensus Requirement**: 2/3 agreement for final classification
3. **Automated Quality Filtering**: Production-ready quality assessment pipeline
4. **Standardization**: Consistent diagnostic criteria applied

### Quality Assessment Results (From Production Pipeline)
Based on processing of 4,737 medical images across datasets:
- **Color Cast Detection**: Identifies images with significant channel imbalances
- **Quality Score Distribution**: Most images achieve 0.8+ quality scores
- **Processing Statistics**: 150-200 images per minute with full quality analysis
- **Report Generation**: Detailed JSON reports with issue categorization

## Enhanced Data Preprocessing Pipeline

### Production Image Preprocessing (`src/preprocessing/image_utils.py`)
```bash
# Production preprocessing with comprehensive quality assessment
python src/preprocessing/image_utils.py

# Strict quality mode - reject images with any quality issues
python src/preprocessing/image_utils.py --strict-quality

# Custom quality threshold (default: 0.8)
python src/preprocessing/image_utils.py --quality-threshold 0.9

# Force reprocessing with detailed logging
python src/preprocessing/image_utils.py --force-reprocess --verbose
```

### Medical-Grade Enhancement Process
1. **CLAHE Enhancement**: LAB color space processing for optimal medical image enhancement
2. **Quality Analysis**: Multi-factor quality assessment with automated scoring
3. **Standardization**: 500x500 PNG format with lossless compression
4. **Validation**: Comprehensive error handling and processing verification
5. **Reporting**: Detailed JSON reports with processing statistics

### Training Data Pipeline
```python
def preprocess_otoscope_image(image_path):
    """
    Enhanced preprocessing for otoscope images
    Note: Images pre-processed by production pipeline (image_utils.py)
    """
    # Load standardized PNG (already CLAHE-enhanced)
    image = Image.open(image_path).convert('RGB')
    
    # Resize to training dimensions (from standardized 500x500)
    image = image.resize((384, 384), Image.LANCZOS)
    
    # Normalize pixel values for training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(image)
```

### Data Augmentation Strategy

#### Class-Specific Augmentation
```python
# Heavy augmentation for minor classes
AUGMENTATION_FACTORS = {
    'Foreign_Bodies': 50,        # 3 → 150 images
    'Pseudo_Membranes': 15,      # 11 → 165 images  
    'Ear_Ventilation_Tube': 10,  # 16 → 160 images
    'Tympanoskleros': 6,         # 28 → 168 images
    'Otitis_Externa': 4,         # 41 → 164 images
    'Chronic_OM': 3,             # 63 → 189 images
    'Normal': 1,                 # No augmentation needed
    'Earwax': 1,                 # Sufficient data
    'AOM': 1                     # Sufficient data
}
```

#### Augmentation Techniques
- **Rotation**: ±15 degrees (anatomically plausible)
- **Brightness/Contrast**: ±20% (lighting variations)
- **Color Jitter**: Slight hue/saturation changes
- **Horizontal Flip**: Mirror images (bilateral symmetry)
- **Zoom**: 90-110% scale (distance variations)
- **Gaussian Noise**: Minimal (sensor noise simulation)

## Dataset Splits

### Training/Validation/Test Strategy
```python
# Stratified split maintaining class proportions
SPLIT_RATIOS = {
    'train': 0.70,      # 669 images
    'validation': 0.15, # 143 images  
    'test': 0.15       # 144 images
}

# Ensure minimum samples per class in each split
MIN_SAMPLES_PER_SPLIT = {
    'train': 2,
    'validation': 1, 
    'test': 1
}
```

### Cross-Validation Strategy
- **K-Fold**: 5-fold stratified cross-validation
- **External Validation**: Separate datasets for final testing
- **Clinical Validation**: Expert review of model predictions

## Data Challenges and Solutions

### Challenge 1: Severe Class Imbalance
**Problem**: Foreign bodies (3 images) vs Normal (535 images)
**Solutions**:
- Aggressive data augmentation for minor classes
- Focal loss function to handle imbalance
- SMOTE for synthetic sample generation
- Ensemble methods combining class-specific models

### Challenge 2: Limited Minority Class Data
**Problem**: Some conditions have <30 images
**Solutions**:
- Transfer learning from related medical imaging tasks
- Few-shot learning techniques
- Synthetic data generation using GANs
- Active learning to prioritize new data collection

### Challenge 3: Dataset Generalization
**Problem**: Single-source dataset may not generalize
**Solutions**:
- Multi-institutional validation
- Domain adaptation techniques
- Federated learning approaches
- Continuous model updates with new data

## Recommended Model Training Strategy

### Phase 1: Baseline Model
1. Train on balanced dataset (after augmentation)
2. Use class weights to handle remaining imbalance
3. Focus on major classes first (Normal, Earwax, AOM)

### Phase 2: Full Model
1. Include all 9 classes with augmentation
2. Implement hierarchical classification
3. Use ensemble methods for minority classes

### Phase 3: Validation
1. External dataset validation
2. Clinical expert validation  
3. Prospective validation study

## Performance Expectations

### Expected Accuracy by Class
| Condition | Expected Accuracy | Confidence |
|-----------|------------------|------------|
| Normal | 95%+ | High |
| Earwax | 90%+ | High |
| AOM | 88%+ | High |
| Chronic OM | 85%+ | Medium |
| Otitis Externa | 82%+ | Medium |
| Tympanoskleros | 78%+ | Medium |
| Ventilation Tube | 75%+ | Low |
| Pseudo Membranes | 70%+ | Low |
| Foreign Bodies | 60%+ | Very Low |

### Overall System Metrics
- **Macro F1-Score**: >0.80 target
- **Weighted F1-Score**: >0.85 target  
- **Sensitivity**: >0.90 for pathological conditions
- **Specificity**: >0.85 for normal classification

## Data Collection Recommendations

### Priority for Additional Data
1. **Foreign Bodies**: Critical need (target: 50+ images)
2. **Pseudo Membranes**: High priority (target: 40+ images)
3. **Ear Ventilation Tubes**: High priority (target: 50+ images)
4. **Tympanoskleros**: Medium priority (target: 60+ images)
5. **Otitis Externa**: Medium priority (target: 80+ images)

### Collection Strategy
- Partner with ENT clinics and hospitals
- Implement active learning to identify valuable cases
- Focus on underrepresented demographics
- Ensure diverse imaging equipment representation
- Maintain strict quality and labeling standards