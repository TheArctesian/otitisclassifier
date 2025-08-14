# Project Structure

This document describes the organized structure of the otitis classifier dual-architecture medical AI system.

## Overview

The project follows industry best practices with clear separation of concerns, modular design, and comprehensive testing infrastructure.

## Directory Structure

```
otitisclassifier/
├── README.md                           # Project overview and getting started
├── CLAUDE.md                          # Claude Code integration guide
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Container configuration
├── docker-compose.yml               # Multi-service orchestration
├── install_requirements.py          # Dependency installation script
│
├── app/                              # Web application
│   └── app.py                       # Streamlit interface with dual model integration
│
├── src/                             # Source code modules
│   ├── models/                      # Model architectures and implementations
│   │   ├── __init__.py
│   │   ├── binary_screening.py     # ✅ Binary screening model (EfficientNet-B3)
│   │   ├── multiclass_diagnostic.py # ✅ Multi-class diagnostic model (EfficientNet-B4)
│   │   └── clinical_models.py      # Clinical model utilities and base classes
│   │
│   ├── data/                        # Data loading and preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── loader.py               # Basic dataset loading utilities
│   │   ├── stage_based_loader.py   # ✅ Dual architecture dataset management
│   │   ├── class_mapping.py        # Medical condition class mappings
│   │   ├── metadata.py             # Dataset metadata management
│   │   └── weights.py              # Class balancing and weighting utilities
│   │
│   ├── core/                        # Core utilities and foundational components
│   │   ├── __init__.py
│   │   ├── classes.py              # Medical condition definitions
│   │   ├── paths.py                # File path management
│   │   ├── transforms.py           # Image transformation pipelines
│   │   └── validation.py           # Data validation utilities
│   │
│   ├── preprocessing/               # Image preprocessing and enhancement
│   │   ├── __init__.py
│   │   └── image_utils.py          # ✅ LAB color space CLAHE processing pipeline
│   │
│   ├── evaluation/                  # Model evaluation and clinical metrics
│   │   ├── __init__.py
│   │   └── clinical_metrics.py     # Clinical performance evaluation
│   │
│   ├── visualization/               # Clinical interpretability and visualization
│   │   ├── __init__.py
│   │   └── clinical_interpretability.py  # Grad-CAM and attention visualization
│   │
│   ├── data_prep.py                # Legacy data preparation utilities
│   ├── model_train.py              # Model training pipelines
│   ├── model_evaluate.py           # Model evaluation utilities
│   └── utils.py                    # General utility functions
│
├── tests/                          # Comprehensive test suite
│   ├── __init__.py
│   ├── test_binary_model_simple.py # ✅ Binary screening model validation
│   ├── test_multiclass_simple.py  # ✅ Multi-class diagnostic model validation
│   ├── test_binary_screening.py   # Dual-architecture integration tests
│   └── test_binary_model_temp.pt  # Test model weights
│
├── examples/                       # Usage examples and demonstrations
│   ├── binary_screening_example.py # ✅ Binary screening model usage
│   └── multiclass_diagnostic_example.py # ✅ Multi-class diagnostic model usage
│
├── scripts/                        # Data processing and utility scripts
│   ├── process_all_datasets.py    # Multi-dataset processing pipeline
│   ├── validate_data_integrity.py # Data quality assurance
│   ├── convert_to_png.py          # Image format conversion
│   ├── raw_unified.py             # Raw data unification
│   └── unified_process.py          # Unified processing pipeline
│
├── docs/                           # Complete technical documentation
│   ├── README.md                   # Documentation overview
│   ├── PROJECT_PLAN.md            # Enhanced project plan with implementation status
│   ├── PROJECT_STRUCTURE.md       # This file - project organization
│   ├── BINARY_SCREENING_IMPLEMENTATION.md  # Binary screening model documentation
│   ├── CLINICAL_INTEGRATION.md    # Clinical workflow integration
│   ├── DATASET_ANALYSIS.md        # Dataset analysis and statistics
│   ├── DECISION_TREE_FRAMEWORK.md # Multi-modal decision logic
│   ├── DEPLOYMENT_GUIDE.md        # Production deployment guidance
│   ├── ML_PIPELINE_ARCHITECTURE.md # Machine learning pipeline design
│   ├── STAGE_BASED_TRAINING.md    # Training methodology documentation
│   └── TECHNICAL_SPECIFICATIONS.md # Technical requirements and specifications
│
├── config/                         # Configuration files for clinical deployment
│   ├── clinical_config.yaml       # Clinical validation parameters
│   ├── dataset_config.yaml        # Dataset configuration
│   └── model_config.yaml          # Model architecture configuration
│
├── papers/                         # Research papers and references
│   ├── Papers.md                   # Paper index and summaries
│   ├── Livingstone et al. Journal of Otolaryngology (2019).pdf
│   ├── OTONet_Deep_Neural_Network_for_Precise_Otoscopy_Image_Classification.pdf
│   ├── applsci-11-01831-with-cover.pdf
│   └── sumotosima.pdf
│
└── data/                          # Medical image datasets and processing
    ├── raw/                       # Original source datasets
    │   ├── ebasaran-kaggale/     # Primary training dataset (956 images)
    │   ├── uci-kaggle/           # Fine-tuning dataset (~900+ images)
    │   ├── vanak-figshare/       # External validation dataset (~270+ images)
    │   ├── sumotosima-github/    # Clinical text annotations (38+ cases)
    │   └── readme.md             # Data source documentation
    │
    ├── processed/                 # Preprocessed datasets ready for training
    │   ├── ebasaran-kaggale/     # ✅ PNG converted with CLAHE enhancement
    │   ├── uci-kaggle/           # Processed fine-tuning data
    │   ├── vanak-figshare/       # Processed validation data
    │   └── sumotosima-github/    # Processed clinical annotations
    │
    └── unified/                   # Unified dataset management
        └── ear_conditions/        # Combined dataset for cross-validation
```

## Key Organizational Principles

### 1. **Separation of Concerns**
- **Models**: Pure model architectures and implementations
- **Data**: Dataset loading, preprocessing, and management
- **Core**: Foundational utilities and shared components
- **Tests**: Comprehensive validation and testing
- **Scripts**: Data processing and utility operations
- **Docs**: Complete technical documentation

### 2. **Unix Philosophy Implementation**
- **Modular Design**: Each module has a single, well-defined responsibility
- **Composable Components**: Functions and classes work together seamlessly
- **Simple Interfaces**: Clear, predictable inputs and outputs
- **Focused Functionality**: Small, understandable components

### 3. **Medical AI Best Practices**
- **Strict Data Isolation**: Training/validation/test sets never overlap
- **Clinical Safety First**: Conservative thresholds and validation protocols
- **Comprehensive Testing**: Unit tests, integration tests, and clinical validation
- **Documentation-Driven**: Every component thoroughly documented

### 4. **Production Readiness**
- **Container Support**: Docker and docker-compose for deployment
- **Configuration Management**: YAML-based configuration for different environments
- **Monitoring Ready**: Structured logging and performance tracking
- **Scalable Architecture**: Modular design supports horizontal scaling

## Implementation Status

### ✅ **Completed Components**
- **Binary Screening Model**: Complete EfficientNet-B3 implementation with LAB color features (11M+ parameters)
- **Multi-Class Diagnostic Model**: Complete EfficientNet-B4 implementation with focal loss and regional attention (22.7M+ parameters)
- **Data Pipeline**: Stage-based dual architecture dataset management
- **Image Processing**: LAB color space CLAHE enhancement pipeline
- **Testing Infrastructure**: Comprehensive test suite with all tests passing
- **Project Organization**: Clean, modular structure following best practices

### 🔄 **Next Phase Priorities**
1. **✅ COMPLETED: Multi-Class Diagnostic Model**: Stage 2 of dual architecture system with EfficientNet-B4 and focal loss
2. **✅ COMPLETED: Regional Analysis Framework**: 8-region anatomical attention mechanism implemented
3. **NEXT: Dual Model Integration**: Seamless Stage 1 → Stage 2 workflow with clinical decision support
4. **Enhanced Clinical Validation**: ENT specialist agreement protocols and clinical deployment pipeline

## Development Workflow

### **Testing**
```bash
# Run binary screening tests
python tests/test_binary_model_simple.py

# Run multi-class diagnostic tests
python tests/test_multiclass_simple.py

# Run integration tests
python tests/test_binary_screening.py

# Run example usage
python examples/binary_screening_example.py
python examples/multiclass_diagnostic_example.py
```

### **Data Processing**
```bash
# Process medical images
python src/preprocessing/image_utils.py

# Validate data integrity
python scripts/validate_data_integrity.py

# Process all datasets
python scripts/process_all_datasets.py
```

### **Application Development**
```bash
# Run Streamlit app
streamlit run app/app.py

# Docker development
docker-compose up

# Docker build
docker build -t otitis-classifier .
```

## Architecture Benefits

### **Maintainability**
- Clear module boundaries reduce coupling
- Consistent naming conventions improve readability
- Comprehensive documentation enables team collaboration

### **Testability** 
- Isolated components enable focused unit testing
- Integration tests validate end-to-end workflows
- Clinical validation ensures medical safety

### **Scalability**
- Modular design supports incremental feature addition
- Container-ready architecture enables cloud deployment
- Configuration-driven approach supports multiple environments

### **Clinical Safety**
- Strict data isolation prevents contamination
- Conservative validation protocols ensure patient safety
- Comprehensive testing validates clinical workflows

This organization supports the project's evolution from research prototype to production-ready medical AI system while maintaining code quality, clinical safety, and development velocity.