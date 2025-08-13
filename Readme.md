# Otitis Classifier

Medical image classifier for ear conditions that analyzes otoscopic images to identify various ear pathologies. Built with modular architecture following Unix philosophy principles.

## Quick Start

### Running the Application
1. **Docker (Recommended)**: `docker-compose up` → Visit [localhost:8501](http://localhost:8501)
2. **Docker Build**: `docker build -t image-classifier .`
3. **Direct Python**: `streamlit run app/app.py --server.address=0.0.0.0`

### Development Setup
```bash
pip install -r requirements.txt
python src/preprocessing/image_utils.py  # Process raw images to PNG
```

## Architecture Overview

This project implements a **stage-based medical AI training architecture** with:
- **2,363 processed PNG images** with CLAHE enhancement at 500x500 resolution
- **9 ear condition classes** from 3 validated medical datasets
- **Stage-based training pipeline** with strict data isolation
- **FDA-compliant validation** on external datasets

## Stage-Based Data Sources
- **[Ebasaran-Kaggle](https://www.kaggle.com/datasets/erdalbasaran/eardrum-dataset-otitis-media)**: Stage 1 base training (955 images)
- **[UCI-Kaggle](https://www.kaggle.com/datasets/omduggineni/otoscopedata)**: Stage 2 fine-tuning (908 images)  
- **[VanAk-Figshare](https://figshare.com/articles/dataset/eardrum_zip/13648166/1)**: Stage 3 external validation (270 images)
- **[Sumotosima-GitHub](https://github.com/anas2908/Sumotosima)**: Clinical annotations for interpretability (38 cases)



## Medical Classifications (9 Classes)

| Class | Count | Clinical Priority | Data Quality |
|-------|-------|------------------|--------------|
| **Normal Tympanic Membrane** | ~800-900 | High (baseline) | Excellent |
| **Acute Otitis Media (AOM)** | ~700+ | Critical (pathology) | Excellent |
| **Earwax/Cerumen Impaction** | ~400+ | Medium (treatable) | Good |
| **Chronic Suppurative Otitis Media** | ~80+ | High (pathology) | Fair |
| **Otitis Externa** | ~60+ | Medium (pathology) | Fair |
| **Tympanoskleros/Myringosclerosis** | ~35+ | Medium (monitoring) | Limited |
| **Ear Ventilation Tube** | ~20+ | Medium (post-surgical) | Limited |
| **Pseudo Membranes** | ~11 | Low (rare pathology) | Critical |
| **Foreign Bodies** | ~3 | High (emergency) | Critical |

## Usage Examples

### Simple Dataset Loading
```python
from src.data.loader import create_simple_dataset
from src.utils import create_dataloader, print_dataset_info

# Load processed dataset
dataset = create_simple_dataset('data/processed/ebasaran-kaggale', image_size=224)
print_dataset_info(dataset)

# Create DataLoader  
dataloader = create_dataloader(dataset, batch_size=32, shuffle=True)

# Use in training loop
for images, labels in dataloader:
    # images: [batch_size, 3, 224, 224]  
    # labels: [batch_size] (class indices)
    pass
```

### Stage-Based Training
```python
from src.data.stage_based_loader import create_medical_ai_datasets

# Create stage-based training pipeline
dataset_manager = create_medical_ai_datasets(
    base_training_path="data/processed/ebasaran-kaggale",
    fine_tuning_path="data/processed/uci-kaggle", 
    validation_path="data/processed/vanak-figshare",
    image_size=500  # Full resolution preservation
)

# Stage 1: Base training
base_loaders = dataset_manager.get_stage_dataloaders('base_training')

# Stage 2: Fine-tuning
finetune_loaders = dataset_manager.get_stage_dataloaders('fine_tuning')

# Stage 3: External validation
validation_loaders = dataset_manager.get_stage_dataloaders('validation')
```

### Data Validation
```python
from src.core.validation import check_dataset_health

# Check dataset health
health = check_dataset_health('data/processed/ebasaran-kaggale')
print(f"Healthy: {health['healthy']}")
print(f"Success rate: {health['images']['success_rate']:.1%}")
```

## Documentation

See `CLAUDE.md` for complete development guide and clinical integration details.
