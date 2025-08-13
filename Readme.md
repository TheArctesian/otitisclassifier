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

This project implements a **modular, Unix philosophy-based architecture** with:
- **2,363 processed PNG images** with CLAHE enhancement
- **9 ear condition classes** from 4 validated medical datasets
- **Modular data loading** system for easy development and scaling
- **Simple, composable functions** that do one thing well

## Data Sources
- **[Ebasaran-Kaggle](https://www.kaggle.com/datasets/erdalbasaran/eardrum-dataset-otitis-media)**: Primary training (955 images)
- **[UCI-Kaggle](https://www.kaggle.com/datasets/omduggineni/otoscopedata)**: High-volume supplement (908 images)  
- **[VanAk-Figshare](https://figshare.com/articles/dataset/eardrum_zip/13648166/1)**: External validation (270 images)
- **[Sumotosima-GitHub](https://github.com/anas2908/Sumotosima)**: Clinical annotations (38 cases)



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

### Multi-Dataset Loading
```python
from src.data.multi import create_standard_multi_dataset
from src.utils import create_dataloader

# Combine multiple datasets with unified classes
multi_dataset = create_standard_multi_dataset(
    config='processed',  # Use processed PNG images
    datasets=['ebasaran', 'uci', 'vanak'],
    image_size=384,
    training=True
)

dataloader = create_dataloader(multi_dataset, batch_size=16)
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
