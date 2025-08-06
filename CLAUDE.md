# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an otitis (ear infection) classifier project that analyzes medical images of ear conditions. The application uses Streamlit to provide a web interface for image classification of various ear conditions including Normal Tympanic Membrane, Acute Otitis Media, Myringosclerosis, Chronic Otitis Media, Cerumen Impaction, and others.

## Essential Commands

### Running the Application
- **Development**: `docker-compose up` - Builds and runs the Streamlit app on http://localhost:8501
- **Docker Build**: `docker build -t image-classifier .`
- **Direct Streamlit**: `streamlit run app/app.py --server.address=0.0.0.0` (requires Python environment with requirements.txt installed)

### Data Processing
- **Convert Images**: Use `data/raw/convert_to_png.py` for TIFF/JPG to PNG conversion
- **Initialize Data Structure**: Run `bash data/unified/init.bash` to create organized directory structure for ear conditions

## Architecture

### Core Components
- **`app/app.py`**: Main Streamlit application with placeholder classification logic
- **`src/`**: Contains empty Python modules intended for:
  - `data_prep.py`: Data preprocessing and preparation
  - `model_train.py`: Model training logic
  - `model_evaluate.py`: Model evaluation and metrics
  - `utils.py`: Utility functions
- **`data/`**: Medical image datasets from multiple sources (Kaggle, UCI, Figshare, GitHub)

### Data Organization
The project consolidates data from 5 different medical image datasets:
- **Ebasaran Kaggle**: Various ear conditions in TIFF format
- **UCI Kaggle**: Otoscope data
- **VanAk Figshare**: Eardrum images  
- **Sumotosima GitHub**: CSV/Excel metadata
- **Roboflow**: Digital otoscope images

### Medical Classifications
The system is designed to classify 9 ear conditions:
1. Normal Tympanic Membrane
2. Acute Otitis Media (AOM)
3. Myringosclerosis
4. Chronic Otitis Media (COM/CSOM)
5. Cerumen Impaction/Earwax
6. Tympanostomy Tubes/Ear Ventilation Tubes
7. Otitis Externa
8. Foreign Objects/Bodies
9. Pseudo Membranes

## Development Notes

### Dependencies
- Python 3.12 base
- Streamlit for web interface
- Pillow (PIL) for image processing
- NumPy for numerical operations

### Current State
- The project is in early development stage
- Main application has placeholder classification logic
- Core source modules (`src/*.py`) are empty stubs
- Data processing scripts are partially implemented
- Docker configuration is ready for deployment
- **Comprehensive documentation available in `docs/` directory**

### Documentation Structure
- **`docs/PROJECT_PLAN.md`**: Complete implementation roadmap and architecture
- **`docs/TECHNICAL_SPECIFICATIONS.md`**: Detailed technical requirements and API specs
- **`docs/DATASET_ANALYSIS.md`**: Medical imaging data analysis and ML strategy
- **`docs/DECISION_TREE_FRAMEWORK.md`**: Multi-modal diagnostic decision system
- **`docs/README.md`**: Documentation overview and quick start guide

### File Structure
```
├── app/app.py                 # Main Streamlit application
├── src/                       # Core ML modules (currently empty)
├── data/                      # Medical image datasets
│   ├── raw/                   # Original datasets
│   └── unified/               # Processed/organized data
├── papers/                    # Research papers and documentation
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
└── docker-compose.yml        # Development environment setup
```