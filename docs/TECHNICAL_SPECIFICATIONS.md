# Technical Specifications - Multi-Modal Ear Diagnosis System

## System Requirements

### Performance Requirements
- **Image Classification**: <3 seconds response time
- **Symptom Processing**: <1 second response time  
- **Decision Tree**: <2 seconds for complete diagnosis
- **Concurrent Users**: Support 100+ simultaneous sessions
- **Uptime**: 99.5% availability target

### Accuracy Requirements
- **Image Classification**: >90% accuracy on validation set
- **Overall System Sensitivity**: >90% for pathological conditions
- **Overall System Specificity**: >85% for normal conditions
- **False Positive Rate**: <15%
- **False Negative Rate**: <10%

## Image Classification Component

### Model Architecture
```python
# Recommended CNN Architecture
class EarConditionClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # Base: EfficientNet-B3 or ResNet-50
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.backbone.global_pool(features)
        return self.classifier(features)
```

### Data Pipeline

#### Enhanced Image Preprocessing (`src/preprocessing/image_utils.py`)
- **Medical-Grade Enhancement**: LAB color space CLAHE processing optimized for otoscopy images
- **Quality Assessment Framework**: Comprehensive image quality analysis with automated scoring
- **Input Standardization**: All images converted to 500x500 PNG format with lossless compression
- **Color Cast Detection**: Automatic detection of severe (ratio >1.5) and moderate (ratio >1.3) color casts
- **Exposure Analysis**: Detection of over/under-exposure issues with brightness thresholds
- **Quality Metrics**: Automated quality scoring on 0-1 scale based on multiple factors
- **Idempotent Processing**: Safe to run multiple times without reprocessing existing files

#### Preprocessing Command Line Interface
```bash
# Basic processing with quality assessment
python src/preprocessing/image_utils.py

# Strict quality mode - reject images with any quality issues
python src/preprocessing/image_utils.py --strict-quality

# Custom quality threshold (0-1 scale, default: 0.8)
python src/preprocessing/image_utils.py --quality-threshold 0.9

# Force reprocessing of existing files
python src/preprocessing/image_utils.py --force-reprocess

# Verbose output with detailed processing information
python src/preprocessing/image_utils.py --verbose
```

#### Quality Assessment Metrics
- **Color Cast Ratio**: Channel imbalance detection (max_channel/min_channel)
- **Brightness Analysis**: Overall image brightness with extreme value detection
- **Exposure Classification**: Over/under-exposure based on pixel value distributions
- **Quality Score**: Composite score (0-1) incorporating all quality factors
- **Processing Report**: Comprehensive JSON report with statistics and quality details

#### Training Data Pipeline
- **Input Size**: 224x224 or 384x384 pixels (resized from standardized 500x500)
- **Normalization**: ImageNet statistics
- **Augmentations**: 
  - Rotation (±15°)
  - Brightness/Contrast (±0.2)
  - Horizontal flip
  - Color jittering
  - Gaussian blur (occasional)

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-4 with cosine annealing
- **Batch Size**: 32-64 (depending on GPU memory)
- **Loss Function**: Cross-entropy with class weights
- **Validation Split**: 80/20 train/validation
- **Early Stopping**: Patience of 10 epochs

## Decision Tree Engine

### Scoring Algorithm
```python
def calculate_diagnostic_score(image_pred, symptoms, history):
    """
    Calculate weighted diagnostic score
    
    Weights:
    - Image: 40%
    - Symptoms: 35% 
    - History: 25%
    """
    
    # Image component (0-1 scale)
    image_score = max(image_pred.values())
    image_weight = 0.4
    
    # Symptom component (0-1 scale)
    symptom_score = calculate_symptom_match(symptoms)
    symptom_weight = 0.35
    
    # History component (0-1 scale)
    history_score = calculate_risk_score(history)
    history_weight = 0.25
    
    final_score = (
        image_score * image_weight +
        symptom_score * symptom_weight + 
        history_score * history_weight
    )
    
    return final_score
```

### Decision Thresholds
- **High Confidence**: ≥0.85 → Provide diagnosis
- **Medium Confidence**: 0.65-0.84 → Probable diagnosis
- **Low Confidence**: <0.65 → Recommend examination
- **Critical Symptoms**: Immediate referral triggers

### Symptom Patterns
```python
SYMPTOM_PATTERNS = {
    'Acute_Otitis_Media': {
        'pain': {'weight': 0.3, 'threshold': 6},
        'fever': {'weight': 0.25, 'presence': True},
        'hearing_loss': {'weight': 0.2, 'presence': True},
        'discharge': {'weight': 0.15, 'type': 'purulent'},
        'duration': {'weight': 0.1, 'max_days': 14}
    },
    'Chronic_Otitis_Media': {
        'discharge': {'weight': 0.35, 'persistent': True},
        'hearing_loss': {'weight': 0.3, 'gradual': True},
        'pain': {'weight': 0.15, 'mild': True},
        'duration': {'weight': 0.2, 'min_days': 30}
    },
    # ... other patterns
}
```

## API Specifications

### Endpoints

#### Image Classification
```http
POST /api/v1/classify-image
Content-Type: multipart/form-data

Response:
{
    "predictions": {
        "Normal_Tympanic_Membrane": 0.85,
        "Acute_Otitis_Media": 0.10,
        "Chronic_Otitis_Media": 0.03,
        ...
    },
    "confidence": 0.85,
    "processing_time": 2.3
}
```

#### Symptom Assessment
```http
POST /api/v1/assess-symptoms
Content-Type: application/json

{
    "symptoms": {
        "pain_level": 7,
        "fever_present": true,
        "discharge_present": true,
        "hearing_changes": "reduced",
        "duration_days": 3
    }
}

Response:
{
    "symptom_score": 0.78,
    "pattern_matches": ["Acute_Otitis_Media", "Otitis_Externa"],
    "red_flags": []
}
```

#### Complete Diagnosis
```http
POST /api/v1/diagnose
Content-Type: application/json

{
    "image_file": "base64_encoded_image",
    "symptoms": {...},
    "patient_history": {...}
}

Response:
{
    "primary_diagnosis": "Acute_Otitis_Media",
    "confidence": 0.89,
    "differential_diagnoses": [
        {"condition": "Otitis_Externa", "probability": 0.23},
        {"condition": "Normal_Tympanic_Membrane", "probability": 0.15}
    ],
    "recommendations": [
        "Consider antibiotic treatment",
        "Follow up in 48-72 hours if symptoms persist"
    ],
    "referral_needed": false
}
```

## Database Schema

### Patients Table
```sql
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Diagnoses Table
```sql
CREATE TABLE diagnoses (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    image_path VARCHAR(255),
    image_predictions JSONB,
    symptoms JSONB,
    patient_history JSONB,
    final_diagnosis VARCHAR(100),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Patient History Table
```sql
CREATE TABLE patient_history (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    previous_ear_infections INTEGER DEFAULT 0,
    allergies TEXT[],
    medications TEXT[],
    risk_factors TEXT[],
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Security Requirements

### Data Protection
- **Encryption**: AES-256 for data at rest
- **Transport**: TLS 1.3 for data in transit
- **Authentication**: JWT tokens with rotation
- **Session Management**: Secure session handling
- **HIPAA Compliance**: PHI data handling protocols

### Access Control
- **Role-Based Access**: Admin, Clinician, Patient roles
- **API Rate Limiting**: 100 requests/minute per user
- **Input Validation**: Strict validation for all inputs
- **Audit Logging**: Complete audit trail for diagnoses

## Deployment Architecture

### Container Structure
```dockerfile
# Multi-stage build
FROM python:3.12-slim as base
# ... dependencies

FROM base as ml-service
COPY src/model_*.py ./
# ... ML service specific setup

FROM base as web-service  
COPY app/ ./
# ... web interface setup
```

### Docker Compose Services
```yaml
version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    
  web-app:
    build: 
      context: .
      target: web-service
    depends_on:
      - ml-service
      - postgres
      
  ml-service:
    build:
      context: .
      target: ml-service
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ear_diagnosis
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
```

## Monitoring and Logging

### Metrics to Track
- **Response Times**: Per endpoint and component
- **Accuracy Metrics**: Model performance over time
- **Error Rates**: 4xx/5xx response codes
- **Resource Usage**: CPU, memory, GPU utilization
- **User Engagement**: Diagnosis completion rates

### Logging Structure
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "diagnosis_completed",
    patient_id=patient_id,
    diagnosis=diagnosis,
    confidence=confidence_score,
    processing_time=processing_time,
    components_used=["image", "symptoms", "history"]
)
```

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": VERSION,
        "components": {
            "database": check_db_connection(),
            "ml_model": check_model_loaded(),
            "redis": check_cache_connection()
        }
    }
```

## Testing Strategy

### Unit Tests
- **Model Tests**: Prediction accuracy, input validation
- **API Tests**: Endpoint functionality, error handling
- **Decision Tree**: Logic validation with known cases

### Integration Tests
- **End-to-End**: Complete diagnostic workflows
- **Performance**: Load testing with concurrent users
- **Security**: Penetration testing, vulnerability scanning

### Clinical Validation
- **Test Cases**: Curated set of known diagnoses
- **Expert Review**: ENT specialist validation
- **Bias Testing**: Performance across demographics