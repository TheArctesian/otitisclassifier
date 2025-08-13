# Technical Specifications - Enhanced Multi-Modal Ear Diagnosis System

## Enhanced System Requirements

### Enhanced Performance Requirements
- **Binary Screening Model**: 95% for pathological conditions
- **Overall System Specificity**: >88% for normal conditions
- **Cross-Dataset Consistency**: 1.5) and moderate (ratio >1.3) color casts
- **Exposure Analysis**: Detection of over/under-exposure issues with brightness thresholds
- **Multi-Scale Support**: Processing pipeline supports 224×224, 384×384, 500×500 resolutions

#### Enhanced Training Pipeline Configuration

# Dual architecture training configuration
TRAINING_CONFIG = {
    'binary_screening': {
        'model': 'efficientnet_b3',
        'input_size': 224,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'sensitivity_target': 0.98,
        'specificity_target': 0.90
    },
    'multi_class_diagnostic': {
        'model': 'dual_scale_fusion',
        'input_sizes': ,
        'batch_size': 16,
        'learning_rate': 5e-5,
        'balanced_accuracy_target': 0.85,
        'rare_class_sensitivity_target': 0.80
    }
}


#### Adaptive Loss Function Implementation

class AdaptiveFocalLoss(nn.Module):
    """Enhanced focal loss with dynamic gamma based on class frequency"""
    def __init__(self, alpha=1, gamma_base=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma_base = gamma_base
        
    def forward(self, inputs, targets, class_frequencies):
        # Dynamic gamma based on class frequency
        gamma = self.gamma_base * (1 / class_frequencies[targets])
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**gamma * ce_loss
        
        return focal_loss.mean()

class UncertaintyQuantification(nn.Module):
    """Monte Carlo dropout for uncertainty estimation"""
    def __init__(self, model, n_samples=100):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
    
    def forward(self, x):
        self.model.train()  # Enable dropout during inference
        predictions = []
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=1)
        
        return mean_pred, uncertainty


### Enhanced Augmentation Strategy

ENHANCED_AUGMENTATION = {
    'binary_screening': {
        'normal_cases': {
            'transforms': [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3)
            ],
            'factor': 2
        },
        'pathological_cases': {
            'transforms': [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.4)
            ],
            'factor': 3
        }
    },
    'multi_class_diagnostic': {
        'Foreign_Bodies': {
            'transforms': [
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.ElasticTransform(alpha=50, sigma=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3)
            ],
            'factor': 20  # 3 → 60 images
        },
        'Pseudo_Membranes': {
            'transforms': [
                A.Rotate(limit=10, p=0.8),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.6),
                A.GaussianNoise(var_limit=(5, 15), p=0.3)
            ],
            'factor': 10  # 11 → 110 images
        },
        'common_pathologies': {
            'transforms': [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4)
            ],
            'factor': 2
        }
    }
}


## Enhanced Decision Tree Engine

### Dual Architecture Integration Algorithm

def enhanced_diagnostic_decision(screening_result, diagnostic_result, symptoms, history):
    """
    Enhanced decision tree with dual model integration and clinical safety protocols
    """
    
    # Stage 1: Binary screening decision with high sensitivity threshold
    if screening_result['pathology_probability'] = 0.85:
        # High confidence pathological - proceed with detailed diagnosis
        pathology_diagnosis = diagnostic_result['top_prediction']
        
        # Enhanced multi-modal integration with weighted evidence
        symptom_score = calculate_enhanced_symptom_score(symptoms, pathology_diagnosis) * 0.35
        history_score = calculate_risk_stratified_score(history, pathology_diagnosis) * 0.25
        image_score = combine_dual_model_scores(screening_result, diagnostic_result) * 0.40
        
        final_confidence = image_score + symptom_score + history_score
        
        return apply_enhanced_clinical_safety_protocols(pathology_diagnosis, final_confidence, symptoms)
    
    # Stage 3: Intermediate confidence - require additional validation
    else:
        return {
            'primary_diagnosis': 'UNCERTAIN',
            'confidence': 'MODERATE',
            'action': 'ADDITIONAL_CLINICAL_EVALUATION_REQUIRED',
            'reason': 'Conflicting evidence between screening and diagnostic models',
            'recommendation': 'Professional otoscopic examination recommended'
        }

def combine_dual_model_scores(screening_result, diagnostic_result):
    """Combine binary screening and multi-class diagnostic scores"""
    screening_confidence = screening_result['pathology_probability']
    diagnostic_confidence = max(diagnostic_result['class_probabilities'].values())
    
    # Weighted combination favoring screening sensitivity
    combined_score = (screening_confidence * 0.6) + (diagnostic_confidence * 0.4)
    
    return combined_score

def apply_enhanced_clinical_safety_protocols(diagnosis, confidence, symptoms):
    """Enhanced safety protocols with automatic referral and monitoring systems"""
    
    # High-risk diagnoses require specialist referral regardless of confidence
    HIGH_RISK_CONDITIONS = ['Chronic_Otitis_Media', 'Foreign_Bodies', 'Pseudo_Membranes']
    EMERGENCY_SYMPTOMS = ['severe_pain_sudden_onset', 'facial_paralysis', 'severe_dizziness']
    
    # Emergency protocol activation
    if any(symptom in symptoms for symptom in EMERGENCY_SYMPTOMS):
        return {
            'primary_diagnosis': diagnosis,
            'confidence': confidence,
            'action': 'EMERGENCY_ENT_REFERRAL',
            'urgency': 'IMMEDIATE',
            'reason': 'Emergency symptoms detected'
        }
    
    # High-risk pathology protocol
    if diagnosis in HIGH_RISK_CONDITIONS:
        return {
            'primary_diagnosis': diagnosis,
            'confidence': confidence,
            'action': 'SPECIALIST_ENT_REFERRAL',
            'reason': 'High-risk pathology detected',
            'urgency': 'WITHIN_24_HOURS',
            'monitoring': 'CONTINUOUS'
        }
    
    # Enhanced confidence thresholds with safety margins
    if confidence >= 0.90:  # Raised threshold for higher certainty
        return {
            'action': 'INITIATE_TREATMENT',
            'monitoring': 'STANDARD',
            'follow_up': '48_HOURS'
        }
    elif confidence >= 0.75:  # Conservative approach for moderate confidence
        return {
            'action': 'PROBABLE_DIAGNOSIS_MONITOR', 
            'follow_up': '24_HOURS',
            'additional_testing': 'CONSIDER'
        }
    else:
        return {
            'action': 'CLINICAL_EXAMINATION_REQUIRED', 
            'urgency': 'WITHIN_24_HOURS',
            'reason': 'Insufficient confidence for remote diagnosis'
        }


### Enhanced Curriculum Learning Integration

CURRICULUM_STAGES = {
    'stage_1_easy': {
        'weeks': '1-2',
        'description': 'Clear, high-quality diagnostic cases',
        'selection_criteria': 'image_quality > 0.9, diagnostic_clarity = high',
        'data_percentage': 0.4
    },
    'stage_2_moderate': {
        'weeks': '3-4', 
        'description': 'Ambiguous and challenging presentations',
        'selection_criteria': 'image_quality > 0.7, diagnostic_clarity = moderate',
        'data_percentage': 0.4
    },
    'stage_3_hard': {
        'weeks': '5-6',
        'description': 'Edge cases and rare pathological presentations',
        'selection_criteria': 'all_remaining_cases, focus_on_rare_classes',
        'data_percentage': 0.2
    }
}


## Enhanced API Specifications

### Dual Architecture Endpoints

#### Binary Screening

POST /api/v1/screen-pathology
Content-Type: multipart/form-data

Response:
{
    "screening_result": {
        "pathology_probability": 0.92,
        "normal_probability": 0.08,
        "confidence": "HIGH",
        "sensitivity_threshold": 0.98
    },
    "processing_time": 1.8,
    "model_version": "screening_v2.1"
}


#### Multi-Class Diagnostic

POST /api/v1/diagnose-pathology
Content-Type: multipart/form-data

Response:
{
    "diagnostic_result": {
        "top_prediction": "Acute_Otitis_Media",
        "class_probabilities": {
            "Acute_Otitis_Media": 0.78,
            "Chronic_Otitis_Media": 0.12,
            "Otitis_Externa": 0.08,
            "Foreign_Bodies": 0.02
        },
        "uncertainty_score": 0.15,
        "rare_class_confidence": 0.85
    },
    "processing_time": 2.1,
    "model_version": "diagnostic_v2.1"
}


#### Complete Dual Architecture Diagnosis

POST /api/v1/dual-diagnose
Content-Type: application/json

{
    "image_file": "base64_encoded_image",
    "symptoms": {...},
    "patient_history": {...}
}

Response:
{
    "screening_stage": {
        "pathology_detected": true,
        "confidence": 0.92,
        "proceed_to_diagnosis": true
    },
    "diagnostic_stage": {
        "primary_diagnosis": "Acute_Otitis_Media",
        "confidence": 0.89,
        "uncertainty_score": 0.12
    },
    "combined_result": {
        "final_diagnosis": "Acute_Otitis_Media",
        "overall_confidence": 0.91,
        "safety_protocol": "STANDARD_TREATMENT",
        "referral_required": false
    },
    "differential_diagnoses": [
        {"condition": "Chronic_Otitis_Media", "probability": 0.15},
        {"condition": "Otitis_Externa", "probability": 0.08}
    ],
    "recommendations": [
        "Initiate appropriate antibiotic treatment",
        "Monitor response to treatment within 48-72 hours",
        "Consider ENT referral if no improvement in 5-7 days"
    ],
    "processing_time": 2.9
}


## Enhanced Database Schema

### Dual Model Predictions Table

CREATE TABLE dual_model_predictions (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    screening_predictions JSONB,  -- Binary screening results
    diagnostic_predictions JSONB, -- Multi-class diagnostic results
    combined_confidence DECIMAL(3,2),
    uncertainty_score DECIMAL(3,2),
    model_versions JSONB,  -- Track both model versions
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


### Enhanced Diagnoses Table

CREATE TABLE enhanced_diagnoses (
    id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    dual_prediction_id INTEGER REFERENCES dual_model_predictions(id),
    image_path VARCHAR(255),
    screening_stage_result JSONB,
    diagnostic_stage_result JSONB,
    symptoms JSONB,
    patient_history JSONB,
    final_diagnosis VARCHAR(100),
    confidence_score DECIMAL(3,2),
    uncertainty_score DECIMAL(3,2),
    safety_protocol VARCHAR(50),
    referral_required BOOLEAN,
    clinical_action VARCHAR(100),
    urgency_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


### Model Performance Tracking

CREATE TABLE model_performance_metrics (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(20),  -- 'screening' or 'diagnostic'
    model_version VARCHAR(20),
    dataset_source VARCHAR(50),
    sensitivity DECIMAL(4,3),
    specificity DECIMAL(4,3),
    balanced_accuracy DECIMAL(4,3),
    rare_class_sensitivity DECIMAL(4,3),
    cross_dataset_variance DECIMAL(4,3),
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


## Enhanced Security Requirements

### Enhanced Data Protection
- **Multi-Model Encryption**: Separate encryption keys for screening and diagnostic models
- **Dual Authentication**: Two-factor authentication for model access
- **Audit Trail Enhancement**: Complete tracking of dual model decision pathways
- **Uncertainty Logging**: Secure logging of uncertainty scores for clinical review

### Enhanced Access Control
- **Model-Specific Permissions**: Granular access control for different model components
- **Clinical Role Management**: Specialized permissions for ENT specialists vs general practitioners
- **Emergency Override Protocols**: Secure emergency access for critical diagnoses
- **Confidence-Based Access**: Different access levels based on diagnostic confidence

## Enhanced Deployment Architecture

### Dual Model Container Structure

# Multi-stage build for dual architecture
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime as base
# ... base dependencies

FROM base as screening-service
COPY src/models/binary_screening.py ./
COPY models/screening_model_v2.1.pth ./
# ... screening model specific setup

FROM base as diagnostic-service  
COPY src/models/multiclass_diagnostic.py ./
COPY models/diagnostic_model_v2.1.pth ./
# ... diagnostic model specific setup

FROM base as integration-service
COPY src/integration/ ./
# ... dual model integration logic


### Enhanced Docker Compose Configuration

version: '3.8'
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - screening-service
      - diagnostic-service
      - integration-service
    
  screening-service:
    build:
      context: .
      target: screening-service
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_TYPE=screening
      - SENSITIVITY_TARGET=0.98
      
  diagnostic-service:
    build:
      context: .
      target: diagnostic-service
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_TYPE=diagnostic
      - BALANCED_ACCURACY_TARGET=0.85
      
  integration-service:
    build:
      context: .
      target: integration-service
    depends_on:
      - screening-service
      - diagnostic-service
      - postgres
      - redis
    environment:
      - SCREENING_SERVICE_URL=http://screening-service:5000
      - DIAGNOSTIC_SERVICE_URL=http://diagnostic-service:5001
      
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: enhanced_ear_diagnosis
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}


## Enhanced Monitoring and Logging

### Dual Model Performance Metrics

ENHANCED_METRICS = {
    'screening_model': [
        'sensitivity', 'specificity', 'false_negative_rate',
        'cross_dataset_consistency', 'inference_time'
    ],
    'diagnostic_model': [
        'balanced_accuracy', 'rare_class_sensitivity', 
        'uncertainty_calibration', 'expert_agreement'
    ],
    'combined_system': [
        'overall_diagnostic_accuracy', 'clinical_decision_impact',
        'referral_appropriateness', 'time_to_diagnosis'
    ]
}


### Enhanced Health Checks

@app.get("/health/dual-architecture")
async def dual_architecture_health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": VERSION,
        "components": {
            "screening_model": {
                "loaded": check_screening_model_loaded(),
                "performance": get_screening_performance_metrics(),
                "sensitivity_target": 0.98
            },
            "diagnostic_model": {
                "loaded": check_diagnostic_model_loaded(), 
                "performance": get_diagnostic_performance_metrics(),
                "balanced_accuracy_target": 0.85
            },
            "integration_engine": check_integration_engine(),
            "database": check_db_connection(),
            "cache": check_cache_connection()
        }
    }


## Enhanced Testing Strategy

### Dual Architecture Testing

class TestDualArchitecture:
    def test_screening_sensitivity(self):
        """Test binary screening model meets 98% sensitivity target"""
        pass
        
    def test_diagnostic_balanced_accuracy(self):
        """Test multi-class diagnostic model meets 85% balanced accuracy"""
        pass
        
    def test_rare_class_performance(self):
        """Test 80%+ sensitivity for Foreign Bodies and Pseudo Membranes"""
        pass
        
    def test_cross_dataset_consistency(self):
        """Test <5% performance variation across datasets"""
        pass
        
    def test_uncertainty_calibration(self):
        """Test uncertainty scores are well-calibrated"""
        pass
        
    def test_clinical_safety_protocols(self):
        """Test automatic referral systems for high-risk conditions"""
        pass


### Enhanced Clinical Validation
- **ENT Specialist Review**: 90%+ agreement target with specialist otolaryngologists
- **Cross-Institutional Testing**: Validation across multiple healthcare institutions
- **Prospective Clinical Study**: Real-world deployment validation
- **Bias Assessment**: Performance evaluation across demographic groups
- **Long-term Monitoring**: Continuous performance tracking in clinical deployment

## Enhanced Success Criteria

### Technical Performance Targets
| Component | Metric | Target | Clinical Impact |
|-----------|--------|--------|-----------------|
| **Binary Screening** | Sensitivity | ≥98% | Critical for patient safety |
| **Binary Screening** | Specificity | ≥90% | Minimize false positive referrals |
| **Multi-Class Diagnostic** | Balanced Accuracy | ≥85% | Specific diagnosis accuracy |
| **Multi-Class Diagnostic** | Rare Class Sensitivity | ≥80% | Foreign Bodies/Pseudo Membranes |
| **Combined System** | Expert Agreement | ≥90% | Clinical validation |
| **Combined System** | Cross-Dataset Consistency | <5% variance | Generalization validation |
| **Combined System** | Inference Time | <3 seconds | Clinical workflow integration |

### Clinical Integration Targets
- **Diagnostic Speed Improvement**: 50% reduction in time to diagnosis
- **Healthcare Cost Impact**: Measurable reduction in unnecessary specialist referrals
- **Clinical Utility Validation**: Positive impact on treatment decisions
- **User Satisfaction**: 85%+ satisfaction rating from healthcare professionals
- **False Referral Reduction**: Systematic decrease in inappropriate ENT referrals
