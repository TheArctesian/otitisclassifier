graph TB
    %% Data Sources
    subgraph DS["📊 Multi-Source Dataset (~2000+ images)"]
        D1["Ebasaran-Kaggle<br/>956 images, 9 classes<br/>Primary Training"]
        D2["UCI-Kaggle<br/>~900+ images, 5 classes<br/>AOM/Normal reinforcement"]
        D3["VanAk-Figshare<br/>~270+ images, 7 classes<br/>External Validation"]
        D4["Sumotosima-GitHub<br/>38+ cases<br/>Clinical Text Descriptions"]
    end

    %% Data Preparation Phase
    subgraph DP["🔧 Data Preparation Pipeline"]
        DP1["Format Standardization<br/>TIFF/JPG/PNG → Unified"]
        DP2["CLAHE Enhancement<br/>Illumination Normalization"]
        DP3["Resize to 500×500<br/>Maintain Aspect Ratio"]
        DP4["Class Mapping<br/>9 Unified Categories"]
        DP5["Source-Aware Splitting<br/>Train: Ebasaran+UCI<br/>Val: VanAk<br/>Test: Holdout"]
    end

    %% Class Distribution
    subgraph CD["📈 Class Distribution & Strategy"]
        CD1["Well-Represented<br/>Normal: ~850<br/>AOM: ~700+<br/>Cerumen: ~400+"]
        CD2["Moderate<br/>CSOM: ~80+<br/>Otitis Externa: ~60+<br/>Myringosclerosis: ~35+"]
        CD3["Critical Shortage<br/>Ventilation Tubes: ~20+<br/>Pseudo Membranes: 11<br/>Foreign Bodies: 3"]
    end

    %% Augmentation Strategy
    subgraph AUG["🔄 Differential Augmentation"]
        AUG1["Conservative (2-3x)<br/>Normal, AOM, Cerumen<br/>Basic transforms only"]
        AUG2["Moderate (5-10x)<br/>CSOM, Otitis Externa<br/>+ Color jitter, blur"]
        AUG3["Aggressive (15-50x)<br/>Rare classes<br/>+ Elastic, grid distortion"]
        AUG4["Otoscope-Specific<br/>• Specular reflection<br/>• Ring illumination<br/>• Viewing angle"]
    end

    %% Model Architecture
    subgraph MA["🧠 Model Architecture"]
        MA1["Base Model<br/>DenseNet-121 or ResNet-50<br/>RadImageNet Pretrained"]
        MA2["Feature Extraction<br/>Global Average Pooling<br/>Dropout (0.3-0.5)"]
        MA3["Classification Head<br/>FC Layer → 9 classes<br/>Temperature Scaling"]
        MA4["Optional: Vision-Language<br/>BiomedCLIP/MedCLIP<br/>Clinical Descriptions"]
    end

    %% Training Strategy
    subgraph TS["🎯 Training Strategy"]
        TS1["Loss Function<br/>Focal Loss (γ=2)<br/>+ Class Weights"]
        TS2["Optimizer<br/>AdamW<br/>Layer-wise LR"]
        TS3["Scheduler<br/>Cosine Annealing<br/>Warmup: 5 epochs"]
        TS4["Progressive Unfreezing<br/>1. Head only (5 ep)<br/>2. Top layers (10 ep)<br/>3. Full model (15 ep)"]
    end

    %% Validation & Metrics
    subgraph VM["📊 Validation & Metrics"]
        VM1["Primary Metrics<br/>• Balanced Accuracy<br/>• Macro F1-Score<br/>• Class-wise Sensitivity"]
        VM2["Clinical Metrics<br/>• Path. Sensitivity >90%<br/>• Normal Specificity >85%<br/>• False Negative Analysis"]
        VM3["Cross-Dataset Val<br/>• Source-aware K-fold<br/>• External validation<br/>• Distribution shift detection"]
    end

    %% Interpretability
    subgraph INT["🔍 Clinical Interpretability"]
        INT1["Grad-CAM++<br/>Anatomical Focus Maps"]
        INT2["Confidence Calibration<br/>Temperature Scaling<br/>Platt Scaling"]
        INT3["Clinical Thresholds<br/>High: ≥0.85<br/>Medium: 0.65-0.84<br/>Low: <0.65"]
        INT4["Decision Explanation<br/>Feature importance<br/>Similar case retrieval"]
    end

    %% Integration Layer
    subgraph IL["🔗 Multi-Modal Integration"]
        IL1["Image CNN (40%)<br/>Primary diagnostic signal"]
        IL2["Symptoms (35%)<br/>iPad questionnaire<br/>(Future Phase)"]
        IL3["History (25%)<br/>Risk factors<br/>(Future Phase)"]
        IL4["Decision Engine<br/>Weighted combination<br/>Red flag detection"]
    end

    %% Deployment
    subgraph DEP["🚀 Production Deployment"]
        DEP1["Streamlit Interface<br/>Real-time inference<br/>Clinical dashboard"]
        DEP2["Docker Container<br/>GPU optimization<br/>Model versioning"]
        DEP3["Performance Targets<br/>• <3s inference<br/>• 100+ concurrent users<br/>• 99.9% uptime"]
        DEP4["Monitoring<br/>• Drift detection<br/>• Error logging<br/>• Clinical audit trail"]
    end

    %% Safety & Compliance
    subgraph SC["🛡️ Safety & Compliance"]
        SC1["Clinical Validation<br/>ENT specialist review<br/>Prospective study"]
        SC2["Regulatory<br/>FDA guidelines<br/>HIPAA compliance"]
        SC3["Safety Protocols<br/>• Red flag alerts<br/>• Human-in-loop<br/>• Conservative thresholds"]
    end

    %% Flow connections
    DS --> DP1
    DP1 --> DP2 --> DP3 --> DP4 --> DP5
    
    DP5 --> CD
    CD --> AUG
    CD1 --> AUG1
    CD2 --> AUG2
    CD3 --> AUG3
    
    AUG --> MA1
    MA1 --> MA2 --> MA3
    MA3 --> TS
    MA4 -.->|Optional| MA3
    
    TS --> VM
    VM --> INT
    
    INT --> IL1
    IL1 --> IL4
    IL2 -.->|Future| IL4
    IL3 -.->|Future| IL4
    
    IL4 --> DEP1
    DEP1 --> DEP2 --> DEP3 --> DEP4
    
    IL4 --> SC
    SC --> SC3
    SC3 -->|Feedback| TS

    %% Styling
    classDef critical fill:#ff6b6b,stroke:#c92a2a,color:#fff
    classDef moderate fill:#ffd43b,stroke:#fab005,color:#000
    classDef good fill:#51cf66,stroke:#2f9e44,color:#fff
    classDef future fill:#e8e8e8,stroke:#868e96,color:#495057,stroke-dasharray: 5 5
    classDef safety fill:#ff8787,stroke:#fa5252,color:#fff
    
    class CD3 critical
    class CD2 moderate
    class CD1 good
    class IL2,IL3,MA4 future
    class SC,SC1,SC2,SC3 safety