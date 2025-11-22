╔═══════════════════════════════════════════════════════════════════════════════════╗
║                           WATER QUALITY PREDICTION SYSTEM                         ║
║                    Using Sentinel-2 Satellite Data & Transformer ML               ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                PHASE 1: INPUT SETUP                              │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────┐
    │   User Specifies    │
    │  - ROI Coordinates  │
    │  - Start Date       │
    │  - End Date         │
    │  - Observation Freq │
    └──────────┬──────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Validate Inputs & Initialize GEE   │
    │  (Google Earth Engine API)           │
    └──────────┬───────────────────────────┘
               │
               
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: SATELLITE DATA ACQUISITION                          │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────────────────┐
    │  Query Sentinel-2 ImageCollection via GEE │
    │  - Filter by Region of Interest (ROI)     │
    │  - Filter by Date Range                   │
    │  - Filter by Cloud Cover (< 60%)          │
    │  - Select Spectral Bands                  │
    │    • B2 (Blue):         490 nm            │
    │    • B3 (Green):        560 nm            │
    │    • B4 (Red):          665 nm            │
    │    • B8 (NIR):          842 nm            │
    └──────────┬────────────────────────────────┘
               │
               ▼
    ┌───────────────────────────────────────────┐
    │   Filter & Download Available Scenes      │
    │   Output: Raw Sentinel-2 Images           │
    └──────────┬────────────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 3: IMAGE PREPROCESSING & CALIBRATION                    │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────┐
    │   Radiometric Calibration               │
    │  - Convert raw Digital Numbers (DN)     │
    │  - To Top-Of-Atmosphere (TOA) Radiance  │
    │  - Apply sensor-specific calibration    │
    │    coefficients                         │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────┐
    │   Geometric Corrections                 │
    │  - Refine sensor position & orientation │
    │  - Align using ground control points    │
    │  - Apply elevation model corrections    │
    │  - Orthorectification                   │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────┐
    │   Atmospheric Correction (Sen2Cor)      │
    │  - Remove atmospheric scattering        │
    │  - Correct for water vapor absorption   │
    │  - Output: Bottom-Of-Atmosphere (BOA)   │
    │    Surface Reflectance (0-1 normalized) │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────┐
    │   Output: Calibrated Band Images        │
    │   B2, B3, B4, B8 (Surface Reflectance)  │
    └──────────┬────────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                   PHASE 4: FEATURE EXTRACTION & ENGINEERING                      │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────┐
    │   Spatial Statistics Computation         │
    │  - Calculate mean spectral values        │
    │    over Region of Interest (ROI)         │
    │  - Per timestamp per band:               │
    │    B2_mean, B3_mean, B4_mean, B8_mean   │
    └──────────┬─────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │   Spectral Index Calculation            │
    │  - NDVI = (B8 - B4) / (B8 + B4)         │
    │    (Vegetation vigor indicator)         │
    │  - NDWI = (B3 - B8) / (B3 + B8)         │
    │    (Water/moisture indicator)           │
    │  - Turbidity_Index = B4 / B3            │
    │    (Water clarity proxy)                │
    └──────────┬─────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────┐
    │   Assemble Feature Matrix               │
    │  Columns: [B2, B3, B4, B8, NDVI, NDWI,  │
    │            Turbidity_Index, date]       │
    │  Rows: One row per timestamp (1 image)  │
    └──────────┬─────────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│              PHASE 5: DATA VALIDATION & QUALITY CONTROL                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────┐
    │   Data Cleaning                  │
    │  - Drop missing values (NaN)     │
    │  - Remove duplicate timestamps   │
    │  - Handle outliers               │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   Imputation (if needed)         │
    │  - Fill missing values by mean   │
    │  - Forward/Backward fill for     │
    │    time series continuity        │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   Output: Validated Features CSV │
    │   (processed_features.csv)       │
    └──────────┬───────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│            PHASE 6: LABEL GENERATION (SUPERVISED TRAINING - OPTIONAL)            │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────┐
    │  Query Landsat-8 Imagery via GEE   │
    │  (Select bands: SR_B3, SR_B4,      │
    │   SR_B5, SR_B6)                    │
    └──────────┬─────────────────────────┘
               │
               ▼
    ┌────────────────────────────────────┐
    │  Estimate Water Quality Labels     │
    │  Using Band Ratios:                │
    │  - TSS ≈ B5 / B4                   │
    │  - Turbidity ≈ B6 / B5             │
    │  - Chlorophyll ≈ NDVI(B5,B4)       │
    └──────────┬─────────────────────────┘
               │
               ▼
    ┌────────────────────────────────────┐
    │  Temporal Merge: Features + Labels │
    │  - Match nearest label within      │
    │    tolerance window (±12 days)     │
    │  - Drop unmatched rows             │
    │  - Output: Merged Dataset          │
    │    (labels.csv)                    │
    └──────────┬─────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                  PHASE 7: PREPROCESSING FOR MACHINE LEARNING                    │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────┐
    │   Feature Normalization          │
    │  - StandardScaler: Scale features│
    │    to mean=0, std=1              │
    │  - Save scaler for inference     │
    │    consistency                   │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   Label Processing               │
    │  - Clip negative values to 0     │
    │    (physical constraints)        │
    │  - Turbidity, Chlorophyll ≥ 0    │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   Train/Validation Split         │
    │  - Temporal split (80/20)        │
    │  - Preserve time series order    │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │   Output:                        │
    │  - features_for_training.csv     │
    │  - labels_for_training.csv       │
    │  - feature_scaler.pkl            │
    └──────────┬───────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 8: MODEL ARCHITECTURE & TRAINING                       │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────┐
    │   HydroTransNet Architecture                │
    │   ┌───────────────────────────────────────┐ │
    │   │ Input Layer:                          │ │
    │   │ - Features: [B2, B3, B4, B8, NDVI,   │ │
    │   │              NDWI, Turbidity_Index]   │ │
    │   │ - Sequence Length: 10 timesteps       │ │
    │   │ - Shape: (seq_len, batch, 7 features)│ │
    │   └───────────────────────────────────────┘ │
    │                    │                        │
    │                    ▼                        │
    │   ┌───────────────────────────────────────┐ │
    │   │ Positional Encoding (PE)              │ │
    │   │ - Encode temporal order information   │ │
    │   │ - Add temporal context to each step   │ │
    │   └───────────────────────────────────────┘ │
    │                    │                        │
    │                    ▼                        │
    │   ┌───────────────────────────────────────┐ │
    │   │ Transformer Encoder Stack             │ │
    │   │ - Layers: 8                           │ │
    │   │ - d_model: 128                        │ │
    │   │ - Attention Heads: 4                  │ │
    │   │ - Feed-Forward Dim: 512               │ │
    │   │ - Dropout: 0.01 (regularization)      │ │
    │   │                                       │ │
    │   │ Each Layer:                           │ │
    │   │ 1. Multi-Head Self-Attention          │ │
    │   │    (Learn temporal dependencies)      │ │
    │   │ 2. Feed-Forward Network               │ │
    │   │    (Non-linear transformations)       │ │
    │   │ 3. Layer Normalization & Residuals    │ │
    │   └───────────────────────────────────────┘ │
    │                    │                        │
    │                    ▼                        │
    │   ┌───────────────────────────────────────┐ │
    │   │ Global Average Pooling                │ │
    │   │ - Aggregate sequence information      │ │
    │   │ - Output: (batch, d_model=128)        │ │
    │   └───────────────────────────────────────┘ │
    │                    │                        │
    │                    ▼                        │
    │   ┌───────────────────────────────────────┐ │
    │   │ Dense Output Layer                    │ │
    │   │ - Hidden: 64 neurons                  │ │
    │   │ - ReLU Activation                     │ │
    │   │ - Output: 3 neurons (predictions)     │ │
    │   │   • TSS (mg/L)                        │ │
    │   │   • Turbidity (NTU)                   │ │
    │   │   • Chlorophyll (μg/L)                │ │
    │   └───────────────────────────────────────┘ │
    └─────────────────────────────────────────────┘
               │
               ▼
    ┌───────────────────────────────────────┐
    │   Training Configuration              │
    │  - Loss Function: MSE                 │
    │  - Optimizer: Adam (lr=1e-3)          │
    │  - Batch Size: 32                     │
    │  - Epochs: 100                        │
    │  - Scheduler: ReduceLROnPlateau       │
    │  - Early Stopping: Monitor val_loss   │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌───────────────────────────────────────┐
    │   Training Loop:                      │
    │  1. Forward pass through network      │
    │  2. Compute loss (predictions vs     │
    │     actual labels)                    │
    │  3. Backward pass (gradients)         │
    │  4. Update weights via optimizer      │
    │  5. Validate on test set              │
    │  6. Save best model checkpoint        │
    └──────────┬────────────────────────────┘
               │
               ▼
    ┌───────────────────────────────────────┐
    │   Output: Trained Model               │
    │  - best_model.pt (checkpoint)         │
    │  - training_history.pkl (logs)        │
    └──────────┬────────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      PHASE 9: MODEL INFERENCE & PREDICTION                      │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────┐
    │   Load Trained Model & Scaler        │
    │  - Load best_model.pt weights        │
    │  - Load feature_scaler.pkl           │
    │  - Set model to eval mode            │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   Prepare New/Test Data              │
    │  - Fetch fresh Sentinel-2 data       │
    │  - Extract same features (B2-B8,     │
    │    NDVI, NDWI, Turbidity_Index)      │
    │  - Apply same preprocessing steps    │
    │  - Scale using saved scaler          │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   Create Sequence Windows            │
    │  - Segment data into sequences of    │
    │    length 10 (seq_len)               │
    │  - Each sequence represents temporal │
    │    context for prediction            │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   Forward Pass (No Gradient)         │
    │  - Pass each sequence through model  │
    │  - torch.no_grad() for efficiency    │
    │  - Output predictions (TSS, Turb,    │
    │    Chlorophyll) for each sequence    │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   Post-Processing                    │
    │  - Align predictions with dates      │
    │  - Extract corresponding NDVI, NDWI │
    │  - Format results as JSON/CSV        │
    └──────────┬───────────────────────────┘
               │

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 10: OUTPUT & VISUALIZATION                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────┐
    │   Prediction Results Per Timestamp:  │
    │  {                                   │
    │   "date": "2023-05-09",              │
    │   "TSS mg/L": 0.122,                 │
    │   "Turbidity NTU": 0.056,            │
    │   "Chlorophyll ug/L": 0.026,         │
    │   "NDVI": -1.683,                    │
    │   "NDWI": 1.701                      │
    │  }                                   │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   Visualization & Export             │
    │  - Time series plots                 │
    │  - Spatial maps (if multi-point ROI) │
    │  - Statistical summaries             │
    │  - CSV/JSON exports                  │
    │  - Web Dashboard (FastAPI)           │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │   User-Facing API Endpoints          │
    │  - /predict (custom date range)      │
    │  - /currdate (last week)             │
    │  - /predict_quick (pre-processed)    │
    └──────────────────────────────────────┘


╔═════════════════════════════════════════════════════════════════════════════════╗
║                            KEY TECHNOLOGIES USED                                 ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║ • Google Earth Engine (GEE)      - Satellite data access & processing            ║
║ • Sentinel-2 Satellite           - Multi-spectral imagery (13 bands)             ║
║ • Python & PyTorch              - Deep learning & data manipulation              ║
║ • Transformer Architecture      - Sequence modeling for time series              ║
║ • FastAPI                       - REST API for predictions                       ║
║ • Scikit-Learn                  - Preprocessing & feature engineering            ║
║ • Docker & AWS                  - Containerization & cloud deployment            ║
╚═════════════════════════════════════════════════════════════════════════════════╝
