# 🔥 Wildfire Prediction System (Multi‑Modal ML Framework)

## Overview
This repository contains a **comprehensive, end‑to‑end wildfire prediction and analysis system** developed as part of **ISyE 6740 (Spring 2021), Georgia Tech**. The project integrates **computer vision, meteorological time‑series modeling, and multi‑modal machine learning** to move wildfire management from a *reactive* to a *proactive* paradigm.

The system is intentionally designed to be **drone‑centric**, scalable, and modular, enabling:
- Early wildfire risk assessment
- Localized ignition detection
- Fuel mapping and semantic understanding
- Fire severity prediction
- Decision‑ready operational integration

---

## Problem Statement
Wildfires pose a growing global threat due to climate change, extended droughts, and increasing human‑wildland interaction. Traditional wildfire detection systems rely heavily on:
- Infrequent satellite passes
- Manual observation
- Static ground sensors

These approaches suffer from **latency, limited spatial resolution, and operational risk**.

### Project Vision
This project proposes a **multi‑model wildfire prediction pipeline** that leverages:
- Drone‑captured RGB & thermal imagery
- High‑frequency meteorological and soil data
- Advanced machine learning architectures

The immediate implementation focuses on **proof‑of‑concept validation using publicly available datasets**, while architecturally preparing for real‑time drone deployment.

---

## Key Innovations

### 1. Intelligent Synthetic Heatmap Ground Truth
Real pixel‑level pre‑ignition wildfire labels are extremely scarce. To overcome this, the project introduces:
- **Synthetic heatmap generation** derived from thermal intensity peaks
- Gaussian‑blurred risk regions with spatial perturbations
- Zero‑risk heatmaps for non‑fire images

This enables **directional learning** for early fire detection without requiring real ground‑truth heatmaps.

### 2. Multi‑Modal Fusion Architecture
The system fuses:
- RGB visual cues
- Thermal infrared intensity
- Meteorological and soil conditions

This allows the models to capture both **visual ignition patterns** and **environmental fire drivers**.

### 3. Advanced Time‑Series Feature Engineering
Meteorological data is enhanced using:
- Lagged features (1h → 48h)
- Rolling window statistics
- Fire Weather Index (FWI)‑inspired composite features

### 4. Ensemble‑Driven Robustness
Multiple specialized models are trained independently and integrated operationally:
- Heatmap prediction (CNN)
- Semantic segmentation (U‑Net)
- Fire occurrence classification (GBMs)
- Fire severity regression (GBMs)

---

## Repository Structure
```
├── Image_Preprocessing_Standardized.ipynb
├── Heatmap_Classification_Prediction.ipynb
├── Visual_Semantic_Segmentation.ipynb
├── Meteorological_Classification.ipynb
├── Meteorological_Regression.ipynb
├── data/
│   ├── images/
│   ├── masks/
│   ├── weather/
├── processed_flame3_dataset/
│   ├── Fire/
│   └── No_Fire/
└── README.md
```

---

## Data Sources

### 1. Hourly Weather Dataset
Used for meteorological classification and regression.
- Temperature, humidity, wind, precipitation
- Soil moisture and soil temperature
- Binary fire occurrence label
- Continuous fire severity target

### 2. FLAME 3 Dataset (UAV Imagery)
Multi‑spectral drone imagery containing:
- Raw RGB images (4000×3000)
- Aligned RGB images (640×512)
- Thermal JPGs
- Radiometric thermal TIFFs

### 3. Forest Cover Type Dataset (UCI)
Used for environmental and fuel‑type contextual understanding.

### 4. Custom Annotated Segmentation Dataset
- RGB images + binary masks
- Used for semantic segmentation (fuel / fire regions)

### 5. External Reference Sources
- NASA FIRMS
- USGS Wildland Fire Science
- National Interagency Fire Center (NIFC)

---

## Image Preprocessing Pipeline
**Notebook:** `Image_Preprocessing_Standardized.ipynb`

This step standardizes all visual inputs for downstream models.

### Steps
1. **RGB–Thermal Pairing**
   - Images matched via filename identifiers

2. **Thermal Regeneration**
   - Red + green channel intensity used as heat proxy
   - Resized to 640×512

3. **RGB Alignment**
   - Prefer corrected FOV RGB images
   - Fallback: resize + center crop

4. **Dataset Structuring**
   - Outputs written to `Fire/` and `No_Fire/` folders

### Outcome
- Uniform spatial resolution
- Consistent thermal‑visual alignment
- Ready‑to‑train dataset

---

## Visual Heatmap Prediction Model
**Notebook:** `Heatmap_Classification_Prediction.ipynb`

### Purpose
Predict **localized fire probability heatmaps** from drone imagery.

### Inputs
- RGB images (aligned)
- Thermal images (grayscale)
- Patch size: 32×32

### Output
- Image‑sized probability heatmap
- Each grid cell ∈ [0,1]

### Architecture
- Dual‑branch CNN
  - RGB branch
  - Thermal branch
- Feature concatenation
- Fully connected fusion layers
- Sigmoid output

### Synthetic Ground Truth Logic
- Fire images: hotspots generated near thermal centroids
- No‑fire images: zero heatmaps

---

## Visual Semantic Segmentation
**Notebook:** `Visual_Semantic_Segmentation.ipynb`

### Purpose
Pixel‑level understanding for:
- Fuel mapping
- Fire boundary detection
- Post‑fire damage assessment

### Model
**U‑Net Architecture**
- Encoder‑decoder with skip connections

### Inputs
- RGB images (192×192)
- Binary masks

### Outputs
- Binary segmentation mask

### Techniques
- Data augmentation
- Nearest‑neighbor mask resizing
- Sigmoid output activation

---

## Meteorological Fire Classification
**Notebook:** `Meteorological_Classification.ipynb`

### Purpose
Binary classification: *Fire vs No Fire*

### Key Engineering
- Temporal features (hour, day, month)
- Lag features (up to 48h)
- Rolling statistics
- FWI‑like indices

### Critical Insight
**Severity was removed from features** due to:
- ≈0.995 correlation with fire occurrence
- Confirmed data leakage

### Class Imbalance Handling
- SMOTE applied on training data

### Models Evaluated
- Random Forest
- XGBoost
- LightGBM (Best performer)

---

## Meteorological Severity Regression
**Notebook:** `Meteorological_Regression.ipynb`

### Purpose
Predict continuous wildfire severity.

### Target Characteristics
- Highly zero‑inflated
- Long‑tailed distribution

### Inputs
- Same features as classification
- Fire occurrence included as categorical predictor

### Models
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Evaluation
- Overall performance
- Non‑zero severity focus

---

## Statistical Analysis

Performed to:
- Understand feature distributions
- Validate physical interpretability
- Guide model selection

Key Findings:
- Tree‑based models outperform linear models
- Temperature & dryness positively correlated
- Humidity & soil moisture negatively correlated

---

## Multi‑Model Operational Framework

### Phase 1: Pre‑Ignition Risk Assessment
- Meteorological classification
- Severity regression

### Phase 2: Early Detection
- Heatmap prediction from drone imagery

### Phase 3: Active Fire Management
- Semantic segmentation
- Resource prioritization

### Phase 4: Post‑Fire Analysis
- Damage assessment
- Recovery planning

### Unified Output
- Single‑cause risk prediction
- Decision‑ready intelligence layer

---

## Conclusion
This project demonstrates a **scalable, modular, and data‑driven wildfire prediction framework** that:
- Combines vision + time‑series ML
- Addresses real‑world data limitations
- Aligns with modern drone‑based operations

The architecture is production‑ready in design and extensible for real‑time deployment.

---

## Future Work
- Real‑time drone video ingestion
- Geospatial GIS integration
- Transformer‑based temporal modeling
- Reinforcement learning for resource allocation
- MLOps pipelines for deployment

---

## Author
**Alok Pratap Singh**  
Department of Analytics  
Georgia Institute of Technology

---

*Dedicated to firefighters and communities impacted by wildfires worldwide.*

