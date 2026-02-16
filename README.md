# Cyclone Rapid Intensification (RI) Tracking System

A machine-learning pipeline that predicts tropical cyclone rapid intensification
using **only real, publicly available data** — no synthetic data at any stage.

## What is Rapid Intensification?

Rapid Intensification (RI) is defined as an increase of ≥ 30 knots in a tropical
cyclone's maximum sustained winds within 24 hours (Kaplan & DeMaria, 2003).
RI events are among the most dangerous and difficult-to-forecast phenomena in
tropical meteorology.

## Data Sources (All Real)

| Dataset | Source | Description |
|---------|--------|-------------|
| **IBTrACS v04r01** | NOAA NCEI | Global best-track cyclone records (1842–present) |
| **SHIPS Predictors** | Colorado State / RAMMB | Pre-computed environmental variables (SST, wind shear, OHC, etc.) |
| **GridSat-B1** | NOAA NCEI CDR | Satellite IR brightness temperatures (0.07°, 3-hourly) |

## Project Structure

```
ri2/
├── data/
│   ├── raw/              # Downloaded IBTrACS, GridSat, SHIPS files
│   └── processed/        # features.csv (ML-ready matrix)
├── models/               # Saved models, scaler, results
├── notebooks/            # EDA notebook
├── src/
│   ├── data/             # Download scripts
│   ├── features/         # Processing & feature engineering
│   ├── models/           # Training & tuning
│   ├── evaluate/         # Case study evaluation
│   └── visualize/        # Dashboard & plots
├── outputs/              # Generated figures & reports
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download real data
```bash
python src/data/download_ibtracs.py     # ~330 MB, IBTrACS global best-track
python src/data/download_ships.py       # SHIPS environmental predictors
python src/data/download_gridsat.py --test  # GridSat IR for Patricia only
```

### 3. Build feature matrix
```bash
python src/features/build_dataset.py
```

### 4. Train models
```bash
python src/models/train.py
```

### 5. Run case study (Hurricane Patricia 2015)
```bash
python src/evaluate/case_study.py
```

### 6. Generate dashboard
```bash
python src/visualize/dashboard.py
```

## Model Features

### Track Features (from IBTrACS)
- Current intensity (VMAX) and pressure (PMIN)
- Intensity change over 6h/12h (DELTA_V_6H, DELTA_V_12H)
- Pressure change (DELTA_P_6H)
- Translational speed and heading
- Absolute latitude (Coriolis proxy)

### Environmental Features (from SHIPS)
- Sea surface temperature (SST)
- 200–850 hPa vertical wind shear (SHRD)
- Upper-level divergence (D200)
- Mid-level relative humidity (RHMD)
- Ocean heat content (OHC)
- Maximum potential intensity (VMPI)

### Satellite Features (from GridSat-B1)
- Mean/min IR brightness temperature
- Deep convection fraction (< 200K, < 220K)
- Convective organization (BT standard deviation)
- Axisymmetry index

## Models

1. **Random Forest** — Balanced baseline with 500 trees
2. **XGBoost** — Main model with scale_pos_weight for RI imbalance

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC-ROC | Overall discrimination |
| POD (Recall) | Probability of Detection — catches real RI events |
| FAR | False Alarm Ratio |
| CSI | Critical Success Index — balanced skill |
| Brier Score | Probabilistic calibration |

## Case Study: Hurricane Patricia (2015)

Hurricane Patricia underwent one of the most extreme RI events on record,
intensifying by **100 kt in just 24 hours** on 23 October 2015.
The model is evaluated on this held-out case to assess real-world skill.

## References

- Kaplan, J., & DeMaria, M. (2003). Large-scale characteristics of rapidly
  intensifying tropical cyclones in the North Atlantic basin.
  *Weather and Forecasting*, 18(6), 1093-1108.
- Knapp, K. R., et al. (2010). The International Best Track Archive for
  Climate Stewardship (IBTrACS). *Bull. Amer. Meteor. Soc.*, 91, 363-376.
- DeMaria, M., Mainelli, M., Shay, L. K., Knaff, J. A., & Kaplan, J. (2005).
  Further improvements to the Statistical Hurricane Intensity Prediction
  Scheme (SHIPS). *Weather and Forecasting*, 20(4), 531-543.
