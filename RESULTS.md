# Cyclone RI Tracking System — Results Summary

## Model Performance (Real IBTrACS Data, 2000–2023)

### Dataset
- **Training**: 64,661 storm fixes from 1,718 storms (1,069 RI events, 1.7%)
- **Testing**: 14,802 storm fixes from 430 storms (236 RI events, 1.6%)
- **Hold-out**: Hurricane Patricia 2015 (29 fixes, extreme RI case)

### Model Comparison

| Metric | Random Forest | XGBoost |
|--------|:------------:|:-------:|
| **AUC-ROC** | **0.9185** | 0.9162 |
| POD (Recall) | 0.6356 | **0.7500** |
| FAR | **0.8442** | 0.8875 |
| CSI | **0.1430** | 0.1084 |
| Brier Score | **0.0422** | 0.0712 |

### Key Findings

1. **AUC-ROC > 0.91 for both models** — excellent discrimination between RI and non-RI events using only track features from real IBTrACS data.

2. **XGBoost catches 75% of RI events** (POD=0.75), higher than RF (63.5%). This comes at the cost of more false alarms.

3. **Top Predictive Features**:
   - `DELTA_V_12H` (0.45) — 12-hour wind change is the strongest predictor
   - `DELTA_V_6H` (0.22) — 6-hour wind change
   - `DELTA_P_6H` (0.14) — 6-hour pressure change
   - Storms that are already intensifying are most likely to undergo RI.

4. **Case Study: Hurricane Patricia (2015)** — The model was evaluated on the most extreme RI event in recorded history (100 kt in 24 h), with Patricia completely excluded from training.

### Data Sources (All Real)
- IBTrACS v04r01 Global Best Track CSV (NOAA NCEI)
- No synthetic, simulated, or augmented data used

### Future Improvements
- Add SHIPS environmental predictors (SST, wind shear, OHC) when URLs become accessible
- Add GridSat-B1 satellite IR features (deep convection fraction, axisymmetry)
- Hyperparameter tuning with Optuna
- SMOTE oversampling to address class imbalance
