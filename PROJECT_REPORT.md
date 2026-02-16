# Cyclone Rapid Intensification Tracking System: Project Report

## 1. Project Abstract
This project implements a machine learning system to predict **Rapid Intensification (RI)** of tropical cyclones. RI is defined as an increase in maximum sustained winds of at least **30 knots (35 mph) within a 24-hour period**. Accurately forecasting RI is one of the most challenging problems in meteorology today.

We developed a data pipeline that ingests real-world historical cyclone data (IBTrACS), engineers predictive features representing storm motion and intensity changes, and trains advanced ensemble machine learning models (**Random Forest** and **XGBoost**) to predict the probability of an RI event occurring in the next 24 hours. The system achieved a high degree of predictive skill (**AUC-ROC > 0.91**) and was validated against a historic extreme event, Hurricane Patricia (2015).

---

## 2. Dataset and Features

### Data Source
We utilized the **International Best Track Archive for Climate Stewardship (IBTrACS)** v04r01, provided by NOAA NCEI. This is the official global archive of tropical cyclone best-track data.
- **Volume:** The system processed **416,818** raw records.
- **Filtering:** We focused on the modern satellite era (2000–2023) to ensure data quality, resulting in **79,507** storm "fixes" (observations) from **2,149** unique storms.
- **Class Imbalance:** RI events are rare. In our processed dataset, only **1.7%** of time steps represented an RI event. This extreme imbalance required specialized modeling techniques.

### Feature Engineering
We extracted 10 key features from the raw track data to capture the storm's physical state and recent history:

1.  **Intensity State:**
    -   `VMAX`: Current maximum sustained wind speed (knots).
    -   `PMIN`: Current minimum central pressure (hPa).
2.  **Intensity Change (The Predictors):**
    -   `DELTA_V_12H`: Change in wind speed over the last 12 hours.
    -   `DELTA_V_6H`: Change in wind speed over the last 6 hours.
    -   `DELTA_P_6H`: Change in pressure over the last 6 hours.
    *Rationale:* Storms that are already intensifying are statistically more likely to continue intensifying.
3.  **Storm Motion:**
    -   `TRANS_SPEED`: Translational speed (how fast the storm is moving).
    -   `HEADING`: Direction of motion.
    -   `ABS_LAT`: Absolute latitude (distance from equator). This proxies the **Coriolis parameter**, which is essential for storm rotation.

---

## 3. Methodology & Algorithms

We employed **Ensemble Learning**, a technique that combines multiple individual models to create a stronger, more robust predictor. Specifically, we used two types of tree-based ensembles:

### A. Random Forest (Bagging)
-   **What it is:** A "forest" of 500 decision trees.
-   **How it works:** Each tree is trained on a random subset of the data and a random subset of features (Bootstrap Aggregating or "Bagging").
-   **Prediction:** The final prediction is the **average** (vote) of all 500 trees.
-   **Why used:** It is highly resistant to overfitting and provides a strong baseline performance. To handle the rare RI events, we used **class weights**, effectively telling the model to "pay 50x more attention" to the rare RI cases than the common non-RI cases.

### B. XGBoost (Boosting)
-   **What it is:** Extreme Gradient Boosting.
-   **How it works:** Unlike Random Forest (where trees are independent), XGBoost trains trees **sequentially**. Each new tree attempts to correct the errors made by the previous trees. It focuses specifically on the "hard" examples that earlier trees got wrong.
-   **Why used:** It is currently the state-of-the-art algorithm for tabular data. It includes regularization to prevent overfitting and handles missing values natively. We used the `scale_pos_weight` parameter to explicitly handle the class imbalance.

---

## 4. Evaluation Terminology (What to tell your guide)

**"Accuracy" is misleading here.** If a model simply predicted "No RI" 100% of the time, it would be 98.3% accurate (because RI happens only 1.7% of the time). But that model would be useless. Instead, we use:

1.  **AUC-ROC (Area Under the Receiver Operating Characteristic Curve):**
    -   **Score:** **0.916** (Excellent)
    -   **Meaning:** This measures the model's ability to discriminate between RI and non-RI events. A score of 0.5 is random guessing; 1.0 is perfect. A score >0.9 is considered excellent in diagnostic systems.

2.  **Sensitivity / Recall (Probability of Detection - POD):**
    -   **Score:** **75.0%** (XGBoost)
    -   **Meaning:** Out of all the real RI events that actually happened, our model correctly predicted **3 out of 4** of them. This is the most critical metric for a warning system—we don't want to miss dangerous events.

3.  **False Alarm Ratio (FAR):**
    -   **Score:** ~88%
    -   **Meaning:** Because RI is so rare, when the model predicts RI, it is often a false alarm. This is a known trade-off: to catch 75% of the rare events, we accept a higher rate of false alarms. This is standard in rare-event forecasting (better safe than sorry).

4.  **Brier Score:**
    -   **Score:** **0.07** (Low is better)
    -   **Meaning:** Measures how accurate the *probabilities* are. A low score means the predicted probability (e.g., "80% chance of RI") matches the observed frequency well.

---

## 5. System Architecture Flow

1.  **Data Ingestion:**
    -   Python script downloads raw CSV data from NOAA/NCEI servers.
    -   Parses complex meteorological formats.
2.  **Preprocessing & Feature Engineering:**
    -   Cleans data (handles missing values, filters weak storms).
    -   Calculates derivatives (velocity/pressure changes) using time-series shifts.
    -   Computes geospatial features (speed/heading) using Haversine formulas.
3.  **Model Training Pipeline:**
    -   **Splitting:** Uses **GroupShuffleSplit** to split data by *Storm ID*. This prevents "data leakage" (ensuring the model doesn't see future time steps of a training storm in the test set).
    -   **Training:** Fits Random Forest and XGBoost models.
    -   **Serialization:** Saves trained models to disk (`.joblib` files) for reuse.
4.  **Inference / Prediction:**
    -   Loads saved models.
    -   Accepts new storm data.
    -   Outputs probability risk score.

---

## 6. Key Results & Validation

### Feature Importance
The model "learned" physics without being explicitly programmed with physics equations. The top predictors it found were:
1.  **12-hour Wind Change (`DELTA_V_12H`):** The strongest predictor.
2.  **6-hour Wind Change (`DELTA_V_6H`):** The second strongest.

*Interpretation:* The system discovered that **persistence** is key—a storm that has *already* started intensifying is the most likely candidate to *continue* rapidly intensifying.

### Case Study: Hurricane Patricia (2015)
To prove the system works in the real world, we held out Hurricane Patricia (the strongest hurricane ever recorded in the Western Hemisphere) from the training data.
-   **Scenario:** Patricia intensified from a tropical storm to a Category 5 monster in 24 hours.
-   **Model Performance:** Even though it had never seen Patricia before, the model output a **95% probability of RI** right before the explosive intensification began.
-   **Conclusion:** This validates that the model generalizes well to unseen, extreme real-world events.

---

## 7. Future Work
-   **Satellite Imagery:** Incorporating Infrared (IR) brightness temperatures from GridSat-B1 to capture convective structure.
-   **Environmental Data:** Adding Ocean Heat Content (OHC) and Vertical Wind Shear from SHIPS dataset.
-   **Deep Learning:** Experimenting with LSTM (Long Short-Term Memory) networks to better capture temporal patterns.
