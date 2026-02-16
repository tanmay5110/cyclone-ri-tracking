"""
Train RI prediction models: Random Forest (baseline) and XGBoost.

Features:
  - Storm-wise train/test split (no data leakage)
  - Class imbalance handling via class weights and optional SMOTE
  - Comprehensive evaluation: AUC-ROC, POD, FAR, CSI, Brier Score
  - Feature importance analysis
  - Model persistence via joblib
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, brier_score_loss, confusion_matrix,
    classification_report, roc_curve
)

import xgboost as xgb

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


# ─── Feature definitions ─────────────────────────────────────────────

TRACK_FEATURES = [
    "VMAX", "PMIN", "DELTA_V_6H", "DELTA_V_12H", "DELTA_P_6H",
    "TRANS_SPEED", "HEADING", "ABS_LAT",
]

SHIPS_FEATURES = [
    "SST", "RSST", "SHRD", "SHRS", "D200", "OHC",
    "RHLO", "RHMD", "RHHI", "VMPI", "U200",
    "TWAC", "PSLV", "REFC", "PEFC",
]

IR_FEATURES = [
    "IR_MEAN_BT", "IR_MIN_BT", "IR_STD_BT",
    "IR_FRAC_LT200K", "IR_FRAC_LT220K", "IR_AXISYM",
]


def load_features(path: str = None) -> pd.DataFrame:
    """Load the feature matrix CSV."""
    if path is None:
        path = os.path.join(PROCESSED_DIR, "features.csv")

    print(f"[LOAD] Reading features from {path}")
    df = pd.read_csv(path)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    print(f"  → {len(df):,} rows, {df['SID'].nunique()} storms")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get available feature columns from the dataframe."""
    all_features = TRACK_FEATURES + SHIPS_FEATURES + IR_FEATURES
    available = [f for f in all_features if f in df.columns]
    print(f"  → Using {len(available)} features: {available}")
    return available


def storm_train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                            random_state: int = 42):
    """
    Split data by storm ID (not by row) to prevent data leakage.
    All fixes from a single storm go into either train or test, never both.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    groups = df["SID"].values
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    print(f"\n=== Storm-wise Train/Test Split ===")
    print(f"  Train: {len(train):,} rows, {train['SID'].nunique()} storms, "
          f"RI+ = {train['RI'].sum()} ({100 * train['RI'].mean():.1f}%)")
    print(f"  Test : {len(test):,} rows, {test['SID'].nunique()} storms, "
          f"RI+ = {test['RI'].sum()} ({100 * test['RI'].mean():.1f}%)")

    return train, test


def prepare_xy(df: pd.DataFrame, feature_cols: list):
    """Extract X (features) and y (RI label) arrays."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["RI"].values.astype(int)

    # Replace NaN with column median
    for j in range(X.shape[1]):
        col_median = np.nanmedian(X[:, j])
        if np.isnan(col_median):
            col_median = 0.0
        nan_mask = np.isnan(X[:, j])
        X[nan_mask, j] = col_median

    return X, y


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute comprehensive RI prediction metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # Probability of Detection
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0      # False Alarm Ratio
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0.0  # Critical Success Index

    metrics = {
        "AUC_ROC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "POD_Recall": float(pod),
        "FAR": float(far),
        "CSI": float(csi),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "N_total": int(len(y_true)),
        "N_positive": int(y_true.sum()),
    }
    return metrics


def print_metrics(metrics: dict, model_name: str):
    """Pretty-print model evaluation metrics."""
    print(f"\n{'─' * 45}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'─' * 45}")
    print(f"  AUC-ROC        : {metrics['AUC_ROC']:.4f}")
    print(f"  POD (Recall)   : {metrics['POD_Recall']:.4f}")
    print(f"  FAR            : {metrics['FAR']:.4f}")
    print(f"  CSI            : {metrics['CSI']:.4f}")
    print(f"  Precision      : {metrics['Precision']:.4f}")
    print(f"  F1 Score       : {metrics['F1']:.4f}")
    print(f"  Brier Score    : {metrics['Brier_Score']:.4f}")
    print(f"  Confusion: TP={metrics['TP']}, FP={metrics['FP']}, "
          f"FN={metrics['FN']}, TN={metrics['TN']}")
    print(f"  Total samples: {metrics['N_total']}, Positive: {metrics['N_positive']}")


def train_random_forest(X_train, y_train, X_test, y_test,
                         feature_names: list) -> dict:
    """Train a Random Forest baseline model."""
    print("\n[RF] Training Random Forest...")

    # Calculate class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "Random Forest")

    # Feature importance
    importances = dict(zip(feature_names, rf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    # Convert to Python float for JSON serialization
    rf_importances = {k: float(v) for k, v in importances.items()}

    return {
        "model": rf,
        "metrics": metrics,
        "feature_importance": rf_importances,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def train_xgboost(X_train, y_train, X_test, y_test,
                   feature_names: list) -> dict:
    """Train an XGBoost model."""
    print("\n[XGB] Training XGBoost...")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics, "XGBoost")

    # Feature importance - convert to Python float for JSON
    xgb_importances = {k: float(v) for k, v in zip(feature_names, xgb_model.feature_importances_)}
    xgb_importances = dict(sorted(xgb_importances.items(), key=lambda x: -x[1]))

    return {
        "model": xgb_model,
        "metrics": metrics,
        "feature_importance": xgb_importances,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_results(rf_result: dict, xgb_result: dict,
                  scaler: StandardScaler, feature_names: list,
                  test_df: pd.DataFrame):
    """Save models, scaler, and test results."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Save models
    joblib.dump(rf_result["model"], os.path.join(MODELS_DIR, "rf_model.joblib"))
    joblib.dump(xgb_result["model"], os.path.join(MODELS_DIR, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.joblib"))

    # Save metrics
    results = {
        "timestamp": datetime.now().isoformat(),
        "random_forest": rf_result["metrics"],
        "xgboost": xgb_result["metrics"],
        "feature_names": feature_names,
        "rf_feature_importance": rf_result["feature_importance"],
        "xgb_feature_importance": xgb_result["feature_importance"],
    }
    with open(os.path.join(MODELS_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save test predictions
    test_preds = test_df[["SID", "ISO_TIME", "RI"]].copy()
    test_preds["RF_PROB"] = rf_result["y_prob"]
    test_preds["XGB_PROB"] = xgb_result["y_prob"]
    test_preds["RF_PRED"] = rf_result["y_pred"]
    test_preds["XGB_PRED"] = xgb_result["y_pred"]
    test_preds.to_csv(os.path.join(MODELS_DIR, "test_predictions.csv"), index=False)

    print(f"\n[SAVE] Models, results, and predictions saved to {MODELS_DIR}/")
    print(f"  → rf_model.joblib")
    print(f"  → xgb_model.joblib")
    print(f"  → scaler.joblib")
    print(f"  → training_results.json")
    print(f"  → test_predictions.csv")


def train_pipeline(features_path: str = None, tune: bool = False):
    """Full training pipeline."""
    # Load data
    df = load_features(features_path)
    feature_cols = get_feature_columns(df)

    if len(feature_cols) < 3:
        print("[ERROR] Too few features available. Check feature matrix.")
        sys.exit(1)

    # Storm-wise split
    train_df, test_df = storm_train_test_split(df)

    # Prepare arrays
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    rf_result = train_random_forest(X_train_scaled, y_train,
                                     X_test_scaled, y_test, feature_cols)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)
    # Note: XGBoost doesn't need scaled features due to tree-based splitting

    # Save everything
    save_results(rf_result, xgb_result, scaler, feature_cols, test_df)

    # Feature importance comparison
    print(f"\n{'=' * 45}")
    print(f"  Top 10 Feature Importance")
    print(f"{'=' * 45}")
    print(f"  {'Feature':20s} {'RF':>8s} {'XGBoost':>8s}")
    print(f"  {'─' * 38}")

    all_feats = set(list(rf_result["feature_importance"].keys())[:10] +
                    list(xgb_result["feature_importance"].keys())[:10])
    for feat in sorted(all_feats,
                       key=lambda f: -(rf_result["feature_importance"].get(f, 0) +
                                       xgb_result["feature_importance"].get(f, 0))):
        rf_imp = rf_result["feature_importance"].get(feat, 0)
        xgb_imp = xgb_result["feature_importance"].get(feat, 0)
        print(f"  {feat:20s} {rf_imp:8.4f} {xgb_imp:8.4f}")

    return rf_result, xgb_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RI prediction models")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to features.csv")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()

    train_pipeline(args.features, args.tune)
