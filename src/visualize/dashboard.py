"""
Visualization dashboard: 6-panel performance summary.

Panels:
  1. ROC curve (RF vs XGBoost)
  2. Feature importance (top 15)
  3. Reliability diagram (calibration)
  4. RI probability histogram
  5. Case-study intensity + probability timeline
  6. Confusion matrix heatmap
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")


def load_results():
    """Load training results and test predictions."""
    results_path = os.path.join(MODELS_DIR, "training_results.json")
    preds_path = os.path.join(MODELS_DIR, "test_predictions.csv")

    with open(results_path, "r") as f:
        results = json.load(f)

    preds = pd.read_csv(preds_path)
    preds["ISO_TIME"] = pd.to_datetime(preds["ISO_TIME"], errors="coerce")

    return results, preds


def create_dashboard():
    """Generate the 6-panel visualization dashboard."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    results, preds = load_results()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Cyclone Rapid Intensification — Model Performance Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)

    y_true = preds["RI"].values

    # ── Panel 1: ROC Curves ──
    ax = axes[0, 0]
    for model_name, prob_col, color in [
        ("Random Forest", "RF_PROB", "#2196F3"),
        ("XGBoost", "XGB_PROB", "#FF5722"),
    ]:
        if prob_col in preds.columns:
            fpr, tpr, _ = roc_curve(y_true, preds[prob_col])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{model_name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Feature Importance (Top 15) ──
    ax = axes[0, 1]
    xgb_imp = results.get("xgb_feature_importance", {})
    if xgb_imp:
        top_n = 15
        sorted_feats = sorted(xgb_imp.items(), key=lambda x: -x[1])[:top_n]
        feat_names = [f[0] for f in sorted_feats][::-1]
        feat_vals = [f[1] for f in sorted_feats][::-1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
        ax.barh(feat_names, feat_vals, color=colors)
        ax.set_xlabel("Importance (XGBoost)")
        ax.set_title("Feature Importance (Top 15)")
    ax.grid(True, alpha=0.3, axis="x")

    # ── Panel 3: Reliability Diagram ──
    ax = axes[0, 2]
    for model_name, prob_col, color in [
        ("RF", "RF_PROB", "#2196F3"),
        ("XGB", "XGB_PROB", "#FF5722"),
    ]:
        if prob_col in preds.columns:
            prob_true, prob_pred = calibration_curve(
                y_true, preds[prob_col], n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2,
                    label=model_name)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 4: RI Probability Histogram ──
    ax = axes[1, 0]
    if "XGB_PROB" in preds.columns:
        ri_probs = preds.loc[y_true == 1, "XGB_PROB"]
        non_ri_probs = preds.loc[y_true == 0, "XGB_PROB"]
        ax.hist(non_ri_probs, bins=30, alpha=0.6, color="#4CAF50",
                label=f"Non-RI (n={len(non_ri_probs):,})", density=True)
        ax.hist(ri_probs, bins=30, alpha=0.7, color="#F44336",
                label=f"RI (n={len(ri_probs):,})", density=True)
    ax.set_xlabel("Predicted RI Probability (XGBoost)")
    ax.set_ylabel("Density")
    ax.set_title("RI Probability Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Case-Study Timeline (if available) ──
    ax = axes[1, 1]
    PROCESSED_DIR_local = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    holdout_path = os.path.join(PROCESSED_DIR_local, "holdout_storms.csv")

    # Try to load holdout data for timeline
    holdout_csv = os.path.join(PROCESSED_DIR_local, "holdout_storms.csv")
    if os.path.exists(holdout_csv):
        hdf = pd.read_csv(holdout_csv)
        hdf["ISO_TIME"] = pd.to_datetime(hdf["ISO_TIME"])
        if "VMAX" in hdf.columns and len(hdf) > 0:
            ax.plot(hdf["ISO_TIME"], hdf["VMAX"], "b-o", markersize=3,
                    linewidth=1.5, label="Max Wind (kt)")
            ax.set_ylabel("Max Wind (kt)")
            ax.set_title("Test Storm Track Intensity")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.legend()
    else:
        ax.text(0.5, 0.5, "Case study plot\n(run case_study.py first)",
                ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.set_title("Case Study Timeline")
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Confusion Matrix Heatmap ──
    ax = axes[1, 2]
    xgb_metrics = results.get("xgboost", {})
    if all(k in xgb_metrics for k in ["TP", "FP", "FN", "TN"]):
        cm = np.array([
            [xgb_metrics["TN"], xgb_metrics["FP"]],
            [xgb_metrics["FN"], xgb_metrics["TP"]],
        ])
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                    xticklabels=["Predicted Non-RI", "Predicted RI"],
                    yticklabels=["Actual Non-RI", "Actual RI"],
                    annot_kws={"size": 14})
        ax.set_title("Confusion Matrix (XGBoost)")
    else:
        ax.text(0.5, 0.5, "No confusion matrix data",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUTPUTS_DIR, "dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Dashboard saved to {out_path}")

    # Also print a text summary
    print(f"\n{'=' * 50}")
    print(f"  MODEL PERFORMANCE SUMMARY")
    print(f"{'=' * 50}")
    for model_name in ["random_forest", "xgboost"]:
        m = results.get(model_name, {})
        label = "Random Forest" if "forest" in model_name else "XGBoost"
        print(f"\n  {label}:")
        print(f"    AUC-ROC  : {m.get('AUC_ROC', 'N/A'):.4f}")
        print(f"    POD      : {m.get('POD_Recall', 'N/A'):.4f}")
        print(f"    FAR      : {m.get('FAR', 'N/A'):.4f}")
        print(f"    CSI      : {m.get('CSI', 'N/A'):.4f}")
        print(f"    Brier    : {m.get('Brier_Score', 'N/A'):.4f}")


if __name__ == "__main__":
    create_dashboard()
