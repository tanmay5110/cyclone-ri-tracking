"""
Case study evaluation: Run the trained RI model on a held-out cyclone
and produce a time-series comparison of predictions vs actual intensity.

Default test case: Hurricane Patricia (2015, EP) — one of the most 
extreme RI events on record (100 kt intensification in 24 hours).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_model_and_scaler():
    """Load the best trained model and scaler."""
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
    rf_path = os.path.join(MODELS_DIR, "rf_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    features_path = os.path.join(MODELS_DIR, "feature_names.joblib")

    model = joblib.load(xgb_path) if os.path.exists(xgb_path) else joblib.load(rf_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_names = joblib.load(features_path) if os.path.exists(features_path) else None

    # Determine if the model needs scaled features
    model_name = type(model).__name__
    needs_scaling = "Forest" in model_name or "Logistic" in model_name

    return model, scaler, feature_names, needs_scaling, model_name


def load_holdout_data(storm_sid: str = None) -> pd.DataFrame:
    """Load holdout storm data."""
    holdout_path = os.path.join(PROCESSED_DIR, "holdout_storms.csv")

    if os.path.exists(holdout_path):
        df = pd.read_csv(holdout_path)
    else:
        # If no holdout file, try to find the storm in features.csv
        features_path = os.path.join(PROCESSED_DIR, "features.csv")
        df = pd.read_csv(features_path)

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    if storm_sid:
        df = df[df["SID"].str.contains(storm_sid, case=False, na=False)].copy()

    if len(df) == 0:
        # If specific SID not found, try pattern matching for Patricia
        patricia_patterns = ["EP202015", "PATRICIA", "2015"]
        for pat in patricia_patterns:
            matches = df[df["SID"].str.contains(pat, case=False, na=False)]
            if len(matches) > 0:
                df = matches.copy()
                break

    df = df.sort_values("ISO_TIME").reset_index(drop=True)
    return df


def run_case_study(storm_sid: str = None, storm_name: str = "Hurricane Patricia"):
    """Run case study prediction on a single cyclone."""
    print(f"\n{'=' * 60}")
    print(f"  CASE STUDY: {storm_name}")
    print(f"{'=' * 60}")

    # Load model
    model, scaler, feature_names, needs_scaling, model_name = load_model_and_scaler()
    print(f"  Model: {model_name}")
    print(f"  Features: {len(feature_names)}")

    # Load storm data
    storm_df = load_holdout_data(storm_sid)
    if len(storm_df) == 0:
        print(f"  [ERROR] No data found for storm '{storm_sid}'. "
              f"Available SIDs in holdout:")
        holdout_path = os.path.join(PROCESSED_DIR, "holdout_storms.csv")
        if os.path.exists(holdout_path):
            hdf = pd.read_csv(holdout_path)
            print(f"    {hdf['SID'].unique().tolist()}")
        return None

    print(f"  Storm fixes: {len(storm_df)}")
    print(f"  Time range: {storm_df['ISO_TIME'].min()} → {storm_df['ISO_TIME'].max()}")

    # Prepare features
    available = [f for f in feature_names if f in storm_df.columns]
    missing = [f for f in feature_names if f not in storm_df.columns]
    if missing:
        print(f"  [WARN] Missing features (will be filled with 0): {missing}")
        for m in missing:
            storm_df[m] = 0.0

    X = storm_df[feature_names].values.astype(np.float32)

    # Fill NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    if needs_scaling and scaler is not None:
        X = scaler.transform(X)

    # Predict RI probability at each fix
    ri_probs = model.predict_proba(X)[:, 1]
    ri_preds = model.predict(X)

    storm_df["RI_PROB"] = ri_probs
    storm_df["RI_PRED"] = ri_preds

    # Print results
    print(f"\n  --- Prediction Results ---")
    for _, row in storm_df.iterrows():
        ri_actual = "RI" if row.get("RI", 0) == 1 else "  "
        ri_pred = "→RI" if row["RI_PRED"] == 1 else "   "
        print(f"  {row['ISO_TIME']} | Vmax={row.get('VMAX', '?'):>5} kt | "
              f"P(RI)={row['RI_PROB']:.3f} {ri_pred} | Actual: {ri_actual}")

    # Evaluate
    if "RI" in storm_df.columns:
        y_true = storm_df["RI"].values
        if y_true.sum() > 0:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, ri_probs)
            print(f"\n  Case Study AUC-ROC: {auc:.4f}")

            hits = ((ri_preds == 1) & (y_true == 1)).sum()
            total_ri = y_true.sum()
            print(f"  RI events detected: {hits}/{total_ri} "
                  f"({100 * hits / total_ri:.0f}% POD)")

    # Plot
    plot_case_study(storm_df, storm_name)

    return storm_df


def plot_case_study(storm_df: pd.DataFrame, storm_name: str):
    """Create the case-study time-series overlay plot."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [2, 1]})

    times = storm_df["ISO_TIME"]

    # --- Top panel: Intensity ---
    if "VMAX" in storm_df.columns:
        ax1.plot(times, storm_df["VMAX"], "b-o", linewidth=2, markersize=4,
                 label="Max Wind (kt)", zorder=3)
        ax1.set_ylabel("Max Sustained Wind (kt)", fontsize=12, color="blue")

        # Color-code RI periods
        if "RI" in storm_df.columns:
            ri_mask = storm_df["RI"] == 1
            ax1.fill_between(times, 0, storm_df["VMAX"].max() * 1.1,
                           where=ri_mask, alpha=0.15, color="red",
                           label="Actual RI period", zorder=1)

    # Add pressure on secondary axis
    if "PMIN" in storm_df.columns:
        ax1b = ax1.twinx()
        ax1b.plot(times, storm_df["PMIN"], "r--", linewidth=1.5, alpha=0.7,
                  label="Min Pressure (hPa)")
        ax1b.set_ylabel("Min Pressure (hPa)", fontsize=12, color="red")
        ax1b.invert_yaxis()

    ax1.set_title(f"{storm_name} — RI Prediction Case Study", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # --- Bottom panel: RI probability ---
    ax2.fill_between(times, 0, storm_df["RI_PROB"], alpha=0.3, color="orange")
    ax2.plot(times, storm_df["RI_PROB"], "o-", color="darkorange", linewidth=2,
             markersize=5, label="Model P(RI)")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")

    # Mark actual RI events
    if "RI" in storm_df.columns:
        ri_times = times[storm_df["RI"] == 1]
        ri_probs = storm_df.loc[storm_df["RI"] == 1, "RI_PROB"]
        ax2.scatter(ri_times, ri_probs, color="red", s=80, zorder=5,
                   marker="*", label="Actual RI fix")

    ax2.set_ylabel("RI Probability", fontsize=12)
    ax2.set_xlabel("Time (UTC)", fontsize=12)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "patricia_case_study.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Case study plot saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", type=str, default=None,
                        help="Storm SID to evaluate")
    parser.add_argument("--name", type=str, default="Hurricane Patricia",
                        help="Storm display name")
    args = parser.parse_args()

    run_case_study(args.sid, args.name)
