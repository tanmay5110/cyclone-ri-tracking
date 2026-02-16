"""
Build the unified feature matrix by merging IBTrACS, SHIPS, and GridSat features.

This is the final data preparation step before model training.
Outputs: data/processed/features.csv
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.features.ibtracs_processor import process_ibtracs
from src.features.ships_processor import process_ships
from src.features.gridsat_processor import process_gridsat_for_fixes

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def build_dataset(
    year_range: tuple = (2000, 2023),
    include_gridsat: bool = False,
    hold_out_storms: list = None,
) -> pd.DataFrame:
    """
    Build the unified ML feature matrix.

    Steps:
      1. Process IBTrACS → track features + RI labels
      2. Process SHIPS → environmental predictors
      3. (Optional) Process GridSat → IR satellite features
      4. Merge on (storm_id, time) with fuzzy time matching
      5. Drop rows with too many NaN values
      6. Save to CSV

    Parameters:
        year_range: (start_year, end_year) inclusive
        include_gridsat: Whether to add satellite IR features (slower)
        hold_out_storms: List of SIDs to exclude (for case study)
    """
    print("=" * 60)
    print("BUILDING UNIFIED FEATURE MATRIX")
    print("=" * 60)

    # === Step 1: IBTrACS ===
    print("\n--- Step 1: Processing IBTrACS ---")
    ibtracs_df = process_ibtracs(year_range=year_range, save=False)

    # === Step 2: SHIPS ===
    print("\n--- Step 2: Processing SHIPS predictors ---")
    ships_df = process_ships(save=False)

    # === Step 3: Merge IBTrACS + SHIPS ===
    print("\n--- Step 3: Merging datasets ---")
    merged = merge_ibtracs_ships(ibtracs_df, ships_df)

    # === Step 4: GridSat IR features (optional) ===
    if include_gridsat:
        print("\n--- Step 4: Adding GridSat IR features ---")
        merged = process_gridsat_for_fixes(merged)
    else:
        print("\n--- Step 4: Skipping GridSat (use --gridsat to include) ---")

    # === Step 5: Hold out test storms ===
    if hold_out_storms:
        mask = merged["SID"].isin(hold_out_storms)
        held_out = merged[mask].copy()
        merged = merged[~mask].copy()
        print(f"\n  → Held out {len(held_out)} rows for test storms: {hold_out_storms}")

        # Save held-out data separately
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        held_out.to_csv(os.path.join(PROCESSED_DIR, "holdout_storms.csv"), index=False)

    # === Step 6: Clean & save ===
    print("\n--- Step 5: Cleaning and saving ---")
    merged = clean_features(merged)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "features.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n  → Saved feature matrix to {out_path}")

    return merged


def merge_ibtracs_ships(ibtracs_df: pd.DataFrame,
                        ships_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge IBTrACS and SHIPS data.

    SHIPS uses a different storm ID scheme (e.g., "AL012005")
    while IBTrACS uses SID (e.g., "2005213N23285").

    We merge on (approximate basin+year) + time, matching within 3 hours.
    If SHIPS data is unavailable, we proceed with IBTrACS features only.
    """
    if ships_df is None or len(ships_df) == 0:
        print("  → No SHIPS data available; proceeding with IBTrACS features only")
        return ibtracs_df

    # Ensure timestamps are datetime
    ibtracs_df["ISO_TIME"] = pd.to_datetime(ibtracs_df["ISO_TIME"])
    ships_df["ISO_TIME"] = pd.to_datetime(ships_df["ISO_TIME"])

    # Try matching by timestamp (within 3h tolerance)
    # We use merge_asof for fuzzy time matching within each group
    ibtracs_sorted = ibtracs_df.sort_values("ISO_TIME").copy()
    ships_sorted = ships_df.sort_values("ISO_TIME").copy()

    # Add a merge key from SHIPS (first 2 chars of STORM_ID = basin)
    if "STORM_ID" in ships_sorted.columns:
        ships_sorted["SHIPS_BASIN"] = ships_sorted["STORM_ID"].str[:2]
        ships_sorted["SHIPS_YEAR"] = ships_sorted["ISO_TIME"].dt.year

    # Simple time-based merge with 3-hour tolerance
    merged = pd.merge_asof(
        ibtracs_sorted,
        ships_sorted.drop(columns=["STORM_ID"], errors="ignore"),
        on="ISO_TIME",
        tolerance=pd.Timedelta("3h"),
        direction="nearest",
    )

    n_matched = merged.drop(columns=ibtracs_df.columns, errors="ignore").notna().any(axis=1).sum()
    print(f"  → Merged: {len(merged)} rows, {n_matched} with SHIPS matches "
          f"({100 * n_matched / len(merged):.0f}%)")

    return merged


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with excessive NaN and select model features."""
    # Define the feature columns we want for the model
    track_features = [
        "VMAX", "PMIN", "LAT", "LON",
        "DELTA_V_6H", "DELTA_V_12H", "DELTA_P_6H",
        "TRANS_SPEED", "HEADING", "ABS_LAT",
    ]

    ships_features = [
        "SST", "RSST", "SHRD", "SHRS", "D200", "OHC",
        "RHLO", "RHMD", "RHHI", "VMPI", "U200",
        "TWAC", "PSLV", "REFC", "PEFC",
    ]

    ir_features = [
        "IR_MEAN_BT", "IR_MIN_BT", "IR_STD_BT",
        "IR_FRAC_LT200K", "IR_FRAC_LT220K", "IR_AXISYM",
    ]

    # Metadata columns to keep
    meta_cols = ["SID", "ISO_TIME", "BASIN", "RI"]

    # Build list of available features
    all_features = []
    for feat_list in [track_features, ships_features, ir_features]:
        for f in feat_list:
            if f in df.columns:
                all_features.append(f)

    cols_to_keep = [c for c in meta_cols if c in df.columns] + all_features
    df = df[cols_to_keep].copy()

    # Drop rows where ALL feature columns are NaN
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df.dropna(subset=feature_cols, how="all")

    # Forward-fill within each storm track for small gaps
    for sid, grp in df.groupby("SID"):
        df.loc[grp.index] = grp.ffill(limit=2)

    # Drop rows that still have > 50% NaN in features
    thresh = len(feature_cols) // 2
    df = df.dropna(subset=feature_cols, thresh=thresh)

    print(f"  → After cleaning: {len(df):,} rows, {df['SID'].nunique()} storms, "
          f"{len(feature_cols)} features")
    print(f"  → RI distribution: {df['RI'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RI feature matrix")
    parser.add_argument("--gridsat", action="store_true",
                        help="Include GridSat IR features (slower)")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--holdout", nargs="*", default=None,
                        help="Storm SIDs to hold out for testing")
    args = parser.parse_args()

    df = build_dataset(
        year_range=(args.start_year, args.end_year),
        include_gridsat=args.gridsat,
        hold_out_storms=args.holdout,
    )

    print(f"\n=== Feature Matrix Summary ===")
    print(f"  Shape          : {df.shape}")
    print(f"  Storms         : {df['SID'].nunique()}")
    print(f"  RI positive    : {df['RI'].sum()} ({100 * df['RI'].mean():.1f}%)")
    print(f"  Feature columns: {[c for c in df.columns if c not in ['SID','ISO_TIME','BASIN','RI']]}")
    print(f"  NaN summary    :")
    for col in df.columns:
        na = df[col].isna().sum()
        if na > 0:
            print(f"    {col:20s}: {na:6d} ({100 * na / len(df):.1f}%)")
