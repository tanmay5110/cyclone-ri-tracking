"""
Process IBTrACS best-track CSV into a clean DataFrame with RI labels.

Key operations:
  1. Parse IBTrACS CSV (skip units row, handle multiple agency columns)
  2. Filter to storms ≥ Tropical Storm strength (≥ 34 kt)
  3. Compute RI label: Δ(Vmax) ≥ 30 kt in the next 24 h
  4. Compute derived features: translational speed, heading,
     Δ-pressure, Δ-wind over 6/12/24 h, lat, distance from land
"""

import os
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt, atan2, degrees

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y)) % 360


def load_ibtracs(csv_path: str = None) -> pd.DataFrame:
    """Load and clean IBTrACS CSV."""
    if csv_path is None:
        csv_path = os.path.join(RAW_DIR, "ibtracs.ALL.list.v04r01.csv")

    print(f"[LOAD] Reading IBTrACS from {csv_path} ...")

    # IBTrACS has two header rows: column names (row 0) + units (row 1)
    df = pd.read_csv(csv_path, skiprows=[1], low_memory=False,
                     na_values=[" ", "", "MM"])

    # Convert key columns to numeric
    for col in ["LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamp
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")

    # Use USA_WIND if available, else WMO_WIND
    df["VMAX"] = df["USA_WIND"].fillna(df["WMO_WIND"])
    df["PMIN"] = df["USA_PRES"].fillna(df["WMO_PRES"])

    # Convert lat/lon to float
    df["LAT"] = df["LAT"].astype(float)
    df["LON"] = df["LON"].astype(float)

    # Drop rows without essential data
    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON", "VMAX"])

    print(f"  → {len(df):,} rows, {df['SID'].nunique():,} storms after cleaning")
    return df


def filter_storms(df: pd.DataFrame,
                  min_wind: float = 34.0,
                  year_range: tuple = (2000, 2023),
                  basins: list = None) -> pd.DataFrame:
    """Filter to relevant storms for RI modelling."""
    # Filter by year
    df = df[df["ISO_TIME"].dt.year.between(year_range[0], year_range[1])].copy()

    # Filter to fixes where storm is at least TS strength
    df = df[df["VMAX"] >= min_wind].copy()

    # Optionally filter by basin
    if basins:
        df = df[df["BASIN"].isin(basins)].copy()

    # Sort by storm and time
    df = df.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    print(f"  → {len(df):,} rows, {df['SID'].nunique():,} storms "
          f"({year_range[0]}-{year_range[1]}, VMAX ≥ {min_wind} kt)")
    return df


def compute_ri_labels(df: pd.DataFrame, threshold_kt: float = 30.0,
                      window_hours: int = 24) -> pd.DataFrame:
    """
    Compute RI label: 1 if the storm's max wind increases by ≥ threshold_kt
    within the next window_hours.

    RI definition (Kaplan & DeMaria 2003): ≥ 30 kt increase in 24 h.
    """
    n_steps = window_hours // 6  # IBTrACS is 6-hourly → 4 steps for 24 h

    df = df.copy()
    df["RI"] = 0

    for sid, grp in df.groupby("SID"):
        idx = grp.index
        vmax = grp["VMAX"].values
        for i in range(len(vmax) - n_steps):
            # Future max wind within window
            future_max = np.nanmax(vmax[i + 1: i + n_steps + 1])
            delta_v = future_max - vmax[i]
            if delta_v >= threshold_kt:
                df.loc[idx[i], "RI"] = 1

    n_ri = df["RI"].sum()
    n_total = len(df)
    print(f"  → RI labels: {n_ri:,} positive ({100 * n_ri / n_total:.1f}%) "
          f"out of {n_total:,} fixes")
    return df


def compute_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features from the storm track."""
    df = df.copy()

    # Per-storm feature computation
    new_cols = {
        "DELTA_V_6H": [],     # Vmax change over PREVIOUS 6 h
        "DELTA_V_12H": [],    # Vmax change over PREVIOUS 12 h
        "DELTA_P_6H": [],     # Pressure change over PREVIOUS 6 h
        "TRANS_SPEED": [],    # Translational speed (km/h)
        "HEADING": [],        # Storm heading (degrees)
        "ABS_LAT": [],        # Absolute latitude (Coriolis proxy)
    }

    result_frames = []

    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").copy()
        v = grp["VMAX"].values
        p = grp["PMIN"].values
        lat = grp["LAT"].values
        lon = grp["LON"].values

        n = len(grp)

        # Δ-wind (previous)
        dv6 = np.full(n, np.nan)
        dv12 = np.full(n, np.nan)
        dp6 = np.full(n, np.nan)
        tspeed = np.full(n, np.nan)
        hdg = np.full(n, np.nan)

        for i in range(1, n):
            dv6[i] = v[i] - v[i - 1]
            if i >= 2:
                dv12[i] = v[i] - v[i - 2]
            dp6[i] = p[i] - p[i - 1] if not (np.isnan(p[i]) or np.isnan(p[i - 1])) else np.nan

            # Translation speed
            dt_hours = (grp["ISO_TIME"].iloc[i] - grp["ISO_TIME"].iloc[i - 1]).total_seconds() / 3600
            if dt_hours > 0:
                dist = haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])
                tspeed[i] = dist / dt_hours
                hdg[i] = bearing(lat[i - 1], lon[i - 1], lat[i], lon[i])

        grp["DELTA_V_6H"] = dv6
        grp["DELTA_V_12H"] = dv12
        grp["DELTA_P_6H"] = dp6
        grp["TRANS_SPEED"] = tspeed
        grp["HEADING"] = hdg
        grp["ABS_LAT"] = np.abs(lat)

        result_frames.append(grp)

    result = pd.concat(result_frames, ignore_index=True)
    print(f"  → Computed track features: DELTA_V_6H, DELTA_V_12H, DELTA_P_6H, "
          f"TRANS_SPEED, HEADING, ABS_LAT")
    return result


def process_ibtracs(csv_path: str = None,
                    year_range: tuple = (2000, 2023),
                    basins: list = None,
                    save: bool = True) -> pd.DataFrame:
    """Full IBTrACS processing pipeline."""
    df = load_ibtracs(csv_path)
    df = filter_storms(df, min_wind=34.0, year_range=year_range, basins=basins)
    df = compute_ri_labels(df)
    df = compute_track_features(df)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "ibtracs_processed.csv")
        df.to_csv(out_path, index=False)
        print(f"  → Saved to {out_path}")

    return df


if __name__ == "__main__":
    df = process_ibtracs()
    print(f"\n=== IBTrACS Processing Summary ===")
    print(f"  Shape          : {df.shape}")
    print(f"  Storms         : {df['SID'].nunique()}")
    print(f"  RI positive    : {df['RI'].sum()} ({100 * df['RI'].mean():.1f}%)")
    print(f"  Date range     : {df['ISO_TIME'].min()} → {df['ISO_TIME'].max()}")
    print(f"  Columns        : {list(df.columns)}")
