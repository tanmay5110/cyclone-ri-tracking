"""
Process GridSat-B1 satellite IR imagery to extract brightness temperature
features around each tropical cyclone fix.

For each storm fix (lat, lon, time), we:
  1. Find the nearest GridSat-B1 file (3-hourly)
  2. Extract a 5° × 5° patch centered on the storm
  3. Compute IR statistics:
     - Mean brightness temperature (BT)
     - Min BT (proxy for cloud-top height / deep convection)
     - Std dev of BT (convective organization)
     - Fraction of pixels < 200 K (deep convective coverage)
     - Fraction of pixels < 220 K (cold cloud coverage)
     - Axisymmetry index (std of azimuthally-averaged BT)
"""

import os
import glob
import numpy as np
import pandas as pd

RAW_GRIDSAT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "gridsat")

# Try to import xarray — it's needed for NetCDF reading
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("[WARN] xarray not installed; GridSat processing will be skipped")


def find_gridsat_file(dt: pd.Timestamp, search_dirs: list = None) -> str | None:
    """Find the GridSat-B1 file nearest to the given datetime."""
    if search_dirs is None:
        # Search all sub-directories under the gridsat raw dir
        search_dirs = []
        if os.path.exists(RAW_GRIDSAT_DIR):
            for d in os.listdir(RAW_GRIDSAT_DIR):
                full = os.path.join(RAW_GRIDSAT_DIR, d)
                if os.path.isdir(full):
                    search_dirs.append(full)
            search_dirs.append(RAW_GRIDSAT_DIR)

    # GridSat files are 3-hourly; round to nearest 3 h
    hour = (dt.hour // 3) * 3
    target_name = f"GRIDSAT-B1.{dt.year}.{dt.month:02d}.{dt.day:02d}.{hour:02d}.v02r01.nc"

    for d in search_dirs:
        candidate = os.path.join(d, target_name)
        if os.path.exists(candidate):
            return candidate

    return None


def extract_ir_patch(nc_path: str, center_lat: float, center_lon: float,
                     half_size: float = 2.5) -> np.ndarray | None:
    """
    Extract a lat/lon patch of IR brightness temperature from a GridSat file.

    Parameters:
        nc_path: Path to the GridSat-B1 NetCDF file
        center_lat: Storm center latitude
        center_lon: Storm center longitude (will be converted to 0-360)
        half_size: Half-width of the patch in degrees (default 2.5° → 5° box)

    Returns:
        2D array of brightness temperatures, or None if extraction fails.
    """
    if not HAS_XARRAY:
        return None

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        print(f"[WARN] Cannot open {nc_path}: {e}")
        return None

    # GridSat-B1 uses 'irwin_cdr' variable for the calibrated IR channel
    var_name = None
    for candidate in ["irwin_cdr", "irwin", "irwvp"]:
        if candidate in ds.data_vars:
            var_name = candidate
            break

    if var_name is None:
        ds.close()
        return None

    ir = ds[var_name]

    # GridSat longitude is 0–360
    glon = center_lon if center_lon >= 0 else center_lon + 360

    try:
        patch = ir.sel(
            lat=slice(center_lat + half_size, center_lat - half_size),
            lon=slice(glon - half_size, glon + half_size)
        )
        if patch.ndim == 3:
            patch = patch.isel(time=0)
        data = patch.values.astype(float)
    except Exception:
        ds.close()
        return None

    ds.close()

    # Replace fill values with NaN
    data[data < 100] = np.nan
    data[data > 350] = np.nan

    return data


def compute_ir_features(patch: np.ndarray) -> dict:
    """Compute statistical features from an IR brightness temperature patch."""
    if patch is None or patch.size == 0 or np.all(np.isnan(patch)):
        return {
            "IR_MEAN_BT": np.nan,
            "IR_MIN_BT": np.nan,
            "IR_STD_BT": np.nan,
            "IR_FRAC_LT200K": np.nan,
            "IR_FRAC_LT220K": np.nan,
            "IR_AXISYM": np.nan,
        }

    valid = patch[~np.isnan(patch)]

    return {
        "IR_MEAN_BT": float(np.nanmean(valid)),
        "IR_MIN_BT": float(np.nanmin(valid)),
        "IR_STD_BT": float(np.nanstd(valid)),
        "IR_FRAC_LT200K": float(np.sum(valid < 200) / len(valid)),
        "IR_FRAC_LT220K": float(np.sum(valid < 220) / len(valid)),
        "IR_AXISYM": _axisymmetry(patch),
    }


def _axisymmetry(patch: np.ndarray) -> float:
    """
    Simple axisymmetry index: standard deviation of row-wise means.
    Lower values → more symmetric cloud pattern.
    """
    if patch is None or patch.size == 0:
        return np.nan

    row_means = np.nanmean(patch, axis=1)
    if np.all(np.isnan(row_means)):
        return np.nan
    return float(np.nanstd(row_means))


def process_gridsat_for_fixes(fixes_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each storm fix in fixes_df (must have LAT, LON, ISO_TIME columns),
    extract IR features from the matching GridSat file.

    Returns the input DataFrame with IR feature columns appended.
    """
    if not HAS_XARRAY:
        print("[WARN] xarray not available — returning empty IR features")
        for col in ["IR_MEAN_BT", "IR_MIN_BT", "IR_STD_BT",
                     "IR_FRAC_LT200K", "IR_FRAC_LT220K", "IR_AXISYM"]:
            fixes_df[col] = np.nan
        return fixes_df

    ir_features = []
    n_found = 0

    for _, row in fixes_df.iterrows():
        dt = pd.Timestamp(row["ISO_TIME"])
        nc_path = find_gridsat_file(dt)

        if nc_path is not None:
            patch = extract_ir_patch(nc_path, row["LAT"], row["LON"])
            feats = compute_ir_features(patch)
            n_found += 1
        else:
            feats = compute_ir_features(None)

        ir_features.append(feats)

    ir_df = pd.DataFrame(ir_features, index=fixes_df.index)
    result = pd.concat([fixes_df, ir_df], axis=1)

    print(f"  → GridSat IR features: matched {n_found}/{len(fixes_df)} fixes")
    return result


if __name__ == "__main__":
    # Quick test: process a small sample
    processed_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "ibtracs_processed.csv"
    )
    if os.path.exists(processed_path):
        df = pd.read_csv(processed_path, nrows=20)
        df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"])
        result = process_gridsat_for_fixes(df)
        print(result[["SID", "ISO_TIME", "VMAX", "IR_MEAN_BT", "IR_MIN_BT"]].head(10))
    else:
        print(f"[INFO] Run ibtracs_processor.py first to generate {processed_path}")
