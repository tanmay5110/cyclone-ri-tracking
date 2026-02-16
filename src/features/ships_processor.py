"""
Parse SHIPS developmental predictor files into a tabular DataFrame.

SHIPS predictor files are fixed-width text files with a specific format:
  - Storm header lines: "ST01L  2005  AL012005  ..."
  - Followed by predictor data blocks per forecast hour
  
Key predictors extracted:
  - SST     : Sea surface temperature (°C × 10)
  - SHRD    : 200-850 hPa shear magnitude (kt × 10)
  - D200    : 200 hPa divergence (× 10^7 s^-1)
  - RHMD    : Mid-level relative humidity (%)
  - VMAX    : Initial storm intensity (kt)
  - VMPI    : Maximum potential intensity (kt)
  - RSST    : Reynolds SST (°C × 10)
  - U200    : 200 hPa zonal wind (kt × 10)
  - TWAC    : Depth-averaged warm core temperature (°C × 10)
  - OHC     : Ocean heat content (KJ/cm² × 10)
"""

import os
import re
import numpy as np
import pandas as pd

RAW_SHIPS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ships")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Predictor names in the SHIPS LSDIAG files (order matters)
# These are the most common predictors in the 7-day files.
# The exact ordering may vary by basin/year; we parse by label.
SHIPS_PREDICTOR_NAMES = [
    "VMAX", "MSLP", "TYPE", "DELV", "INCV",
    "LAT", "LON", "SPD", "HDG",
    "SST", "RSST", "DSST", "DSTA", "OHC", "RHLO", "RHMD", "RHHI",
    "PSLV", "D200", "REFC", "PEFC", "T200", "T250", "THETA_E",
    "SHRD", "SHRS", "SHTD", "SHTS", "U200", "U20C", "V20C",
    "E000", "EPOS", "ENEG", "EPSS", "EMPI", "VMPI",
    "VVAV", "VVAC", "TGRD", "TADV", "PENC", "TWAC", "TWXC",
    "G150", "G200", "G250",
    "NTMX", "NDTX", "NDFX",
    "DTL",
]


def parse_ships_file(filepath: str) -> pd.DataFrame:
    """
    Parse a SHIPS LSDIAG predictor file.
    
    Returns a DataFrame with one row per storm/forecast-time combination.
    Only the analysis time (t=0) row is kept for RI modelling.
    """
    records = []
    
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()
    
    i = 0
    n_storms = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for storm header lines (e.g., "AL012005" or "EP202015")
        header_match = re.match(
            r"^([A-Z]{2}\d{6})\s+(\d{10})\s+(\d+)\s+(.*)$", line
        )
        
        if header_match is None:
            # Try alternative header format
            header_match = re.match(
                r"^(ST\d{2}[A-Z])\s+(\d{4})\s+([A-Z]{2}\d{6})\s+(\d{10})", line
            )
            if header_match:
                storm_id = header_match.group(3)
                timestamp_str = header_match.group(4)
            else:
                i += 1
                continue
        else:
            storm_id = header_match.group(1)
            timestamp_str = header_match.group(2)
        
        n_storms += 1
        
        # Parse timestamp
        try:
            dt = pd.Timestamp(
                year=int(timestamp_str[:4]),
                month=int(timestamp_str[4:6]),
                day=int(timestamp_str[6:8]),
                hour=int(timestamp_str[8:10])
            )
        except (ValueError, IndexError):
            i += 1
            continue
        
        # Read predictor data lines that follow the header
        i += 1
        predictor_data = {}
        predictor_data["STORM_ID"] = storm_id
        predictor_data["ISO_TIME"] = dt
        
        # Read the next several lines for predictor blocks
        # SHIPS format: "PREDICTOR_NAME" followed by values for each forecast hour
        # We only want the t=0 (analysis) value
        while i < len(lines) and lines[i].strip():
            pline = lines[i].strip()
            
            # Check if this is a new storm header
            if re.match(r"^([A-Z]{2}\d{6}|ST\d{2}[A-Z])", pline):
                break
            
            # Try to parse as "LABEL  val0  val1  val2  ..."
            parts = pline.split()
            if len(parts) >= 2 and parts[0].isalpha():
                label = parts[0]
                try:
                    # Take the first value (t=0 analysis)
                    val = float(parts[1])
                    predictor_data[label] = val
                except ValueError:
                    pass
            elif len(parts) >= 2:
                # Lines with just numbers — skip
                pass
            
            i += 1
        
        records.append(predictor_data)
    
    df = pd.DataFrame(records)
    print(f"  → Parsed {filepath}: {n_storms} storm headers, {len(df)} records")
    return df


def load_all_ships(ships_dir: str = None) -> pd.DataFrame:
    """Load and concatenate all SHIPS predictor files."""
    if ships_dir is None:
        ships_dir = RAW_SHIPS_DIR
    
    if not os.path.exists(ships_dir):
        print(f"[WARN] SHIPS directory not found: {ships_dir}")
        return pd.DataFrame()
    
    dat_files = [f for f in os.listdir(ships_dir) 
                 if f.endswith(".dat") and "lsdiag" in f.lower()]
    
    if not dat_files:
        print(f"[WARN] No SHIPS .dat files found in {ships_dir}")
        return pd.DataFrame()
    
    dfs = []
    for f in sorted(dat_files):
        filepath = os.path.join(ships_dir, f)
        df = parse_ships_file(filepath)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Scale SHIPS predictors (many are stored × 10)
    scale_cols = {
        "SST": 10, "RSST": 10, "DSST": 10, "DSTA": 10,
        "OHC": 10, "SHRD": 10, "SHRS": 10, "SHTD": 10, "SHTS": 10,
        "U200": 10, "U20C": 10, "V20C": 10,
        "D200": 1, "TWAC": 10, "TWXC": 10,
    }
    
    for col, divisor in scale_cols.items():
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce") / divisor
    
    # Basic cleanup
    for col in combined.columns:
        if col not in ["STORM_ID", "ISO_TIME"]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    
    print(f"\n  → Combined SHIPS: {len(combined):,} records, "
          f"{combined['STORM_ID'].nunique()} storms, "
          f"{len(combined.columns)} columns")
    
    return combined


def process_ships(save: bool = True) -> pd.DataFrame:
    """Full SHIPS processing pipeline."""
    df = load_all_ships()
    
    if save and len(df) > 0:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "ships_processed.csv")
        df.to_csv(out_path, index=False)
        print(f"  → Saved to {out_path}")
    
    return df


if __name__ == "__main__":
    df = process_ships()
    if len(df) > 0:
        print(f"\n=== SHIPS Processing Summary ===")
        print(f"  Shape   : {df.shape}")
        print(f"  Storms  : {df['STORM_ID'].nunique()}")
        print(f"  Columns : {list(df.columns[:20])} ...")
        print(f"  Date range: {df['ISO_TIME'].min()} → {df['ISO_TIME'].max()}")
        
        # Show key predictor stats
        for col in ["SST", "SHRD", "OHC", "VMPI", "RHMD"]:
            if col in df.columns:
                print(f"  {col:6s} : mean={df[col].mean():.1f}, "
                      f"std={df[col].std():.1f}, "
                      f"missing={df[col].isna().sum()}")
    else:
        print("[INFO] No SHIPS data processed")
