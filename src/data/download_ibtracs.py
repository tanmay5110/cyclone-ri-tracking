"""
Download IBTrACS v04r01 best-track data (ALL basins).

Source: NOAA National Centers for Environmental Information (NCEI)
URL: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/

This dataset contains every tropical cyclone globally since 1842 with
6-hourly position, maximum wind speed (kt), minimum pressure (hPa),
storm category, and agency-specific estimates.
"""

import os
import sys
import requests
from tqdm import tqdm

# IBTrACS v04r01 — full global dataset (CSV)
IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> str:
    """Stream-download a large file with a progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[SKIP] {dest_path} already exists ({size_mb:.1f} MB)")
        return dest_path

    print(f"[DOWNLOAD] {url}")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    print(f"[OK] Saved {dest_path} ({size_mb:.1f} MB)")
    return dest_path


def download_ibtracs():
    """Download the full IBTrACS global CSV."""
    dest = os.path.join(RAW_DIR, "ibtracs.ALL.list.v04r01.csv")
    download_file(IBTRACS_URL, dest)
    return dest


def verify_ibtracs(path: str):
    """Quick sanity check on the downloaded file."""
    import pandas as pd

    # IBTrACS CSV has two header rows (column names + units)
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    n_storms = df["SID"].nunique()
    n_rows = len(df)
    basins = df["BASIN"].unique().tolist()

    print(f"\n=== IBTrACS Verification ===")
    print(f"  Rows        : {n_rows:,}")
    print(f"  Unique SIDs : {n_storms:,}")
    print(f"  Basins      : {basins}")
    print(f"  Date range  : {df['ISO_TIME'].min()} → {df['ISO_TIME'].max()}")
    print(f"  Columns     : {len(df.columns)}")

    assert n_rows > 100_000, f"Expected >100k rows, got {n_rows}"
    assert n_storms > 5_000, f"Expected >5k storms, got {n_storms}"
    print("  ✓ Verification passed\n")
    return df


if __name__ == "__main__":
    path = download_ibtracs()
    verify_ibtracs(path)
