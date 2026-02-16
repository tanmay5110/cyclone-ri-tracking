"""
Download SHIPS developmental predictor files from RAMMB / Colorado State.

Source: https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/
       → Developmental Data sub-page

The SHIPS predictor files contain pre-computed environmental variables
for each tropical cyclone fix, including:
  - Sea surface temperature (SST)
  - 200-850 hPa vertical wind shear (magnitude and direction)
  - 200 hPa divergence
  - Mid-level relative humidity
  - Ocean heat content (OHC)
  - Maximum potential intensity (MPI)
  - Initial storm intensity
  - And many more (~100+ predictors)

These are the same predictors used by the operational SHIPS/RII models.
"""

import os
import re
import requests
from tqdm import tqdm

# SHIPS predictor file URLs (developmental data)
# We download the Atlantic (AL) and East Pacific (EP) 7-day predictor files
# which include Patricia (EP) and many Atlantic RI cases.
SHIPS_BASE_URL = "https://rammb2.cira.colostate.edu/wp-content/uploads/2021/08/"

SHIPS_FILES = {
    # Atlantic 7-day predictors (1982-2023)
    "lsdiaga_1982_2023_sat_ts.dat": "https://rammb-data.cira.colostate.edu/tc_realtime/products/ships/developmental_data/lsdiaga_1982_2023_sat_ts.dat",
    # East Pacific 7-day predictors (Patricia is EP202015)
    "lsdiage_1982_2023_sat_ts.dat": "https://rammb-data.cira.colostate.edu/tc_realtime/products/ships/developmental_data/lsdiage_1982_2023_sat_ts.dat",
}

# Fallback URLs
SHIPS_FALLBACK_URLS = {
    "lsdiaga_1982_2023_sat_ts.dat": [
        "https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/developmental_data/lsdiaga_1982_2023_sat_ts.dat",
    ],
    "lsdiage_1982_2023_sat_ts.dat": [
        "https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/developmental_data/lsdiage_1982_2023_sat_ts.dat",
    ],
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "ships")


def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> bool:
    """Download a file, returning True on success."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"[SKIP] {dest_path} already exists ({size_mb:.1f} MB)")
        return True

    print(f"[DOWNLOAD] {url}")
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Failed: {e}")
        return False

    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    size_kb = os.path.getsize(dest_path) / 1024
    print(f"[OK] Saved {dest_path} ({size_kb:.0f} KB)")
    return True


def download_ships():
    """Download SHIPS predictor files, trying primary and fallback URLs."""
    downloaded = []
    for filename, primary_url in SHIPS_FILES.items():
        dest = os.path.join(RAW_DIR, filename)

        # Try primary URL
        if download_file(primary_url, dest):
            downloaded.append(dest)
            continue

        # Try fallback URLs
        fallbacks = SHIPS_FALLBACK_URLS.get(filename, [])
        success = False
        for fallback_url in fallbacks:
            fallback_name = os.path.basename(fallback_url)
            fallback_dest = os.path.join(RAW_DIR, fallback_name)
            if download_file(fallback_url, fallback_dest):
                downloaded.append(fallback_dest)
                success = True
                break

        if not success:
            print(f"[ERROR] Could not download {filename} from any URL")

    return downloaded


def verify_ships(files: list):
    """Basic verification of downloaded SHIPS files."""
    print(f"\n=== SHIPS Verification ===")
    for f in files:
        size_kb = os.path.getsize(f) / 1024
        # Count the number of storm header lines
        with open(f, "r", errors="replace") as fh:
            lines = fh.readlines()
        n_lines = len(lines)
        # SHIPS headers start with storm ID pattern like "AL012005"
        headers = [l for l in lines if re.match(r"^[A-Z]{2}\d{6}", l.strip())]
        print(f"  {os.path.basename(f)}: {size_kb:.0f} KB, {n_lines:,} lines, "
              f"{len(headers):,} storm headers")
    print(f"  ✓ {len(files)} SHIPS file(s) downloaded\n")


if __name__ == "__main__":
    files = download_ships()
    if files:
        verify_ships(files)
    else:
        print("[ERROR] No SHIPS files downloaded. Check URLs or network.")
