"""
Download GridSat-B1 satellite infrared brightness temperature data.

Source: NOAA NCEI Climate Data Record
URL: https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/

GridSat-B1 provides:
  - Global geostationary satellite IR (~11 µm) brightness temperatures
  - 0.07° spatial resolution
  - 3-hourly temporal resolution
  - NetCDF4 format
  - Data from 1980 to present

For the RI tracking model, we download GridSat files matching the
date ranges of storms in our training set. This script downloads
files for a configurable list of storm date ranges.
"""

import os
import sys
from datetime import datetime, timedelta
import requests
from tqdm import tqdm


GRIDSAT_BASE_URL = (
    "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/"
    "access/{year}/GRIDSAT-B1.{year}.{month}.{day}.{hour}.v02r01.nc"
)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "gridsat")


def gridsat_url(dt: datetime) -> str:
    """Build the URL for a specific GridSat-B1 file (3-hourly)."""
    # GridSat files are at 00, 03, 06, 09, 12, 15, 18, 21 UTC
    hour = (dt.hour // 3) * 3
    return GRIDSAT_BASE_URL.format(
        year=dt.strftime("%Y"),
        month=dt.strftime("%m"),
        day=dt.strftime("%d"),
        hour=f"{hour:02d}",
    )


def gridsat_filename(dt: datetime) -> str:
    """Local filename for a GridSat-B1 file."""
    hour = (dt.hour // 3) * 3
    return f"GRIDSAT-B1.{dt.strftime('%Y')}.{dt.strftime('%m')}.{dt.strftime('%d')}.{hour:02d}.v02r01.nc"


def download_gridsat_file(dt: datetime, dest_dir: str = RAW_DIR) -> str | None:
    """Download a single GridSat-B1 NetCDF file."""
    os.makedirs(dest_dir, exist_ok=True)
    url = gridsat_url(dt)
    dest = os.path.join(dest_dir, gridsat_filename(dt))

    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        return dest  # already downloaded

    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Failed to download {url}: {e}")
        return None

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=gridsat_filename(dt), leave=False
    ) as pbar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return dest


def download_gridsat_daterange(start: datetime, end: datetime,
                                dest_dir: str = RAW_DIR) -> list[str]:
    """Download all 3-hourly GridSat-B1 files for a date range."""
    files = []
    current = start.replace(hour=0, minute=0, second=0)
    while current <= end:
        result = download_gridsat_file(current, dest_dir)
        if result:
            files.append(result)
        current += timedelta(hours=3)
    return files


# === Key storms for RI training/testing ===
# We download GridSat imagery for select high-impact RI storms.
# These are chosen to cover the date ranges of storms we'll use for training.
STORM_DATE_RANGES = {
    # Hurricane Patricia 2015 — the test case (hold-out)
    "EP202015_Patricia": (datetime(2015, 10, 20), datetime(2015, 10, 24)),
    # Hurricane Irma 2017
    "AL112017_Irma": (datetime(2017, 8, 30), datetime(2017, 9, 12)),
    # Hurricane Michael 2018
    "AL142018_Michael": (datetime(2018, 10, 7), datetime(2018, 10, 11)),
    # Hurricane Dorian 2019
    "AL052019_Dorian": (datetime(2019, 8, 24), datetime(2019, 9, 7)),
    # Typhoon Haiyan 2013
    "WP312013_Haiyan": (datetime(2013, 11, 3), datetime(2013, 11, 11)),
}


def download_all_storms(storms: dict | None = None):
    """Download GridSat for all configured storm date ranges."""
    storms = storms or STORM_DATE_RANGES
    all_files = {}

    for storm_id, (start, end) in storms.items():
        print(f"\n[STORM] Downloading GridSat for {storm_id} "
              f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})")
        storm_dir = os.path.join(RAW_DIR, storm_id)
        files = download_gridsat_daterange(start, end, storm_dir)
        all_files[storm_id] = files
        print(f"  → {len(files)} files downloaded for {storm_id}")

    return all_files


def verify_gridsat():
    """List downloaded GridSat files."""
    print(f"\n=== GridSat-B1 Verification ===")
    total = 0
    for storm_dir in sorted(os.listdir(RAW_DIR)):
        storm_path = os.path.join(RAW_DIR, storm_dir)
        if os.path.isdir(storm_path):
            nc_files = [f for f in os.listdir(storm_path) if f.endswith(".nc")]
            total_mb = sum(os.path.getsize(os.path.join(storm_path, f))
                          for f in nc_files) / (1024 * 1024)
            print(f"  {storm_dir}: {len(nc_files)} files, {total_mb:.1f} MB")
            total += len(nc_files)
    print(f"  Total: {total} NetCDF files")
    print(f"  ✓ GridSat download complete\n")


if __name__ == "__main__":
    # By default, download for all configured storms.
    # For faster initial testing, pass --test to download only Patricia.
    if "--test" in sys.argv:
        storms = {"EP202015_Patricia": STORM_DATE_RANGES["EP202015_Patricia"]}
        download_all_storms(storms)
    else:
        download_all_storms()
    verify_gridsat()
