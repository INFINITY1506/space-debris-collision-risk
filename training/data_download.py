"""
data_download.py
----------------
Downloads TLE data from CelesTrak for 18,000+ space objects,
parses orbital elements, and saves a structured satellite catalog CSV.

Usage:
    python training/data_download.py
"""

import os
import sys
import time
import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
TLE_SOURCES = [
    ("active",   "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"),
    ("cosmos",   "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle"),
    ("fengyun",  "https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle"),
    ("iridium",  "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle"),
    ("starlink", "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"),
    ("stations", "https://celestrak.org/NORAD/elements/gp.php?NAME=ISS%20(ZARYA)&FORMAT=tle"),
]
HEADERS = {"User-Agent": "SpaceDebrisPredictor/1.0 (research project)"}


def download_tle(name: str, url: str, retries: int = 3) -> str | None:
    """Download TLE data from a URL and save to data/raw/<name>.tle."""
    out_path = RAW_DIR / f"{name}.tle"
    if out_path.exists():
        log.info(f"[CACHE] {name}.tle already exists, skipping download.")
        return out_path.read_text()

    for attempt in range(1, retries + 1):
        try:
            log.info(f"[{attempt}/{retries}] Downloading {name} TLEs from {url}")
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            text = resp.text.strip()
            if len(text) < 100:
                log.warning(f"Suspiciously short response for {name}: {len(text)} chars")
                continue
            out_path.write_text(text)
            log.info(f"Saved {name}.tle ({len(text) // 1024} KB, {text.count(chr(10))} lines)")
            return text
        except Exception as e:
            log.error(f"Attempt {attempt} failed for {name}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


def parse_tle_block(text: str, source: str) -> list[dict]:
    """
    Parse a block of TLE text (3-line format) into a list of dicts.
    Each dict contains NORAD ID, name, and key orbital elements.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    records = []

    i = 0
    while i < len(lines):
        # Try to identify name + TLE line1 + TLE line2
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            name_line = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]
        elif i + 1 < len(lines) and lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            name_line = "UNKNOWN"
            line1 = lines[i]
            line2 = lines[i + 1]
            i -= 1  # compensate for the i += 3 below
        else:
            i += 1
            continue

        try:
            sat = Satrec.twoline2rv(line1, line2)
            rec = {
                "name": name_line.strip(),
                "norad_id": sat.satnum,
                "epoch_year": sat.epochyr,
                "epoch_day": sat.epochdays,
                "inclination_deg": np.degrees(sat.inclo),
                "raan_deg": np.degrees(sat.nodeo),
                "eccentricity": sat.ecco,
                "arg_perigee_deg": np.degrees(sat.argpo),
                "mean_anomaly_deg": np.degrees(sat.mo),
                "mean_motion_rev_per_day": sat.no_kozai / (2 * np.pi) * 86400,
                "bstar_drag": sat.bstar,
                "classification": sat.classification,
                "source": source,
                "line1": line1,
                "line2": line2,
            }
            # Derived: semi-major axis from mean motion (km)
            mu = 3.986004418e5  # Earth GM km^3/s^2
            n_rad_s = sat.no_kozai  # rad/s (mean motion in rad/min from sgp4 / 60)
            # no_kozai is in rad/min
            n_rad_s_val = sat.no_kozai / 60.0
            if n_rad_s_val > 0:
                rec["semi_major_axis_km"] = (mu / (n_rad_s_val ** 2)) ** (1.0 / 3.0)
            else:
                rec["semi_major_axis_km"] = None

            # Orbital period in minutes
            if sat.no_kozai > 0:
                rec["orbital_period_min"] = 2 * np.pi / sat.no_kozai
            else:
                rec["orbital_period_min"] = None

            # Approximate altitude (km) assuming circular orbit
            EARTH_RADIUS_KM = 6371.0
            if rec["semi_major_axis_km"]:
                rec["altitude_km"] = rec["semi_major_axis_km"] - EARTH_RADIUS_KM
            else:
                rec["altitude_km"] = None

            records.append(rec)
        except Exception as e:
            log.debug(f"Failed to parse TLE block at line {i}: {e}")

        i += 3

    return records


def build_catalog() -> pd.DataFrame:
    """Download all TLE sources and build a unified satellite catalog."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for name, url in TLE_SOURCES:
        text = download_tle(name, url)
        if text is None:
            log.warning(f"Skipping {name} — download failed.")
            continue
        records = parse_tle_block(text, source=name)
        log.info(f"Parsed {len(records):,} objects from {name}")
        all_records.extend(records)

    if not all_records:
        raise RuntimeError("No TLE data could be downloaded. Check your internet connection.")

    df = pd.DataFrame(all_records)

    # Deduplicate by NORAD ID (keep first occurrence — most recent for 'active')
    before = len(df)
    df = df.drop_duplicates(subset=["norad_id"], keep="first").reset_index(drop=True)
    log.info(f"Deduplicated: {before:,} → {len(df):,} unique objects")

    # Filter out objects with no valid orbital data
    df = df.dropna(subset=["semi_major_axis_km", "inclination_deg"])
    log.info(f"After filtering invalid orbits: {len(df):,} objects")

    # Save full catalog
    catalog_path = RAW_DIR / "catalog.csv"
    df.to_csv(catalog_path, index=False)
    log.info(f"Catalog saved: {catalog_path} ({len(df):,} objects)")

    # Also save just the TLE line pairs (for later propagation)
    tle_pairs_path = RAW_DIR / "tle_pairs.csv"
    df[["norad_id", "name", "line1", "line2", "source"]].to_csv(tle_pairs_path, index=False)
    log.info(f"TLE pairs saved: {tle_pairs_path}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the downloaded satellite catalog."""
    print("\n" + "=" * 60)
    print("[*] SATELLITE CATALOG SUMMARY")
    print("=" * 60)
    print(f"  Total objects:      {len(df):,}")
    print(f"  Sources:            {df['source'].value_counts().to_dict()}")
    print(f"  Altitude range:     {df['altitude_km'].min():.0f} - {df['altitude_km'].max():.0f} km")
    print(f"  Inclination range:  {df['inclination_deg'].min():.1f}deg - {df['inclination_deg'].max():.1f}deg")
    print(f"  Eccentricity max:   {df['eccentricity'].max():.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    log.info("Starting TLE data download...")
    df = build_catalog()
    print_summary(df)
    log.info("✅ Data download complete.")
