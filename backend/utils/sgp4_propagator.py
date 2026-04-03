"""
sgp4_propagator.py
------------------
SGP4 orbital propagation for all objects in the satellite catalog.
Generates ECI position/velocity arrays over a 7-day (168-hour) window.

Usage (standalone):
    python backend/utils/sgp4_propagator.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def get_propagation_times(start_dt: datetime, horizon_hours: int = 168, step_hours: int = 1):
    """
    Generate (jd, fr) tuples for julian day propagation times.

    Returns:
        times_dt: list of datetime objects
        times_jd: list of (jd, fr) tuples for sgp4
    """
    times_dt = [start_dt + timedelta(hours=h) for h in range(horizon_hours + 1)]
    times_jd = [jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
                for t in times_dt]
    return times_dt, times_jd


def propagate_object(line1: str, line2: str, times_jd: list, norad_id: int = 0) -> np.ndarray | None:
    """
    Propagate a single object over the given time points.

    Returns:
        ndarray of shape [T, 6] with [x, y, z, vx, vy, vz] in km / km/s
        or None if the satellite has decayed or is invalid.
    """
    try:
        sat = Satrec.twoline2rv(line1, line2)
        positions = []
        for jd, fr in times_jd:
            e, r, v = sat.sgp4(jd, fr)
            if e != 0:  # error code != 0 means propagation error (decay, etc.)
                return None
            positions.append([r[0], r[1], r[2], v[0], v[1], v[2]])
        return np.array(positions, dtype=np.float32)
    except Exception:
        return None


def propagate_all(
    catalog_path: str | Path,
    output_path: str | Path,
    horizon_hours: int = 168,
    step_hours: int = 1,
    epoch: datetime | None = None,
    chunk_size: int = 500,
) -> dict:
    """
    Propagate all objects in the catalog and save positions to HDF5.

    Output HDF5 structure:
        /positions  — float16 array [N, T, 6]
        /norad_ids  — int32 array [N]
        /names      — bytes array [N]
        /valid_mask — bool array [N] (True if propagation succeeded)
        /times      — float64 array [T] (Unix timestamps)
        /metadata   — group with configuration attrs

    Returns:
        dict with statistics
    """
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load catalog
    df = pd.read_csv(catalog_path)
    log.info(f"Loaded catalog: {len(df):,} objects")

    # Epoch for propagation start
    if epoch is None:
        epoch = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)

    times_dt, times_jd = get_propagation_times(epoch, horizon_hours, step_hours)
    T = len(times_jd)
    N = len(df)

    log.info(f"Propagating {N:,} objects × {T} timesteps = {N * T:,} states")
    log.info(f"Epoch: {epoch.isoformat()}, Horizon: {horizon_hours}h")

    # Results storage
    positions_valid = []       # list of [T, 6] arrays
    valid_norad_ids = []
    valid_names = []
    failed = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            log.info(f"  Propagating object {idx:,}/{N:,} ...")
        pos = propagate_object(
            str(row["line1"]),
            str(row["line2"]),
            times_jd,
            norad_id=int(row["norad_id"])
        )
        if pos is not None:
            positions_valid.append(pos.astype(np.float16))
            valid_norad_ids.append(int(row["norad_id"]))
            valid_names.append(str(row["name"]).encode("utf-8"))
        else:
            failed += 1

    log.info(f"Propagation complete: {len(positions_valid):,} valid, {failed:,} failed/decayed")

    # Stack arrays
    positions_array = np.stack(positions_valid, axis=0)  # [N_valid, T, 6]
    norad_ids_array = np.array(valid_norad_ids, dtype=np.int32)
    times_unix = np.array([t.timestamp() for t in times_dt], dtype=np.float64)

    # Save to HDF5
    with h5py.File(output_path, "w") as f:
        f.create_dataset("positions", data=positions_array, compression="gzip", compression_opts=4)
        f.create_dataset("norad_ids", data=norad_ids_array)
        f.create_dataset("names", data=np.array(valid_names))
        f.create_dataset("times", data=times_unix)

        md = f.create_group("metadata")
        md.attrs["epoch"] = epoch.isoformat()
        md.attrs["horizon_hours"] = horizon_hours
        md.attrs["step_hours"] = step_hours
        md.attrs["n_objects"] = len(positions_valid)
        md.attrs["n_timesteps"] = T
        md.attrs["n_failed"] = failed
        md.attrs["features"] = "x,y,z,vx,vy,vz (ECI frame, km/km/s)"

    size_mb = output_path.stat().st_size / (1024 ** 2)
    log.info(f"Saved: {output_path} ({size_mb:.1f} MB)")
    log.info(f"Array shape: {positions_array.shape} ({positions_array.dtype})")

    return {
        "n_valid": len(positions_valid),
        "n_failed": failed,
        "n_timesteps": T,
        "shape": positions_array.shape,
        "output_path": str(output_path),
    }


def load_positions(positions_path: str | Path) -> dict:
    """
    Load propagated positions from HDF5 file.

    Returns dict with keys: positions, norad_ids, names, times, metadata
    """
    with h5py.File(positions_path, "r") as f:
        return {
            "positions": f["positions"][:],          # [N, T, 6]
            "norad_ids": f["norad_ids"][:],          # [N]
            "names": [n.decode("utf-8") for n in f["names"][:]],
            "times": f["times"][:],                  # [T] Unix timestamps
            "metadata": dict(f["metadata"].attrs),
        }


def compute_miss_distance(
    pos1: np.ndarray,   # [T, 3]
    pos2: np.ndarray,   # [T, 3]
) -> np.ndarray:
    """
    Compute miss distance (km) at each timestep between two objects.

    Returns: [T] array of distances in km
    """
    diff = pos1 - pos2  # [T, 3]
    return np.linalg.norm(diff, axis=1)  # [T]


def find_tca(distances: np.ndarray, times: np.ndarray) -> tuple[float, float, int]:
    """
    Find the Time of Closest Approach (TCA).

    Returns:
        min_distance (km), tca_unix_timestamp, tca_idx
    """
    tca_idx = int(np.argmin(distances))
    return float(distances[tca_idx]), float(times[tca_idx]), tca_idx


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    catalog = Path("data/raw/catalog.csv")
    output = Path("data/processed/positions.h5")

    if not catalog.exists():
        print("❌ Catalog not found. Run training/data_download.py first.")
        sys.exit(1)

    stats = propagate_all(catalog, output)
    print(f"\n✅ Propagation complete. Shape: {stats['shape']}")
    print(f"   Valid objects: {stats['n_valid']:,}")
    print(f"   Failed/decayed: {stats['n_failed']:,}")
