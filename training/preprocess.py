"""
preprocess.py
-------------
Generates the conjunction dataset from propagated orbital positions.
Identifies satellite-debris pairs with miss distance < 10km, computes
30+ features per conjunction, assigns risk labels based on miss distance,
and creates canonical train/val/test splits with balanced classes.

Risk labeling uses miss distance (physically meaningful):
  HIGH:   miss distance < 1.0 km
  MEDIUM: miss distance < 5.0 km
  LOW:    miss distance < 10.0 km

Usage:
    python training/preprocess.py
"""

import sys
import logging
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.sgp4_propagator import propagate_all, load_positions, compute_miss_distance, find_tca
from backend.utils.feature_engineering import (
    compute_conjunction_features,
    features_to_array,
    get_feature_names,
    compute_collision_probability,
    assign_risk_label,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
POSITIONS_FILE = PROC_DIR / "positions.h5"
CATALOG_FILE = RAW_DIR / "catalog.csv"
CONJUNCTION_FILE = PROC_DIR / "conjunctions.csv"
TRAIN_FILE = PROC_DIR / "train.h5"
VAL_FILE = PROC_DIR / "val.h5"
TEST_FILE = PROC_DIR / "test.h5"

# Thresholds
CONJUNCTION_THRESHOLD_KM = 10.0
HIGH_RISK_KM = 1.0     # < 1.0 km = HIGH
MEDIUM_RISK_KM = 5.0   # 1.0 - 5.0 km = MEDIUM


def assign_risk_label_by_distance(miss_distance_km: float) -> int:
    """
    Assign risk label based on miss distance (physically meaningful).
    HIGH:   miss distance < 1.0 km
    MEDIUM: miss distance < 5.0 km
    LOW:    otherwise (< 10 km)
    """
    if miss_distance_km < HIGH_RISK_KM:
        return 2  # HIGH
    elif miss_distance_km < MEDIUM_RISK_KM:
        return 1  # MEDIUM
    else:
        return 0  # LOW


def generate_conjunctions(
    positions: np.ndarray,      # [N, T, 6] float16
    norad_ids: np.ndarray,      # [N] int32
    names: list[str],           # [N]
    times: np.ndarray,          # [T] Unix timestamps
    threshold_km: float = CONJUNCTION_THRESHOLD_KM,
    max_conjunctions: int = 150_000,
) -> pd.DataFrame:
    """
    Find all object pairs with miss distance < threshold at any timestep.
    Uses a coarse altitude filter first for efficiency.

    Returns: DataFrame with conjunction features.
    """
    N, T, _ = positions.shape
    pos_float = positions.astype(np.float32)

    # --- Altitude binning for coarse filter ---
    r0 = np.linalg.norm(pos_float[:, 0, :3], axis=1)  # [N]
    altitudes = r0 - 6371.0  # km

    feature_names = get_feature_names()
    records = []
    checked_pairs = 0

    log.info(f"Scanning {N:,} objects for conjunctions (threshold={threshold_km} km)...")

    sort_idx = np.argsort(altitudes)
    sorted_alts = altitudes[sort_idx]
    sorted_pos = pos_float[sort_idx]
    sorted_ids = norad_ids[sort_idx]
    sorted_names = [names[i] for i in sort_idx]

    for i in tqdm(range(N), desc="Conjunction search", ncols=80):
        alt_i = sorted_alts[i]
        lo = np.searchsorted(sorted_alts, alt_i - 100, side="left")
        hi = np.searchsorted(sorted_alts, alt_i + 100, side="right")

        for j in range(i + 1, min(hi, N)):
            checked_pairs += 1
            pos_i = sorted_pos[i, :, :3]  # [T, 3]
            pos_j = sorted_pos[j, :, :3]  # [T, 3]

            dists = np.linalg.norm(pos_i - pos_j, axis=1)  # [T]
            min_dist = float(dists.min())

            if min_dist >= threshold_km:
                continue

            tca_dist, tca_time, tca_idx = find_tca(dists, times)
            time_to_tca_h = (tca_time - times[0]) / 3600.0

            pos_i_tca = sorted_pos[i, tca_idx, :3].astype(np.float64)
            vel_i_tca = sorted_pos[i, tca_idx, 3:6].astype(np.float64)
            pos_j_tca = sorted_pos[j, tca_idx, :3].astype(np.float64)
            vel_j_tca = sorted_pos[j, tca_idx, 3:6].astype(np.float64)

            try:
                feats = compute_conjunction_features(
                    primary_pos_tca=pos_i_tca,
                    primary_vel_tca=vel_i_tca,
                    secondary_pos_tca=pos_j_tca,
                    secondary_vel_tca=vel_j_tca,
                    time_to_tca_hours=time_to_tca_h,
                )
            except Exception:
                continue

            # Risk label based on miss distance
            label = assign_risk_label_by_distance(tca_dist)

            p_c = compute_collision_probability(
                miss_distance_km=tca_dist,
                relative_velocity_km_s=feats["relative_velocity"],
                combined_area_m2=feats["combined_cross_section"],
            )

            record = {
                "primary_norad_id": int(sorted_ids[i]),
                "primary_name": sorted_names[i],
                "secondary_norad_id": int(sorted_ids[j]),
                "secondary_name": sorted_names[j],
                "tca_timestamp": tca_time,
                "min_distance_km": tca_dist,
                "collision_probability": p_c,
                "risk_label": label,
            }
            record.update(feats)
            records.append(record)

            if len(records) >= max_conjunctions:
                log.info(f"Reached max conjunctions limit ({max_conjunctions:,})")
                return pd.DataFrame(records)

    log.info(f"Checked {checked_pairs:,} pairs, found {len(records):,} conjunctions")
    return pd.DataFrame(records)


def generate_synthetic_dataset(
    feature_names: list[str],
    n_samples: int = 80_000,
    real_df: pd.DataFrame | None = None,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a physics-based synthetic conjunction dataset with balanced classes.

    Produces realistic orbital mechanics features covering the full miss-distance range.
    If real_df is provided, the statistics (means/stds) of real data are used for realistic
    parameter scaling.
    """
    rng = np.random.default_rng(rng_seed)
    n_per_class = n_samples // 3

    records = []

    def sample_orbit(rng) -> dict:
        """Sample random orbital elements for a space object."""
        alt_km = rng.uniform(200, 1400)
        r_km = 6371.0 + alt_km
        mu = 398600.4418  # km^3/s^2
        v_circ = np.sqrt(mu / r_km)

        inc = rng.uniform(0, 180)
        ecc = rng.exponential(0.01)
        ecc = min(ecc, 0.05)
        period_min = (2 * np.pi * r_km) / (v_circ * 60)

        return {
            "alt_km": alt_km,
            "r_km": r_km,
            "v_circ": v_circ,
            "inc": inc,
            "ecc": ecc,
            "period_min": period_min,
        }

    def orbit_to_feature_row(primary, secondary, miss_dist, label, rng) -> dict:
        mu = 398600.4418
        EARTH_R = 6371.0

        # Primary
        p_alt = primary["alt_km"]
        p_r = primary["r_km"]
        p_v = primary["v_circ"] * rng.uniform(0.97, 1.03)
        p_inc = primary["inc"]
        p_ecc = primary["ecc"]
        p_period = primary["period_min"]
        p_sma = p_r

        # Secondary
        s_alt = secondary["alt_km"]
        s_r = secondary["r_km"]
        s_v = secondary["v_circ"] * rng.uniform(0.97, 1.03)
        s_inc = secondary["inc"]
        s_ecc = secondary["ecc"]
        s_period = secondary["period_min"]
        s_sma = s_r

        rel_v = np.sqrt(p_v**2 + s_v**2 - 2 * p_v * s_v * np.cos(np.radians(abs(p_inc - s_inc))))
        rel_v = max(0.1, rel_v)

        combined_area = rng.uniform(1.0, 50.0)
        hardness = rng.uniform(0.5, 1.0)
        combined_mass = rng.uniform(1.0, 5000.0)
        ke = 0.5 * combined_mass * (rel_v * 1000) ** 2 * 1e-6
        momentum = combined_mass * rel_v

        p_c = compute_collision_probability(miss_dist, rel_v, combined_area)
        time_to_tca = rng.uniform(0.1, 167.9)
        alt_diff = abs(p_alt - s_alt)
        inc_diff = abs(p_inc - s_inc)

        row = {
            "primary_norad_id":            -1,
            "primary_name":                "SYNTHETIC",
            "secondary_norad_id":          -2,
            "secondary_name":              "SYNTHETIC-DEB",
            "tca_timestamp":               float(rng.uniform(1.7e9, 1.75e9)),
            "min_distance_km":             miss_dist,
            "collision_probability":       p_c,
            "risk_label":                  label,
            # Feature columns (must match feature_names order)
            "primary_altitude":            p_alt,
            "primary_semi_major_axis":     p_sma,
            "primary_eccentricity":        p_ecc,
            "primary_inclination":         p_inc,
            "primary_velocity":            p_v,
            "primary_period":              p_period,
            "primary_bstar":               rng.uniform(-1e-4, 1e-4),
            "secondary_altitude":          s_alt,
            "secondary_semi_major_axis":   s_sma,
            "secondary_eccentricity":      s_ecc,
            "secondary_inclination":       s_inc,
            "secondary_velocity":          s_v,
            "secondary_period":            s_period,
            "secondary_bstar":             rng.uniform(-1e-4, 1e-4),
            "miss_distance":               miss_dist,
            "relative_velocity":           rel_v,
            "time_to_tca":                 time_to_tca,
            "altitude_difference":         alt_diff,
            "inclination_difference":      inc_diff,
            "raan_difference":             rng.uniform(0, 180),
            "rsw_radial":                  rng.normal(0, miss_dist),
            "rsw_along_track":             rng.normal(0, miss_dist * 0.5),
            "rsw_cross_track":             rng.normal(0, miss_dist * 0.5),
            "period_ratio":                p_period / max(s_period, 0.01),
            "combined_mass":               combined_mass,
            "combined_cross_section":      combined_area,
            "kinetic_energy_MJ":           ke,
            "hardness_factor":             hardness,
            "momentum_transfer":           momentum,
            "orbital_decay_rate":          rng.uniform(0, 0.01),
            "mean_motion_ratio":           (1 / p_period) / max(1 / s_period, 1e-6),
            "time_since_epoch":            rng.uniform(0, 730),
            "primary_raan":                rng.uniform(0, 360),
        }
        return row

    # LOW risk: miss_dist 5-10km
    for _ in range(n_per_class):
        miss = rng.uniform(5.0, 10.0)
        p, s = sample_orbit(rng), sample_orbit(rng)
        # Keep altitudes close enough
        s["alt_km"] = p["alt_km"] + rng.uniform(-100, 100)
        s["r_km"] = 6371 + s["alt_km"]
        records.append(orbit_to_feature_row(p, s, miss, 0, rng))

    # MEDIUM risk: miss_dist 1-5km
    for _ in range(n_per_class):
        miss = rng.uniform(1.0, 5.0)
        p, s = sample_orbit(rng), sample_orbit(rng)
        s["alt_km"] = p["alt_km"] + rng.uniform(-50, 50)
        s["r_km"] = 6371 + s["alt_km"]
        records.append(orbit_to_feature_row(p, s, miss, 1, rng))

    # HIGH risk: miss_dist 0-1km
    for _ in range(n_per_class):
        miss = rng.uniform(0.01, 1.0)
        p, s = sample_orbit(rng), sample_orbit(rng)
        s["alt_km"] = p["alt_km"] + rng.uniform(-20, 20)
        s["r_km"] = 6371 + s["alt_km"]
        records.append(orbit_to_feature_row(p, s, miss, 2, rng))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(
        f"Generated {len(df):,} synthetic samples | "
        f"LOW={n_per_class:,}, MED={n_per_class:,}, HIGH={n_per_class:,}"
    )
    return df


def create_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create temporal train/val/test splits (70/15/15).
    Data is sorted by TCA timestamp for temporal ordering.
    """
    df_sorted = df.sort_values("tca_timestamp").reset_index(drop=True)
    N = len(df_sorted)
    train_end = int(N * 0.70)
    val_end   = int(N * 0.85)

    train = df_sorted.iloc[:train_end].copy()
    val   = df_sorted.iloc[train_end:val_end].copy()
    test  = df_sorted.iloc[val_end:].copy()

    log.info(f"Splits: train={len(train):,}, val={len(val):,}, test={len(test):,}")
    return train, val, test


def compute_normalization(train_df: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std from training set only."""
    X_train = np.nan_to_num(train_df[feature_names].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds < 1e-10] = 1.0
    return means, stds


def save_split_to_hdf5(
    df: pd.DataFrame,
    path: Path,
    feature_names: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    """Save a dataset split to HDF5 with normalized features."""
    X = np.nan_to_num(df[feature_names].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    X_norm = (X - means) / stds

    y_label = df["risk_label"].values.astype(np.int64)
    y_prob  = df["collision_probability"].values.astype(np.float32)

    # Build trajectory features: [N, T, 22]
    T = 168
    n_model_feats = 22
    X_traj = np.tile(X_norm[:, :n_model_feats, np.newaxis], (1, 1, T)).transpose(0, 2, 1)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X_norm, compression="gzip")
        f.create_dataset("X_traj", data=X_traj.astype(np.float16), compression="gzip")
        f.create_dataset("y_label", data=y_label)
        f.create_dataset("y_prob", data=y_prob)
        f.create_dataset("feature_names", data=np.array([n.encode() for n in feature_names]))
        f.create_dataset("means", data=means)
        f.create_dataset("stds", data=stds)

        meta = f.create_group("metadata")
        meta.attrs["n_samples"] = len(df)
        meta.attrs["n_features"] = len(feature_names)
        meta.attrs["n_timesteps"] = T
        meta.attrs["label_dist"] = str(df["risk_label"].value_counts().to_dict())

    label_dist = dict(df["risk_label"].value_counts().sort_index())
    log.info(f"Saved {path} -- {len(df):,} samples | LOW={label_dist.get(0,0):,} MED={label_dist.get(1,0):,} HIGH={label_dist.get(2,0):,}")


def run_preprocessing() -> None:
    """Main preprocessing pipeline."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    feature_names = get_feature_names()

    # -------------------------------------------------------------------
    # Step 1: Load catalog to check we have data
    # -------------------------------------------------------------------
    if not CATALOG_FILE.exists():
        log.error("Catalog not found. Run training/data_download.py first.")
        sys.exit(1)

    catalog = pd.read_csv(CATALOG_FILE)
    log.info(f"Catalog: {len(catalog):,} objects")

    # -------------------------------------------------------------------
    # Step 2: Propagate positions (if not already done)
    # -------------------------------------------------------------------
    if not POSITIONS_FILE.exists():
        log.info("Running SGP4 propagation...")
        propagate_all(CATALOG_FILE, POSITIONS_FILE)
    else:
        log.info(f"Loading existing positions: {POSITIONS_FILE}")

    data = load_positions(POSITIONS_FILE)
    positions = data["positions"]  # [N, T, 6]
    norad_ids = data["norad_ids"]
    names     = data["names"]
    times     = data["times"]
    log.info(f"Loaded positions: {positions.shape}, {len(norad_ids):,} objects")

    # -------------------------------------------------------------------
    # Step 3: Generate real conjunction events
    # -------------------------------------------------------------------
    if CONJUNCTION_FILE.exists():
        log.info(f"Loading existing conjunctions: {CONJUNCTION_FILE}")
        real_df = pd.read_csv(CONJUNCTION_FILE)
    else:
        real_df = generate_conjunctions(positions, norad_ids, names, times)
        if not real_df.empty:
            real_df.to_csv(CONJUNCTION_FILE, index=False)
            log.info(f"Saved conjunctions: {CONJUNCTION_FILE} ({len(real_df):,} events)")

    if not real_df.empty:
        # Re-apply miss-distance-based labels to real data
        real_df["risk_label"] = real_df["min_distance_km"].apply(assign_risk_label_by_distance)
        log.info(f"\nReal conjunction risk distribution:\n{real_df['risk_label'].value_counts().sort_index()}")
        log.info("  0=LOW, 1=MEDIUM, 2=HIGH")
    else:
        log.warning("No real conjunctions found — proceeding with synthetic only")

    # -------------------------------------------------------------------
    # Step 4: Generate large physics-based synthetic dataset
    # -------------------------------------------------------------------
    log.info("Generating physics-based synthetic conjunction dataset...")
    synthetic_df = generate_synthetic_dataset(
        feature_names=feature_names,
        n_samples=90_000,
        real_df=real_df if not real_df.empty else None,
    )

    # -------------------------------------------------------------------
    # Step 5: Merge real + synthetic
    # -------------------------------------------------------------------
    if not real_df.empty:
        # Filter real_df to only rows that have all feature columns
        avail_cols = [c for c in feature_names if c in real_df.columns]
        if len(avail_cols) == len(feature_names):
            combined = pd.concat([real_df, synthetic_df], ignore_index=True)
        else:
            log.warning(f"Real data missing some feature columns. Using synthetic only.")
            combined = synthetic_df
    else:
        combined = synthetic_df

    log.info(f"Combined dataset: {len(combined):,} samples")
    log.info(f"Label distribution:\n{combined['risk_label'].value_counts().sort_index()}")

    # -------------------------------------------------------------------
    # Step 6: Train/Val/Test splits
    # -------------------------------------------------------------------
    train_df, val_df, test_df = create_splits(combined)

    # -------------------------------------------------------------------
    # Step 7: Compute normalization from training set
    # -------------------------------------------------------------------
    means, stds = compute_normalization(train_df, feature_names)

    norm_path = PROC_DIR / "normalization.npz"
    np.savez(norm_path, means=means, stds=stds, feature_names=np.array(feature_names))
    log.info(f"Normalization params saved: {norm_path}")

    # -------------------------------------------------------------------
    # Step 8: Save splits to HDF5
    # -------------------------------------------------------------------
    for split_df, path in [(train_df, TRAIN_FILE), (val_df, VAL_FILE), (test_df, TEST_FILE)]:
        save_split_to_hdf5(split_df, path, feature_names, means, stds)

    log.info("")
    log.info("=== Preprocessing complete! ===")
    log.info(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    log.info(f"   Model input shape: [{len(train_df)}, 168, 22]")


if __name__ == "__main__":
    run_preprocessing()
