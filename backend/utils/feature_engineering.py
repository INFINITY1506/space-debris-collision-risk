"""
feature_engineering.py
-----------------------
Computes the full feature vector (30+ features) for each conjunction event.
Used both during preprocessing and live inference.
"""

import numpy as np
import pandas as pd
from typing import Optional

# Earth constants
EARTH_RADIUS_KM = 6371.0
EARTH_MU = 3.986004418e5  # km^3/s^2


def orbital_period(semi_major_axis_km: float) -> float:
    """Orbital period in seconds from semi-major axis (km)."""
    return 2 * np.pi * np.sqrt(semi_major_axis_km ** 3 / EARTH_MU)


def semi_major_axis_from_velocity(r_km: np.ndarray, v_km_s: np.ndarray) -> float:
    """Compute semi-major axis from ECI position and velocity via vis-viva."""
    r_mag = np.linalg.norm(r_km)
    v_mag = np.linalg.norm(v_km_s)
    if r_mag < 1e-10:
        return 0.0
    inv_a = 2.0 / r_mag - v_mag ** 2 / EARTH_MU
    return 1.0 / inv_a if abs(inv_a) > 1e-10 else 0.0


def eccentricity_vector(r_km: np.ndarray, v_km_s: np.ndarray) -> np.ndarray:
    """Eccentricity vector from ECI state."""
    r_mag = np.linalg.norm(r_km)
    v_mag = np.linalg.norm(v_km_s)
    h = np.cross(r_km, v_km_s)
    e_vec = (np.cross(v_km_s, h) / EARTH_MU) - (r_km / r_mag)
    return e_vec


def inclination_from_state(r_km: np.ndarray, v_km_s: np.ndarray) -> float:
    """Inclination (degrees) from ECI state."""
    h = np.cross(r_km, v_km_s)
    h_mag = np.linalg.norm(h)
    if h_mag < 1e-10:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(h[2] / h_mag, -1, 1))))


def compute_rsw_decomposition(
    r_rel: np.ndarray,  # relative position (km)
    r_primary: np.ndarray,  # primary position (km)
    v_primary: np.ndarray,  # primary velocity (km/s)
) -> tuple[float, float, float]:
    """
    Decompose relative position into RSW (Radial-Along-track-Cross-track) frame.

    Returns: (radial_miss_km, intrack_miss_km, crosstrack_miss_km)
    """
    r_hat = r_primary / (np.linalg.norm(r_primary) + 1e-10)
    h_vec = np.cross(r_primary, v_primary)
    w_hat = h_vec / (np.linalg.norm(h_vec) + 1e-10)
    s_hat = np.cross(w_hat, r_hat)

    radial = float(np.dot(r_rel, r_hat))
    intrack = float(np.dot(r_rel, s_hat))
    crosstrack = float(np.dot(r_rel, w_hat))
    return radial, intrack, crosstrack


def compute_conjunction_features(
    primary_pos_tca: np.ndarray,       # [3] ECI position of satellite at TCA (km)
    primary_vel_tca: np.ndarray,       # [3] ECI velocity of satellite at TCA (km/s)
    secondary_pos_tca: np.ndarray,     # [3] ECI position of debris at TCA (km)
    secondary_vel_tca: np.ndarray,     # [3] ECI velocity of debris at TCA (km/s)
    time_to_tca_hours: float,          # hours until TCA
    primary_catalog_row: Optional[pd.Series] = None,
    secondary_catalog_row: Optional[pd.Series] = None,
    primary_mass_kg: float = 500.0,    # default satellite mass
    secondary_mass_kg: float = 1.0,    # default debris mass
    primary_area_m2: float = 5.0,      # default cross-section
    secondary_area_m2: float = 0.01,   # default debris cross-section
) -> dict:
    """
    Compute the full feature vector (30+ features) for a conjunction event.
    Returns a dict of all features.
    """
    features = {}

    # -------------------------------------------------------------------
    # 1. Primary Orbital Features (6)
    # -------------------------------------------------------------------
    sma_p = semi_major_axis_from_velocity(primary_pos_tca, primary_vel_tca)
    e_vec_p = eccentricity_vector(primary_pos_tca, primary_vel_tca)
    ecc_p = float(np.linalg.norm(e_vec_p))
    inc_p = inclination_from_state(primary_pos_tca, primary_vel_tca)
    h_p = np.cross(primary_pos_tca, primary_vel_tca)
    raan_p = float(np.degrees(np.arctan2(h_p[0], -h_p[1]))) % 360
    arg_p = 0.0  # simplified
    r_p = np.linalg.norm(primary_pos_tca)
    ma_p = float(np.degrees(np.arctan2(primary_pos_tca[1], primary_pos_tca[0]))) % 360

    features["primary_semi_major_axis"] = sma_p
    features["primary_eccentricity"] = ecc_p
    features["primary_inclination"] = inc_p
    features["primary_raan"] = raan_p
    features["primary_arg_perigee"] = arg_p
    features["primary_mean_anomaly"] = ma_p
    features["primary_altitude"] = r_p - EARTH_RADIUS_KM

    # -------------------------------------------------------------------
    # 2. Secondary (Debris) Orbital Features (6)
    # -------------------------------------------------------------------
    sma_s = semi_major_axis_from_velocity(secondary_pos_tca, secondary_vel_tca)
    e_vec_s = eccentricity_vector(secondary_pos_tca, secondary_vel_tca)
    ecc_s = float(np.linalg.norm(e_vec_s))
    inc_s = inclination_from_state(secondary_pos_tca, secondary_vel_tca)
    h_s = np.cross(secondary_pos_tca, secondary_vel_tca)
    raan_s = float(np.degrees(np.arctan2(h_s[0], -h_s[1]))) % 360
    r_s = np.linalg.norm(secondary_pos_tca)
    ma_s = float(np.degrees(np.arctan2(secondary_pos_tca[1], secondary_pos_tca[0]))) % 360

    features["secondary_semi_major_axis"] = sma_s
    features["secondary_eccentricity"] = ecc_s
    features["secondary_inclination"] = inc_s
    features["secondary_raan"] = raan_s
    features["secondary_arg_perigee"] = 0.0
    features["secondary_mean_anomaly"] = ma_s
    features["secondary_altitude"] = r_s - EARTH_RADIUS_KM

    # -------------------------------------------------------------------
    # 3. Relative Features (10)
    # -------------------------------------------------------------------
    r_rel = secondary_pos_tca - primary_pos_tca
    v_rel = secondary_vel_tca - primary_vel_tca

    miss_distance = float(np.linalg.norm(r_rel))
    rel_velocity = float(np.linalg.norm(v_rel))
    alt_diff = features["secondary_altitude"] - features["primary_altitude"]
    inc_diff = abs(inc_s - inc_p)
    raan_diff = abs(raan_s - raan_p)
    period_ratio = (orbital_period(sma_s) / orbital_period(sma_p)) if (sma_p > 0 and sma_s > 0) else 1.0
    radial, intrack, crosstrack = compute_rsw_decomposition(r_rel, primary_pos_tca, primary_vel_tca)

    features["miss_distance"] = miss_distance
    features["relative_velocity"] = rel_velocity
    features["time_to_tca"] = time_to_tca_hours
    features["altitude_difference"] = alt_diff
    features["inclination_difference"] = inc_diff
    features["raan_difference"] = raan_diff
    features["radial_miss"] = radial
    features["intrack_miss"] = intrack
    features["crosstrack_miss"] = crosstrack
    features["orbital_period_ratio"] = period_ratio

    # -------------------------------------------------------------------
    # 4. Physical Features (5)
    # -------------------------------------------------------------------
    combined_mass = primary_mass_kg + secondary_mass_kg
    combined_area = primary_area_m2 + secondary_area_m2
    # Kinetic energy of relative motion (Joules)
    v_rel_m_s = rel_velocity * 1000.0
    kinetic_energy = 0.5 * secondary_mass_kg * v_rel_m_s ** 2
    # Hardness factor (empirical measure of penetrability)
    hardness_factor = min(combined_mass / (combined_area + 1e-10), 1e6)
    # Momentum transfer (kg * m/s)
    momentum_transfer = secondary_mass_kg * v_rel_m_s

    features["combined_mass"] = combined_mass
    features["combined_cross_section"] = combined_area
    features["kinetic_energy"] = kinetic_energy
    features["hardness_factor"] = hardness_factor
    features["momentum_transfer"] = momentum_transfer

    # -------------------------------------------------------------------
    # 5. Temporal Features (3)
    # -------------------------------------------------------------------
    # Rough orbital decay rate based on altitude (simplified)
    altitude_avg = (features["primary_altitude"] + features["secondary_altitude"]) / 2
    if altitude_avg < 300:
        decay_rate = 2.0  # km/day
    elif altitude_avg < 500:
        decay_rate = 0.1
    elif altitude_avg < 800:
        decay_rate = 0.01
    else:
        decay_rate = 0.001

    features["hours_until_tca"] = time_to_tca_hours
    features["orbital_decay_rate"] = decay_rate
    features["time_since_epoch"] = 0.0  # filled during preprocessing

    return features


def features_to_array(features: dict, feature_names: list | None = None) -> np.ndarray:
    """Convert feature dict to numpy array in a consistent order."""
    if feature_names is None:
        feature_names = get_feature_names()
    return np.array([features.get(k, 0.0) for k in feature_names], dtype=np.float32)


def get_feature_names() -> list[str]:
    """Return the canonical ordered list of 30 feature names."""
    return [
        # Primary orbital (7)
        "primary_semi_major_axis", "primary_eccentricity", "primary_inclination",
        "primary_raan", "primary_arg_perigee", "primary_mean_anomaly", "primary_altitude",
        # Secondary orbital (7)
        "secondary_semi_major_axis", "secondary_eccentricity", "secondary_inclination",
        "secondary_raan", "secondary_arg_perigee", "secondary_mean_anomaly", "secondary_altitude",
        # Relative (10)
        "miss_distance", "relative_velocity", "time_to_tca", "altitude_difference",
        "inclination_difference", "raan_difference", "radial_miss", "intrack_miss",
        "crosstrack_miss", "orbital_period_ratio",
        # Physical (5)
        "combined_mass", "combined_cross_section", "kinetic_energy",
        "hardness_factor", "momentum_transfer",
        # Temporal (3)
        "hours_until_tca", "orbital_decay_rate", "time_since_epoch",
    ]


def compute_collision_probability(
    miss_distance_km: float,
    relative_velocity_km_s: float,
    combined_area_m2: float = 0.05,
    covariance_scale_km: float = 0.5,
) -> float:
    """
    Chan's formula approximation for collision probability.
    P_c = A_combined / (2*pi*sigma_r * sigma_t) * exp(-d^2 / (2*sigma^2))
    """
    sigma = covariance_scale_km
    A_km2 = combined_area_m2 / 1e6  # convert m^2 -> km^2
    if sigma < 1e-10 or miss_distance_km < 0:
        return 0.0
    p = (A_km2 / (2 * np.pi * sigma ** 2)) * np.exp(-(miss_distance_km ** 2) / (2 * sigma ** 2))
    return float(np.clip(p, 0.0, 1.0))


def assign_risk_label(
    collision_probability: float,
    high_threshold: float = 1e-3,
    medium_threshold: float = 1e-4,
) -> int:
    """
    Assign risk label.
    2 = HIGH, 1 = MEDIUM, 0 = LOW
    """
    if collision_probability >= high_threshold:
        return 2
    elif collision_probability >= medium_threshold:
        return 1
    return 0
