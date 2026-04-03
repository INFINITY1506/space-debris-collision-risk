"""
bplane.py
---------
B-plane conjunction geometry computation.

The B-plane is perpendicular to the relative velocity vector at the
secondary object, passing through the primary. Standard in orbital safety
(Alfano 2005, Chan 1997).

Outputs: B·T, B·R components, covariance ellipses, hard-body radius,
and B-plane collision probability.
"""

import numpy as np
from typing import Optional


def compute_bplane_parameters(
    r_primary: np.ndarray,          # [3] ECI position at TCA (km)
    v_primary: np.ndarray,          # [3] ECI velocity at TCA (km/s)
    r_secondary: np.ndarray,        # [3] ECI position at TCA (km)
    v_secondary: np.ndarray,        # [3] ECI velocity at TCA (km/s)
    sigma_primary_km: float = 0.1,  # position uncertainty (km)
    sigma_secondary_km: float = 0.5,
    primary_radius_m: float = 5.0,
    secondary_radius_m: float = 0.1,
) -> dict:
    """
    Compute B-plane parameters for a conjunction event.

    Returns dict with B·T, B·R, covariance ellipses (1σ/2σ/3σ),
    hard-body radius, and collision probability in the B-plane.
    """
    # Relative velocity and miss vector
    v_rel = v_secondary - v_primary
    v_rel_mag = np.linalg.norm(v_rel)
    r_miss = r_secondary - r_primary
    miss_distance = np.linalg.norm(r_miss)

    if v_rel_mag < 1e-10:
        return _empty_bplane(miss_distance)

    # B-plane unit vectors
    # S: along incoming relative velocity (asymptote direction)
    s_hat = v_rel / v_rel_mag

    # T: in B-plane, toward ascending node (cross S with Z-axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    t_vec = np.cross(s_hat, z_axis)
    t_mag = np.linalg.norm(t_vec)
    if t_mag < 1e-10:
        # Relative velocity nearly along Z — use X-axis instead
        t_vec = np.cross(s_hat, np.array([1.0, 0.0, 0.0]))
        t_mag = np.linalg.norm(t_vec)
    t_hat = t_vec / t_mag

    # R: completes right-hand system (roughly radial in B-plane)
    r_hat = np.cross(s_hat, t_hat)

    # Project miss vector onto B-plane
    b_t = float(np.dot(r_miss, t_hat))
    b_r = float(np.dot(r_miss, r_hat))
    b_mag = np.sqrt(b_t**2 + b_r**2)

    # Combined covariance in B-plane (simplified: isotropic per object)
    sigma_t = np.sqrt(sigma_primary_km**2 + sigma_secondary_km**2)
    sigma_r = sigma_t  # isotropic assumption

    # Build 2x2 covariance matrix
    cov = np.array([[sigma_t**2, 0.0], [0.0, sigma_r**2]])

    # Eigendecomposition for ellipse parameters
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    semi_major = np.sqrt(max(eigenvalues))
    semi_minor = np.sqrt(min(eigenvalues))
    rotation_deg = float(np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])))

    # Covariance ellipses at 1σ, 2σ, 3σ
    ellipses = []
    for n_sigma in [1, 2, 3]:
        ellipses.append({
            "sigma": n_sigma,
            "semi_major_km": round(float(semi_major * n_sigma), 6),
            "semi_minor_km": round(float(semi_minor * n_sigma), 6),
            "rotation_deg": round(float(rotation_deg), 2),
        })

    # Hard-body collision radius (km)
    hb_radius_km = (primary_radius_m + secondary_radius_m) / 1000.0

    # B-plane collision probability (Chan's method in B-plane)
    if sigma_t > 1e-12 and sigma_r > 1e-12:
        pc = (hb_radius_km**2 / (2.0 * sigma_t * sigma_r)) * \
             np.exp(-0.5 * (b_t**2 / sigma_t**2 + b_r**2 / sigma_r**2))
        pc = float(np.clip(pc, 0.0, 1.0))
    else:
        pc = 0.0

    # Encounter geometry angles
    theta_deg = float(np.degrees(np.arctan2(b_r, b_t)))
    v_rel_angle_deg = float(np.degrees(np.arccos(
        np.clip(np.dot(s_hat, r_miss / (miss_distance + 1e-10)), -1, 1)
    )))

    return {
        "b_t_km": round(float(b_t), 6),
        "b_r_km": round(float(b_r), 6),
        "b_magnitude_km": round(float(b_mag), 6),
        "miss_distance_km": round(float(miss_distance), 6),
        "hard_body_radius_km": round(float(hb_radius_km), 6),
        "collision_probability_bplane": round(float(pc), 10),
        "sigma_t_km": round(float(sigma_t), 6),
        "sigma_r_km": round(float(sigma_r), 6),
        "ellipses": ellipses,
        "encounter_angle_deg": round(float(theta_deg), 2),
        "relative_velocity_angle_deg": round(float(v_rel_angle_deg), 2),
        "relative_velocity_km_s": round(float(v_rel_mag), 4),
    }


def _empty_bplane(miss_distance: float) -> dict:
    """Return empty B-plane data when relative velocity is near zero."""
    return {
        "b_t_km": 0.0,
        "b_r_km": 0.0,
        "b_magnitude_km": round(miss_distance, 6),
        "miss_distance_km": round(miss_distance, 6),
        "hard_body_radius_km": 0.0,
        "collision_probability_bplane": 0.0,
        "sigma_t_km": 0.0,
        "sigma_r_km": 0.0,
        "ellipses": [],
        "encounter_angle_deg": 0.0,
        "relative_velocity_angle_deg": 0.0,
        "relative_velocity_km_s": 0.0,
    }
