"""
maneuver.py
-----------
Collision avoidance maneuver computation using Clohessy-Wiltshire (CW)
linearized relative motion equations.

Computes required delta-v for radial (R), along-track (S), and
cross-track (W) maneuvers to achieve a target miss distance.
"""

import numpy as np
from datetime import datetime, timezone, timedelta

EARTH_MU = 3.986004418e5  # km³/s²
EARTH_RADIUS_KM = 6371.0


def compute_avoidance_maneuvers(
    r_primary: np.ndarray,        # [3] ECI position at TCA (km)
    v_primary: np.ndarray,        # [3] ECI velocity at TCA (km/s)
    r_secondary: np.ndarray,      # [3] ECI position at TCA (km)
    v_secondary: np.ndarray,      # [3] ECI velocity at TCA (km/s)
    tca_timestamp: float,         # Unix timestamp of TCA
    current_timestamp: float,     # current Unix timestamp
    target_miss_km: float = 5.0,
) -> dict:
    """
    Compute avoidance maneuver options for R, S, W directions.

    Uses Clohessy-Wiltshire linearized relative motion equations
    to estimate delta-v required in each direction.
    """
    r_miss = r_secondary - r_primary
    current_miss_km = float(np.linalg.norm(r_miss))
    dt_seconds = max(tca_timestamp - current_timestamp, 60.0)
    dt_hours = dt_seconds / 3600.0

    # Orbital parameters of primary
    r_mag = np.linalg.norm(r_primary)
    if r_mag < EARTH_RADIUS_KM:
        r_mag = EARTH_RADIUS_KM + 400  # fallback to ~LEO
    a = r_mag  # approximate semi-major axis (circular orbit assumption)
    n = np.sqrt(EARTH_MU / a**3)  # mean motion (rad/s)
    period_s = 2 * np.pi / n
    period_h = period_s / 3600.0

    # RSW frame unit vectors
    r_hat = r_primary / (np.linalg.norm(r_primary) + 1e-10)
    h_vec = np.cross(r_primary, v_primary)
    w_hat = h_vec / (np.linalg.norm(h_vec) + 1e-10)
    s_hat = np.cross(w_hat, r_hat)

    n_dt = n * dt_seconds
    sin_ndt = np.sin(n_dt)
    cos_ndt = np.cos(n_dt)

    maneuvers = []

    # --- Along-track (S) maneuver ---
    # CW: δx_radial ≈ (2/n)(1 - cos(n·dt)) · δv_S
    # CW: δx_along  ≈ (2·sin(n·dt)/n - 3·dt) · δv_S  (simplified)
    # For displacement at TCA: need target_miss_km displacement
    cw_s_factor = abs(2.0 * (1.0 - cos_ndt) / (n + 1e-15))
    if cw_s_factor > 1e-6:
        dv_s = target_miss_km / cw_s_factor  # km/s
    else:
        dv_s = target_miss_km / (dt_seconds + 1e-10)  # fallback

    dv_s_ms = round(float(abs(dv_s) * 1000.0), 4)
    maneuvers.append({
        "direction": "along-track",
        "direction_label": "S (in-plane, along velocity)",
        "delta_v_m_s": dv_s_ms,
        "delta_v_components": {"R": 0.0, "S": dv_s_ms, "W": 0.0},
        "post_maneuver_miss_km": round(float(target_miss_km), 3),
        "fuel_cost_relative": 0.0,
        "efficiency": "high",
        "recommended": False,
    })

    # --- Radial (R) maneuver ---
    cw_r_factor = abs(sin_ndt / (n + 1e-15))
    if cw_r_factor > 1e-6:
        dv_r = target_miss_km / cw_r_factor
    else:
        dv_r = target_miss_km / (dt_seconds + 1e-10)

    dv_r_ms = round(float(abs(dv_r) * 1000.0), 4)
    maneuvers.append({
        "direction": "radial",
        "direction_label": "R (radial, away from Earth)",
        "delta_v_m_s": dv_r_ms,
        "delta_v_components": {"R": dv_r_ms, "S": 0.0, "W": 0.0},
        "post_maneuver_miss_km": round(float(target_miss_km), 3),
        "fuel_cost_relative": 0.0,
        "efficiency": "medium",
        "recommended": False,
    })

    # --- Cross-track (W) maneuver ---
    cw_w_factor = abs(sin_ndt / (n + 1e-15))
    if cw_w_factor > 1e-6:
        dv_w = target_miss_km / cw_w_factor
    else:
        dv_w = target_miss_km / (dt_seconds + 1e-10)

    dv_w_ms = round(float(abs(dv_w) * 1000.0), 4)
    maneuvers.append({
        "direction": "cross-track",
        "direction_label": "W (out-of-plane, normal)",
        "delta_v_m_s": dv_w_ms,
        "delta_v_components": {"R": 0.0, "S": 0.0, "W": dv_w_ms},
        "post_maneuver_miss_km": round(float(target_miss_km), 3),
        "fuel_cost_relative": 0.0,
        "efficiency": "low",
        "recommended": False,
    })

    # Compute relative fuel costs and mark recommendation
    dvs = [m["delta_v_m_s"] for m in maneuvers]
    max_dv = max(dvs) if max(dvs) > 0 else 1.0
    for m in maneuvers:
        m["fuel_cost_relative"] = round(float(m["delta_v_m_s"] / max_dv), 4)

    # Recommend lowest delta-v option
    min_idx = int(np.argmin(dvs))
    maneuvers[min_idx]["recommended"] = True

    # Maneuver windows: every half-orbit before TCA
    windows = _compute_maneuver_windows(
        current_timestamp, tca_timestamp, period_s
    )

    # Overall recommendation
    if current_miss_km >= target_miss_km:
        action = "MONITOR"
    elif dt_hours < 2.0:
        action = "EXECUTE_MANEUVER"
    else:
        action = "PLAN_MANEUVER"

    return {
        "time_to_tca_hours": round(float(dt_hours), 2),
        "current_miss_distance_km": round(float(current_miss_km), 3),
        "target_miss_distance_km": round(float(target_miss_km), 3),
        "orbital_period_hours": round(float(period_h), 2),
        "mean_motion_rad_s": round(float(n), 8),
        "maneuvers": maneuvers,
        "maneuver_windows": windows,
        "recommended_action": action,
    }


def _compute_maneuver_windows(
    current_ts: float,
    tca_ts: float,
    period_s: float,
) -> list[dict]:
    """Generate maneuver windows at half-orbit intervals before TCA."""
    windows = []
    half_period = period_s / 2.0
    dt = tca_ts - current_ts

    # Generate windows starting from ~1 orbit before TCA back to current time
    t = tca_ts - half_period
    window_idx = 0
    while t > current_ts and window_idx < 8:
        window_start = t - half_period * 0.25
        window_end = t + half_period * 0.25
        optimal = t

        # Classify window
        time_before_tca_h = (tca_ts - t) / 3600.0
        if time_before_tca_h < period_s / 3600.0:
            wtype = "last-orbit"
        elif time_before_tca_h < 2 * period_s / 3600.0:
            wtype = "penultimate-orbit"
        else:
            wtype = "early-warning"

        windows.append({
            "window_start_utc": _ts_to_utc(max(window_start, current_ts)),
            "window_end_utc": _ts_to_utc(min(window_end, tca_ts)),
            "optimal_time_utc": _ts_to_utc(optimal),
            "hours_before_tca": round(float(time_before_tca_h), 2),
            "type": wtype,
        })

        t -= half_period
        window_idx += 1

    windows.reverse()  # chronological order
    return windows


def _ts_to_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
