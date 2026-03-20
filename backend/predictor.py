"""
predictor.py
------------
Inference pipeline for the Space Debris Collision Risk Transformer.
Loads the trained model and catalog, then for a given satellite:
  1. Looks up its TLE data
  2. Propagates its trajectory and all debris objects
  3. Computes features for all satellite-debris pairs
  4. Runs batch inference through the transformer
  5. Returns ranked list of top-N debris threats

Used by the FastAPI backend on each /predict request.
"""

import sys
import logging
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.models.transformer import build_model
from backend.utils.sgp4_propagator import (
    get_propagation_times, propagate_object, compute_miss_distance, find_tca
)
from backend.utils.feature_engineering import (
    compute_conjunction_features, get_feature_names, features_to_array,
    compute_collision_probability,
)
from backend.utils.bplane import compute_bplane_parameters
from backend.utils.maneuver import compute_avoidance_maneuvers
from backend.utils.interpret import (
    extract_attention_weights, compute_feature_importance, ensemble_predictions
)

log = logging.getLogger(__name__)

RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
RISK_COLORS = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}


class SatellitePredictor:
    """
    Main inference class. Loaded once at server startup.
    Holds the model, satellite catalog, and normalization parameters.
    """

    def __init__(
        self,
        model_path: str | Path,
        catalog_path: str | Path,
        norm_path: Optional[str | Path] = None,
        device: str = "auto",
        batch_size: int = 512,
    ):
        self.model_path   = Path(model_path)
        self.catalog_path = Path(catalog_path)
        self.batch_size   = batch_size
        self.loaded       = False
        self.ensemble_paths = []

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Normalization parameters (loaded from file or defaults)
        self.feature_names = get_feature_names()
        self.means = np.zeros(len(self.feature_names), dtype=np.float32)
        self.stds  = np.ones(len(self.feature_names), dtype=np.float32)

        self._load(norm_path)

    def _load(self, norm_path: Optional[str | Path]) -> None:
        """Load model, catalog, and normalization parameters."""
        log.info(f"Loading predictor on {self.device}...")

        # --- Model ---
        if not self.model_path.exists():
            log.warning(f"Model file not found: {self.model_path}. Using untrained model.")
            self.model = build_model()
        else:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            # Support both old ("config") and new ("model_cfg") checkpoint formats
            config = checkpoint.get("config", None)
            if config is None and "model_cfg" in checkpoint:
                config = {"model": checkpoint["model_cfg"]}
            self.model = build_model(config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            log.info(f"  Loaded model from epoch {checkpoint.get('epoch', '?')}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # --- Catalog ---
        if not self.catalog_path.exists():
            log.warning(f"Catalog not found: {self.catalog_path}. Using empty catalog.")
            self.catalog = pd.DataFrame(columns=["norad_id", "name", "line1", "line2"])
        else:
            self.catalog = pd.read_csv(self.catalog_path)
            log.info(f"  Loaded catalog: {len(self.catalog):,} objects")

        # Build lookup indices
        self.name_to_idx  = {str(r["name"]).upper(): i for i, r in self.catalog.iterrows()}
        self.norad_to_idx = {int(r["norad_id"]): i for i, r in self.catalog.iterrows()}

        # --- Normalization ---
        if norm_path and Path(norm_path).exists():
            npz = np.load(norm_path, allow_pickle=True)
            self.means = npz["means"].astype(np.float32)
            self.stds  = npz["stds"].astype(np.float32)
            log.info(f"  Loaded normalization from {norm_path}")

        # Ensemble checkpoints for interpretability
        models_dir = self.model_path.parent
        self.ensemble_paths = sorted(models_dir.glob("ckpt_ep*.pth"))
        if self.model_path.exists():
            self.ensemble_paths.insert(0, self.model_path)
        log.info(f"  Ensemble: {len(self.ensemble_paths)} checkpoints available")

        self.loaded = True
        log.info(f"✅ Predictor ready ({len(self.catalog):,} objects in catalog)")

    def find_satellite(self, satellite_name: str = None, norad_id: int = None) -> Optional[pd.Series]:
        """Look up a satellite by name or NORAD ID."""
        if norad_id is not None:
            idx = self.norad_to_idx.get(int(norad_id))
        elif satellite_name is not None:
            # Try exact match first, then partial
            upper = satellite_name.upper().strip()
            idx = self.name_to_idx.get(upper)
            if idx is None:
                # Partial match
                for name, i in self.name_to_idx.items():
                    if upper in name:
                        idx = i
                        break
        else:
            return None

        return self.catalog.iloc[idx] if idx is not None else None

    def _propagate_pair(
        self,
        sat_row: pd.Series,
        debris_rows: pd.DataFrame,
        horizon_hours: int = 168,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Propagate satellite and all debris objects.
        Returns:
            sat_positions:     [T, 6]
            debris_positions:  [N, T, 6]  (None rows for failed objects)
            times:             [T] Unix timestamps
        """
        epoch = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        _, times_jd = get_propagation_times(epoch, horizon_hours)
        times_unix  = np.array([epoch.timestamp() + h * 3600 for h in range(horizon_hours + 1)])

        # Primary satellite
        sat_pos = propagate_object(str(sat_row["line1"]), str(sat_row["line2"]), times_jd)

        # Debris objects (batch)
        debris_positions = []
        for _, row in debris_rows.iterrows():
            pos = propagate_object(str(row["line1"]), str(row["line2"]), times_jd)
            debris_positions.append(pos)

        return sat_pos, debris_positions, times_unix

    def _compute_features_batch(
        self,
        sat_pos: np.ndarray,          # [T, 6]
        debris_positions: list,        # [N] each [T, 6] or None
        debris_rows: pd.DataFrame,
        times_unix: np.ndarray,        # [T]
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Compute features for all valid satellite-debris pairs.
        Returns:
            X:           [M, 30] feature matrix (normalized)
            X_traj:      [M, T, 22] trajectory tensors for transformer
            pair_info:   list of dicts with metadata per pair
        """
        T = sat_pos.shape[0]
        n_feat = len(self.feature_names)
        n_model_feat = 22

        feature_rows = []
        pair_info    = []
        valid_debris_rows = []

        sat_xyz = sat_pos[:, :3].astype(np.float64)
        sat_vel = sat_pos[:, 3:6].astype(np.float64)

        for i, (debris_pos, (_, debris_row)) in enumerate(zip(debris_positions, debris_rows.iterrows())):
            if debris_pos is None:
                continue

            debris_xyz = debris_pos[:, :3].astype(np.float64)
            distances  = np.linalg.norm(sat_xyz - debris_xyz, axis=1)  # [T]
            min_dist   = float(distances.min())

            # Skip very distant objects for efficiency
            if min_dist > 500:
                continue

            tca_dist, tca_ts, tca_idx = find_tca(distances, times_unix)
            time_to_tca_h = (tca_ts - times_unix[0]) / 3600.0

            feats = compute_conjunction_features(
                primary_pos_tca=sat_xyz[tca_idx],
                primary_vel_tca=sat_vel[tca_idx],
                secondary_pos_tca=debris_pos[tca_idx, :3].astype(np.float64),
                secondary_vel_tca=debris_pos[tca_idx, 3:6].astype(np.float64),
                time_to_tca_hours=time_to_tca_h,
            )
            feat_arr = features_to_array(feats, self.feature_names)
            feature_rows.append(feat_arr)

            p_c = compute_collision_probability(
                miss_distance_km=tca_dist,
                relative_velocity_km_s=feats["relative_velocity"],
                combined_area_m2=feats["combined_cross_section"],
            )

            pair_info.append({
                "debris_name":    str(debris_row["name"]),
                "debris_norad_id": int(debris_row["norad_id"]),
                "min_distance_km": round(min_dist, 3),
                "tca_timestamp":   tca_ts,
                "tca_utc":         datetime.fromtimestamp(tca_ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "relative_velocity_km_s": round(feats["relative_velocity"], 3),
                "physics_probability": p_c,
                "feat_idx": len(feature_rows) - 1,
            })

        if not feature_rows:
            return np.empty((0, n_feat)), np.empty((0, T, n_model_feat)), []

        X = np.stack(feature_rows, axis=0).astype(np.float32)  # [M, 30]
        X_norm = (X - self.means) / self.stds                  # normalize

        # Build trajectory tensors [M, T, 22] by tiling static features
        X_traj = np.tile(X_norm[:, :n_model_feat, np.newaxis], (1, 1, T)).transpose(0, 2, 1)
        return X_norm, X_traj, pair_info

    @torch.no_grad()
    def _run_model_inference(self, X_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run batch inference through the transformer.
        Returns: probabilities [M, 3], uncertainty [M], epistemic [M]
        """
        M, T, F = X_traj.shape
        all_probs, all_unc, all_epi = [], [], []

        for start in range(0, M, self.batch_size):
            chunk = X_traj[start:start + self.batch_size]
            x_tensor = torch.tensor(chunk, dtype=torch.float32, device=self.device)

            with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                evidence, alpha, uncertainty, prob = self.model(x_tensor)

            S = alpha.sum(dim=-1)
            epistemic = ((alpha * (S.unsqueeze(-1) - alpha)) / (S.unsqueeze(-1) ** 2 * (S.unsqueeze(-1) + 1))).sum(-1)

            all_probs.append(prob.cpu().numpy())
            all_unc.append(uncertainty.cpu().numpy())
            all_epi.append(epistemic.cpu().numpy())

        probs     = np.vstack(all_probs)    # [M, 3]
        unc       = np.concatenate(all_unc) # [M]
        epistemic = np.concatenate(all_epi) # [M]
        return probs, unc, epistemic

    def predict(
        self,
        satellite_name: str = None,
        norad_id: int = None,
        top_n: int = 10,
    ) -> dict:
        """
        Main prediction method.
        Returns top-N debris threats with risk scores for a given satellite.
        """
        start_time = time.time()

        if not self.loaded:
            return {"error": "Predictor not loaded"}

        # --- Find satellite ---
        sat_row = self.find_satellite(satellite_name, norad_id)
        if sat_row is None:
            return {
                "error": f"Satellite not found: name='{satellite_name}', norad_id={norad_id}"
            }

        log.info(f"Predicting for: {sat_row['name']} (NORAD {sat_row['norad_id']})")

        # --- Propagate ---
        t_prop_start = time.time()
        # Use all non-primary objects as potential debris
        debris_rows = self.catalog[self.catalog["norad_id"] != int(sat_row["norad_id"])]
        sat_pos, debris_positions, times_unix = self._propagate_pair(sat_row, debris_rows)
        t_prop = time.time() - t_prop_start

        if sat_pos is None:
            return {"error": f"Could not propagate satellite {sat_row['name']}"}

        # --- Features ---
        X_norm, X_traj, pair_info = self._compute_features_batch(
            sat_pos, debris_positions, debris_rows, times_unix
        )
        n_pairs = len(pair_info)
        log.info(f"Found {n_pairs:,} candidate pairs (propagation: {t_prop:.1f}s)")

        if n_pairs == 0:
            return {
                "satellite": self._format_satellite(sat_row),
                "threats": [],
                "n_candidates_analyzed": 0,
                "inference_time_s": round(time.time() - start_time, 2),
                "message": "No close approaches found within 500km.",
            }

        # --- Model inference ---
        t_infer_start = time.time()
        probs, uncertainty, epistemic = self._run_model_inference(X_traj)
        t_infer = time.time() - t_infer_start

        # --- Rank and format results ---
        # Score = P(HIGH) * 0.6 + P(MEDIUM) * 0.3 + (1 - min_dist_norm) * 0.1
        high_probs = probs[:, 2]
        med_probs  = probs[:, 1]
        low_probs  = probs[:, 0]

        distances = np.array([p["min_distance_km"] for p in pair_info])
        dist_norm = np.clip(distances / 10.0, 0, 1)  # normalize to [0, 1]
        scores = high_probs * 0.6 + med_probs * 0.3 + (1 - dist_norm) * 0.1

        ranked_idx = np.argsort(scores)[::-1][:top_n]

        threats = []
        for i, idx in enumerate(ranked_idx):
            info = pair_info[idx]
            p = probs[idx]
            risk_class = int(np.argmax(p))
            risk_label = RISK_LABELS[risk_class]
            collision_prob = max(float(p[2] * 0.1), info["physics_probability"])  # blend both estimates

            threats.append({
                "rank": i + 1,
                "debris_name": info["debris_name"],
                "debris_norad_id": info["debris_norad_id"],
                "collision_probability": round(collision_prob * 100, 6),  # as %
                "collision_probability_pct": f"{collision_prob * 100:.4f}%",
                "uncertainty_pct": round(float(uncertainty[idx]) * 100, 3),
                "uncertainty_range": f"±{float(uncertainty[idx]) * 100:.2f}%",
                "risk_level": risk_label,
                "risk_color": RISK_COLORS[risk_label],
                "probability_low": round(float(p[0]), 5),
                "probability_medium": round(float(p[1]), 5),
                "probability_high": round(float(p[2]), 5),
                "epistemic_uncertainty": round(float(epistemic[idx]), 6),
                "miss_distance_km": info["min_distance_km"],
                "tca_utc": info["tca_utc"],
                "tca_timestamp": info["tca_timestamp"],
                "relative_velocity_km_s": info["relative_velocity_km_s"],
            })

        elapsed = time.time() - start_time
        return {
            "satellite": self._format_satellite(sat_row),
            "threats": threats,
            "n_candidates_analyzed": n_pairs,
            "propagation_time_s": round(t_prop, 2),
            "inference_time_s": round(t_infer, 2),
            "total_time_s": round(elapsed, 2),
        }

    def _format_satellite(self, row: pd.Series) -> dict:
        return {
            "name": str(row["name"]),
            "norad_id": int(row["norad_id"]),
            "inclination_deg": round(float(row.get("inclination_deg", 0)), 3),
            "altitude_km": round(float(row.get("altitude_km", 0)), 1),
            "eccentricity": round(float(row.get("eccentricity", 0)), 6),
            "source": str(row.get("source", "unknown")),
        }

    def predict_detailed(
        self,
        satellite_name: str = None,
        norad_id: int = None,
        top_n: int = 10,
        n_monte_carlo: int = 50,
    ) -> dict:
        """
        Extended prediction with temporal risk, B-plane geometry, and Monte Carlo analysis.
        """
        base = self.predict(satellite_name=satellite_name, norad_id=norad_id, top_n=top_n)
        if "error" in base:
            return base

        sat_row = self.find_satellite(satellite_name, norad_id)
        debris_rows = self.catalog[self.catalog["norad_id"] != int(sat_row["norad_id"])]

        epoch = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        from backend.utils.sgp4_propagator import get_propagation_times
        _, times_jd = get_propagation_times(epoch, 168)
        times_unix = np.array([epoch.timestamp() + h * 3600 for h in range(169)])

        sat_pos = propagate_object(str(sat_row["line1"]), str(sat_row["line2"]), times_jd)
        if sat_pos is None:
            return base

        # Enrich each threat with B-plane and temporal risk
        for threat in base["threats"]:
            deb_row = self.find_satellite(norad_id=threat["debris_norad_id"])
            if deb_row is None:
                threat["bplane"] = None
                threat["temporal_risk"] = []
                continue

            deb_pos = propagate_object(str(deb_row["line1"]), str(deb_row["line2"]), times_jd)
            if deb_pos is None:
                threat["bplane"] = None
                threat["temporal_risk"] = []
                continue

            # Find TCA index
            distances = np.linalg.norm(sat_pos[:, :3] - deb_pos[:, :3], axis=1)
            tca_idx = int(np.argmin(distances))

            # B-plane at TCA
            threat["bplane"] = compute_bplane_parameters(
                r_primary=sat_pos[tca_idx, :3],
                v_primary=sat_pos[tca_idx, 3:6],
                r_secondary=deb_pos[tca_idx, :3],
                v_secondary=deb_pos[tca_idx, 3:6],
            )

            # Temporal risk: Chan's Pc at each timestep
            temporal = []
            for t_idx in range(len(times_unix)):
                d = float(distances[t_idx])
                v_rel = float(np.linalg.norm(
                    deb_pos[t_idx, 3:6] - sat_pos[t_idx, 3:6]
                ))
                pc = compute_collision_probability(d, v_rel)
                temporal.append({
                    "hour": t_idx,
                    "timestamp_utc": datetime.fromtimestamp(
                        times_unix[t_idx], tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "miss_distance_km": round(d, 4),
                    "collision_probability": pc,
                })
            threat["temporal_risk"] = temporal

        # Monte Carlo analysis on the top threat
        mc_results = []
        if base["threats"]:
            top = base["threats"][0]
            deb_row = self.find_satellite(norad_id=top["debris_norad_id"])
            if deb_row is not None:
                deb_pos_base = propagate_object(str(deb_row["line1"]), str(deb_row["line2"]), times_jd)
                if deb_pos_base is not None:
                    distances_base = np.linalg.norm(sat_pos[:, :3] - deb_pos_base[:, :3], axis=1)
                    tca_idx = int(np.argmin(distances_base))
                    rng = np.random.default_rng(42)

                    for _ in range(n_monte_carlo):
                        # Perturb positions: σ=0.1km, velocities: σ=0.00001km/s
                        dr = rng.normal(0, 0.1, 3)
                        dv = rng.normal(0, 0.00001, 3)
                        r_pert = sat_pos[tca_idx, :3] + dr
                        v_pert = sat_pos[tca_idx, 3:6] + dv
                        d_mc = float(np.linalg.norm(deb_pos_base[tca_idx, :3] - r_pert))
                        v_rel_mc = float(np.linalg.norm(deb_pos_base[tca_idx, 3:6] - v_pert))
                        pc_mc = compute_collision_probability(d_mc, v_rel_mc)
                        mc_results.append(pc_mc)

        mc_arr = np.array(mc_results) if mc_results else np.array([0.0])
        base["monte_carlo"] = {
            "n_samples": len(mc_results),
            "mean_probability": round(float(mc_arr.mean()), 10),
            "std_probability": round(float(mc_arr.std()), 10),
            "p90": round(float(np.percentile(mc_arr, 90)), 10) if len(mc_results) > 1 else 0.0,
            "p99": round(float(np.percentile(mc_arr, 99)), 10) if len(mc_results) > 1 else 0.0,
            "samples": [round(float(v), 12) for v in mc_arr],
        }

        return base

    def compute_maneuver(
        self,
        satellite_name: str = None,
        norad_id: int = None,
        debris_norad_id: int = None,
        target_miss_km: float = 5.0,
    ) -> dict:
        """Compute avoidance maneuvers for a specific satellite-debris pair."""
        sat_row = self.find_satellite(satellite_name, norad_id)
        if sat_row is None:
            return {"error": f"Satellite not found"}

        deb_row = self.find_satellite(norad_id=debris_norad_id)
        if deb_row is None:
            return {"error": f"Debris object {debris_norad_id} not found"}

        epoch = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        _, times_jd = get_propagation_times(epoch, 168)
        times_unix = np.array([epoch.timestamp() + h * 3600 for h in range(169)])

        sat_pos = propagate_object(str(sat_row["line1"]), str(sat_row["line2"]), times_jd)
        deb_pos = propagate_object(str(deb_row["line1"]), str(deb_row["line2"]), times_jd)

        if sat_pos is None or deb_pos is None:
            return {"error": "Propagation failed"}

        distances = np.linalg.norm(sat_pos[:, :3] - deb_pos[:, :3], axis=1)
        tca_idx = int(np.argmin(distances))

        result = compute_avoidance_maneuvers(
            r_primary=sat_pos[tca_idx, :3],
            v_primary=sat_pos[tca_idx, 3:6],
            r_secondary=deb_pos[tca_idx, :3],
            v_secondary=deb_pos[tca_idx, 3:6],
            tca_timestamp=times_unix[tca_idx],
            current_timestamp=times_unix[0],
            target_miss_km=target_miss_km,
        )

        result["satellite"] = self._format_satellite(sat_row)
        result["debris"] = {
            "name": str(deb_row["name"]),
            "norad_id": int(deb_row["norad_id"]),
        }
        return result

    def interpret(
        self,
        satellite_name: str = None,
        norad_id: int = None,
        debris_norad_id: int = None,
    ) -> dict:
        """Run model interpretability analysis for a specific conjunction."""
        sat_row = self.find_satellite(satellite_name, norad_id)
        if sat_row is None:
            return {"error": "Satellite not found"}

        deb_row = self.find_satellite(norad_id=debris_norad_id)
        if deb_row is None:
            return {"error": f"Debris object {debris_norad_id} not found"}

        epoch = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        _, times_jd = get_propagation_times(epoch, 168)
        times_unix = np.array([epoch.timestamp() + h * 3600 for h in range(169)])

        sat_pos = propagate_object(str(sat_row["line1"]), str(sat_row["line2"]), times_jd)
        deb_pos = propagate_object(str(deb_row["line1"]), str(deb_row["line2"]), times_jd)

        if sat_pos is None or deb_pos is None:
            return {"error": "Propagation failed"}

        # Build trajectory tensor for this pair
        distances = np.linalg.norm(sat_pos[:, :3] - deb_pos[:, :3], axis=1)
        tca_idx = int(np.argmin(distances))
        tca_ts = times_unix[tca_idx]
        time_to_tca_h = (tca_ts - times_unix[0]) / 3600.0

        feats = compute_conjunction_features(
            primary_pos_tca=sat_pos[tca_idx, :3],
            primary_vel_tca=sat_pos[tca_idx, 3:6],
            secondary_pos_tca=deb_pos[tca_idx, :3],
            secondary_vel_tca=deb_pos[tca_idx, 3:6],
            time_to_tca_hours=time_to_tca_h,
        )
        feat_arr = features_to_array(feats, self.feature_names)
        feat_norm = (feat_arr - self.means) / self.stds

        n_model_feat = 22
        T = sat_pos.shape[0]
        x_single = np.tile(feat_norm[:n_model_feat, np.newaxis], (1, T)).T  # [T, 22]
        x_tensor = torch.tensor(x_single[np.newaxis], dtype=torch.float32, device=self.device)

        # Attention weights
        attention = extract_attention_weights(self.model, x_tensor, self.device)

        # Feature importance
        importance = compute_feature_importance(
            self.model, x_tensor, self.feature_names[:n_model_feat], self.device
        )

        # Ensemble predictions
        ensemble = {}
        if len(self.ensemble_paths) > 1:
            ensemble = ensemble_predictions(
                model_paths=self.ensemble_paths,
                x_tensor=x_tensor,
                device=self.device,
                build_model_fn=build_model,
            )
        else:
            # Single model fallback
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                    _, alpha, uncertainty, prob = self.model(x_tensor)
            p = prob[0].cpu().numpy()
            ensemble = {
                "individual_predictions": [{
                    "checkpoint": self.model_path.name,
                    "probabilities": [round(float(v), 6) for v in p],
                    "uncertainty": round(float(uncertainty[0].cpu()), 6),
                    "predicted_class": RISK_LABELS[int(np.argmax(p))],
                }],
                "mean_probabilities": [round(float(v), 6) for v in p],
                "std_probabilities": [0.0, 0.0, 0.0],
                "agreement_score": 1.0,
                "consensus_class": RISK_LABELS[int(np.argmax(p))],
                "n_models": 1,
            }

        return {
            "satellite": self._format_satellite(sat_row),
            "debris": {"name": str(deb_row["name"]), "norad_id": int(deb_row["norad_id"])},
            "attention": attention,
            "feature_importance": importance,
            "ensemble": ensemble,
        }

    def list_satellites(self, limit: int = 1000, search: str = None, include_tle: bool = False) -> list[dict]:
        """Return a list of all trackable satellites."""
        df = self.catalog
        if search:
            mask = df["name"].str.upper().str.contains(search.upper(), na=False)
            df = df[mask]
        df = df.head(limit)
        
        results = []
        for _, r in df.iterrows():
            item = {
                "norad_id": int(r["norad_id"]),
                "name": str(r["name"]),
                "altitude_km": round(float(r.get("altitude_km", 0)), 1),
                "inclination_deg": round(float(r.get("inclination_deg", 0)), 2),
                "source": str(r.get("source", "unknown")),
            }
            if include_tle:
                item["line1"] = str(r.get("line1", ""))
                item["line2"] = str(r.get("line2", ""))
            results.append(item)
        return results
