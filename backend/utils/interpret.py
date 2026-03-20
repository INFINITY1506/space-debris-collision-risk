"""
interpret.py
------------
Model interpretability utilities for the collision risk transformer.

Provides:
  - Attention weight extraction (cross-attention + self-attention)
  - Gradient-based feature importance
  - Ensemble predictions from multiple checkpoints
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
import logging

log = logging.getLogger(__name__)


def extract_attention_weights(
    model: "CollisionRiskTransformer",
    x_tensor: torch.Tensor,  # [1, T, 22]
    device: torch.device,
) -> dict:
    """
    Extract attention weights from the transformer.

    Registers forward hooks on the PairwiseInteractionModule to capture
    cross-attention weights (which timesteps the model focuses on).

    Returns dict with cross_attention [T] and layer info.
    """
    model.eval()
    x_tensor = x_tensor.to(device)

    cross_attn_weights = []

    # Hook into the pairwise interaction module
    def pairwise_hook(module, input, output):
        # Re-compute attention weights (same as forward but capture them)
        sequence_repr, global_context = input
        B, T, D = sequence_repr.shape
        q = module.query_proj(global_context).unsqueeze(1)
        k = module.key_proj(sequence_repr)
        attn = torch.bmm(q, k.transpose(1, 2)) / module.scale
        attn = F.softmax(attn, dim=-1)  # [B, 1, T]
        cross_attn_weights.append(attn.detach().cpu().numpy())

    handle = model.pairwise.register_forward_hook(pairwise_hook)

    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
            model(x_tensor)

    handle.remove()

    # Process cross-attention
    if cross_attn_weights:
        ca = cross_attn_weights[0][0, 0, :]  # [T] - first batch, squeeze query dim
    else:
        T = x_tensor.shape[1]
        ca = np.ones(T) / T

    # Find peak attention timesteps
    top_k = min(10, len(ca))
    top_indices = np.argsort(ca)[::-1][:top_k]
    peak_timesteps = [{"hour": int(idx), "weight": round(float(ca[idx]), 6)} for idx in top_indices]

    return {
        "cross_attention": [round(float(v), 6) for v in ca],
        "peak_timesteps": peak_timesteps,
        "attention_entropy": round(float(-np.sum(ca * np.log(ca + 1e-10))), 4),
    }


def compute_feature_importance(
    model: "CollisionRiskTransformer",
    x_tensor: torch.Tensor,  # [1, T, 22]
    feature_names: list[str],
    device: torch.device,
) -> list[dict]:
    """
    Compute gradient-based feature importance.

    Uses ∂P(HIGH)/∂input to determine which features most influence
    the high-risk probability prediction.
    """
    model.eval()
    x = x_tensor.clone().detach().to(device).requires_grad_(True)

    with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
        _, _, _, prob = model(x)

    # Gradient of P(HIGH) w.r.t. input
    prob[:, 2].backward()
    grad = x.grad.detach().cpu().numpy()  # [1, T, 22]

    # Feature importance = mean absolute gradient across timesteps
    importance = np.mean(np.abs(grad[0]), axis=0)  # [22]

    # Use first 22 feature names (model input dimension)
    n_model_features = min(len(feature_names), importance.shape[0])
    names = feature_names[:n_model_features]

    # Normalize to [0, 1]
    imp_max = importance.max()
    if imp_max > 0:
        importance_norm = importance / imp_max
    else:
        importance_norm = importance

    results = []
    for i in range(n_model_features):
        results.append({
            "feature": names[i],
            "importance": round(float(importance_norm[i]), 6),
            "raw_gradient": round(float(importance[i]), 8),
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results


def ensemble_predictions(
    model_paths: list[Path],
    x_tensor: torch.Tensor,  # [1, T, 22]
    device: torch.device,
    build_model_fn: callable,
    config: Optional[dict] = None,
) -> dict:
    """
    Run ensemble predictions from multiple checkpoints.

    Loads each checkpoint, runs inference, and computes
    mean/std across all models. Provides agreement score.
    """
    all_probs = []
    all_uncertainty = []
    individual = []

    for path in model_paths:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            ckpt_config = checkpoint.get("config", None)
            if ckpt_config is None and "model_cfg" in checkpoint:
                ckpt_config = {"model": checkpoint["model_cfg"]}
            if ckpt_config is None:
                ckpt_config = config

            model = build_model_fn(ckpt_config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                    evidence, alpha, uncertainty, prob = model(x_tensor.to(device))

            p = prob[0].cpu().numpy()
            u = float(uncertainty[0].cpu())
            all_probs.append(p)
            all_uncertainty.append(u)

            epoch = checkpoint.get("epoch", "?")
            individual.append({
                "checkpoint": path.name,
                "epoch": epoch,
                "probabilities": [round(float(v), 6) for v in p],
                "uncertainty": round(u, 6),
                "predicted_class": ["LOW", "MEDIUM", "HIGH"][int(np.argmax(p))],
            })

            # Free GPU memory
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            log.warning(f"Failed to load checkpoint {path.name}: {e}")
            continue

    if not all_probs:
        return {
            "individual_predictions": [],
            "mean_probabilities": [0.333, 0.333, 0.334],
            "std_probabilities": [0.0, 0.0, 0.0],
            "agreement_score": 0.0,
            "n_models": 0,
        }

    probs_arr = np.array(all_probs)  # [N_models, 3]
    mean_prob = np.mean(probs_arr, axis=0)
    std_prob = np.std(probs_arr, axis=0)

    # Agreement: 1 - mean std (higher = more agreement)
    agreement = float(1.0 - np.mean(std_prob))

    # Prediction consensus
    predicted_classes = [np.argmax(p) for p in all_probs]
    consensus_class = int(np.bincount(predicted_classes).argmax())
    consensus_pct = float(np.mean([c == consensus_class for c in predicted_classes]) * 100)

    return {
        "individual_predictions": individual,
        "mean_probabilities": [round(float(v), 6) for v in mean_prob],
        "std_probabilities": [round(float(v), 6) for v in std_prob],
        "mean_uncertainty": round(float(np.mean(all_uncertainty)), 6),
        "agreement_score": round(agreement, 4),
        "consensus_class": ["LOW", "MEDIUM", "HIGH"][consensus_class],
        "consensus_pct": round(consensus_pct, 1),
        "n_models": len(all_probs),
    }
