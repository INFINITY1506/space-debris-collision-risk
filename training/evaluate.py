"""
evaluate.py
-----------
Model evaluation on held-out test set.
Generates classification metrics, calibration plots, confusion matrix,
ROC curves, uncertainty decomposition, and attention visualization.

Usage:
    python training/evaluate.py
    python training/evaluate.py --model data/models/best_model.pth
"""

import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, brier_score_loss,
    classification_report, roc_curve, auc,
)
from sklearn.calibration import calibration_curve

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.models.transformer import build_model
from training.train import ConjunctionDataset, collate_fn, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RISK_LABELS = ["LOW", "MEDIUM", "HIGH"]
COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]
DOCS_DIR = Path("docs")


def load_model_and_config(model_path: str | Path, config: dict) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    # Use config from checkpoint if available
    ckpt_config = checkpoint.get("config", config)
    model = build_model(ckpt_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')} — val AUC: {checkpoint.get('val_metrics', {}).get('auc', '?')}")
    return model, ckpt_config


@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """Run full inference on a DataLoader and collect all outputs."""
    all_labels, all_preds, all_probs = [], [], []
    all_uncertainty, all_epistemic, all_aleatoric = [], [], []

    for batch in loader:
        x      = batch["x"].to(device)
        labels = batch["label"].to(device)
        mask   = batch["mask"].to(device)

        with autocast(enabled=device.type == "cuda"):
            evidence, alpha, uncertainty, prob = model(x, src_key_padding_mask=mask)

        S = alpha.sum(dim=-1)
        # Epistemic uncertainty: sum of variance
        epistemic = ((alpha * (S.unsqueeze(-1) - alpha)) / (S.unsqueeze(-1) ** 2 * (S.unsqueeze(-1) + 1))).sum(-1)
        # Aleatoric: entropy
        aleatoric = -(prob * torch.log(prob.clamp(1e-8))).sum(-1)

        preds = prob.argmax(dim=-1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.append(prob.cpu().numpy())
        all_uncertainty.extend(uncertainty.cpu().numpy().tolist())
        all_epistemic.extend(epistemic.cpu().numpy().tolist())
        all_aleatoric.extend(aleatoric.cpu().numpy().tolist())

    return {
        "labels": np.array(all_labels),
        "preds":  np.array(all_preds),
        "probs":  np.vstack(all_probs),         # [N, 3]
        "uncertainty": np.array(all_uncertainty),
        "epistemic":   np.array(all_epistemic),
        "aleatoric":   np.array(all_aleatoric),
    }


def compute_ece(y_true, y_prob, n_bins=10) -> float:
    """Expected Calibration Error."""
    class_probs = y_prob[np.arange(len(y_true)), y_true]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (class_probs >= lo) & (class_probs < hi)
        if mask.sum() == 0:
            continue
        avg_conf = class_probs[mask].mean()
        avg_acc  = (y_true[mask] == np.argmax(y_prob[mask], axis=1) if y_prob.ndim > 1 else 0)
        bin_acc  = (y_true[mask] == y_true[mask]).mean()  # proportion correct
        ece += (mask.sum() / len(y_true)) * abs(avg_conf - avg_conf)  # simplified
    # More robust ECE
    preds = np.argmax(y_prob, axis=1)
    confidences = y_prob.max(axis=1)
    correct = (preds == y_true).astype(float)
    ece_robust = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece_robust += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece_robust)


def plot_confusion_matrix(y_true, y_pred, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=RISK_LABELS, yticklabels=RISK_LABELS, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title("Confusion Matrix — Risk Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_roc_curves(y_true, y_probs, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    auc_scores = []
    for i, (label, color) in enumerate(zip(RISK_LABELS, COLORS)):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auc_score = auc(fpr, tpr)
        auc_scores.append(auc_score)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{label} (AUC = {auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — Macro AUC: {np.mean(auc_scores):.4f}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_reliability_diagram(y_true, y_probs, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (label, color) in enumerate(zip(RISK_LABELS, COLORS)):
        y_bin = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_probs[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, "s-", color=color, lw=2, label=label, markersize=6)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Reliability Diagram (Calibration)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_uncertainty_distribution(results: dict, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Total Uncertainty", "Epistemic Uncertainty", "Aleatoric Uncertainty"]
    keys = ["uncertainty", "epistemic", "aleatoric"]

    for ax, title, key in zip(axes, titles, keys):
        for i, (label, color) in enumerate(zip(RISK_LABELS, COLORS)):
            mask = results["labels"] == i
            ax.hist(results[key][mask], bins=30, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Uncertainty", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Uncertainty Decomposition by Risk Level", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def run_evaluation(model_path: str, config: dict) -> dict:
    """Full evaluation pipeline."""
    DOCS_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    model, cfg = load_model_and_config(model_path, config)
    model = model.to(device)

    test_ds = ConjunctionDataset(config["data"]["test_file"], augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    log.info(f"Running inference on {len(test_ds):,} test samples...")
    results = run_inference(model, test_loader, device)

    y_true = results["labels"]
    y_pred = results["preds"]
    y_prob = results["probs"]

    # ---------- Classification Metrics ----------
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        macro_auc = 0.0

    # Per-class brier scores
    brier_scores = {}
    for i, label in enumerate(RISK_LABELS):
        y_bin = (y_true == i).astype(int)
        brier_scores[label] = round(float(brier_score_loss(y_bin, y_prob[:, i])), 5)

    ece = compute_ece(y_true, y_prob)

    metrics = {
        "accuracy": round(float(acc), 5),
        "precision_macro": round(float(prec), 5),
        "recall_macro": round(float(rec), 5),
        "f1_macro": round(float(f1), 5),
        "auc_roc_macro": round(float(macro_auc), 5),
        "ece": round(float(ece), 5),
        "brier_score": brier_scores,
        "n_test_samples": int(len(y_true)),
        "class_distribution": {l: int((y_true == i).sum()) for i, l in enumerate(RISK_LABELS)},
        "uncertainty": {
            "mean_total":     round(float(results["uncertainty"].mean()), 5),
            "mean_epistemic": round(float(results["epistemic"].mean()), 5),
            "mean_aleatoric": round(float(results["aleatoric"].mean()), 5),
        }
    }

    log.info("\n" + "=" * 60)
    log.info("TEST SET RESULTS")
    log.info("=" * 60)
    log.info(f"  Accuracy:      {acc:.4f}")
    log.info(f"  Precision:     {prec:.4f}")
    log.info(f"  Recall:        {rec:.4f}")
    log.info(f"  F1:            {f1:.4f}")
    log.info(f"  AUC-ROC:       {macro_auc:.4f}")
    log.info(f"  ECE:           {ece:.4f}")
    log.info(f"  Brier scores:  {brier_scores}")
    log.info("=" * 60)
    log.info(f"\n{classification_report(y_true, y_pred, target_names=RISK_LABELS)}")

    # Save metrics
    metrics_path = DOCS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved: {metrics_path}")

    # Generate plots
    plot_confusion_matrix(y_true, y_pred, DOCS_DIR / "confusion_matrix.png")
    plot_roc_curves(y_true, y_prob, DOCS_DIR / "roc_curves.png")
    plot_reliability_diagram(y_true, y_prob, DOCS_DIR / "reliability_diagram.png")
    plot_uncertainty_distribution(results, DOCS_DIR / "uncertainty_distribution.png")

    log.info("\n✅ Evaluation complete. Plots saved to docs/")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="data/models/best_model.pth")
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    if not Path(args.model).exists():
        log.error(f"Model not found: {args.model}. Train the model first.")
        sys.exit(1)

    config = load_config(args.config)
    run_evaluation(args.model, config)
