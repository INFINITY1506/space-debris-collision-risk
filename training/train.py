"""
train.py
--------
Training loop for the Space Debris Collision Risk Transformer.
Implements:
  - Evidential Deep Learning loss (EDL)
  - Mixed precision (FP16) training via torch.cuda.amp
  - AdamW + OneCycleLR scheduler
  - Gradient clipping
  - Early stopping
  - TensorBoard logging
  - Best-N checkpoint saving

Usage:
    python training/train.py
    python training/train.py --config training/config.yaml --resume data/models/last.pth
"""

import sys
import os
import time
import math
import json
import logging
import argparse
import random
import yaml
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import h5py
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.models.transformer import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ConjunctionDataset(Dataset):
    """
    PyTorch Dataset reading conjunction features from HDF5.
    Each sample is a trajectory tensor [T, F] with a risk label and
    optional collision probability.
    """

    def __init__(self, h5_path: str | Path, augment: bool = False):
        self.h5_path = Path(h5_path)
        self.augment = augment

        with h5py.File(self.h5_path, "r") as f:
            self.X_traj = f["X_traj"][:].astype(np.float32)   # [N, T, F]
            self.y_label = f["y_label"][:]                     # [N]
            self.y_prob  = f["y_prob"][:]                      # [N]

        self.N = len(self.y_label)
        log.info(f"Loaded {self.N:,} samples from {self.h5_path.name}")

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        x = torch.tensor(self.X_traj[idx], dtype=torch.float32)  # [T, F]
        y = int(self.y_label[idx])
        p = float(self.y_prob[idx])

        if self.augment:
            # Temporal jitter: random time sub-sequence of length 120-168
            T = x.shape[0]
            t_start = random.randint(0, max(0, T - 120))
            x = x[t_start:]  # will be padded in collate_fn if needed

            # Add small Gaussian noise to features
            x = x + torch.randn_like(x) * 0.01

        return {"x": x, "label": y, "prob": p}


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to max length in batch and build padding mask."""
    max_len = max(item["x"].shape[0] for item in batch)
    B = len(batch)
    F = batch[0]["x"].shape[1]

    x_padded = torch.zeros(B, max_len, F)
    mask = torch.ones(B, max_len + 1, dtype=torch.bool)  # +1 for CLS token

    for i, item in enumerate(batch):
        T_i = item["x"].shape[0]
        x_padded[i, :T_i, :] = item["x"]
        mask[i, 1:T_i + 1] = False  # unmasked positions (CLS always visible)
    mask[:, 0] = False  # unmask CLS token

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    probs  = torch.tensor([item["prob"]  for item in batch], dtype=torch.float32)

    return {"x": x_padded, "label": labels, "prob": probs, "mask": mask}


# ---------------------------------------------------------------------------
# Loss Function: Evidential Deep Learning
# ---------------------------------------------------------------------------
def edl_mse_loss(
    alpha: torch.Tensor,   # [batch, n_classes] Dirichlet params
    y: torch.Tensor,       # [batch] integer class labels
    epoch: int,
    n_classes: int = 3,
    kl_weight: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Evidential Deep Learning loss combining:
      1. MSE loss between predicted expected probability and one-hot target
      2. KL divergence regularizer to penalize spurious evidence

    Args:
        alpha: Dirichlet concentration parameters [batch, K]
        y: Integer class labels [batch]
        epoch: Current epoch (for annealing KL weight)
        n_classes: Number of classes
        kl_weight: Base KL regularization weight

    Returns:
        total_loss, loss_dict
    """
    # One-hot encode
    y_onehot = F.one_hot(y, num_classes=n_classes).float()  # [batch, K]

    # Total Dirichlet strength
    S = alpha.sum(dim=-1, keepdim=True)  # [batch, 1]

    # MSE between expected probability and target
    p_hat = alpha / S
    mse = ((y_onehot - p_hat) ** 2).sum(dim=-1).mean()

    # Variance term
    var = (alpha * (S - alpha)) / (S ** 2 * (S + 1))
    var_loss = var.sum(dim=-1).mean()

    loss_mse_var = mse + var_loss

    # KL divergence from Dirichlet(alpha_tilde) to Dirichlet(1, ..., 1)
    # where alpha_tilde = 1 + (1 - y_onehot) * evidence (remove target class evidence)
    alpha_tilde = y_onehot + (1 - y_onehot) * alpha
    # Anneal KL weight over epochs
    annealed_weight = min(1.0, epoch / 10.0) * kl_weight

    sum_alpha_tilde = alpha_tilde.sum(dim=-1)  # [batch]
    kl_loss = (
        torch.lgamma(sum_alpha_tilde)
        - torch.lgamma(torch.tensor(float(n_classes), device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1) * (
            torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde.unsqueeze(-1))
        )).sum(dim=-1)
    ).mean()

    total_loss = loss_mse_var + annealed_weight * kl_loss

    return total_loss, {
        "mse_loss": mse.item(),
        "var_loss": var_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Keeps the best N checkpoints by a tracked metric."""

    def __init__(self, save_dir: Path, keep_n: int = 3, metric: str = "val_auc", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n
        self.metric_name = metric
        self.mode = mode
        self.checkpoints: list[tuple[float, Path]] = []

    def save(self, state: dict, epoch: int, metric_val: float) -> Path:
        fname = self.save_dir / f"ckpt_epoch{epoch:03d}_{self.metric_name}{metric_val:.4f}.pth"
        torch.save(state, fname)

        self.checkpoints.append((metric_val, fname))
        # Sort: best first
        self.checkpoints.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Remove extras
        while len(self.checkpoints) > self.keep_n:
            _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                log.info(f"Removed old checkpoint: {old_path.name}")

        # Always save as "best" if it's the top
        best_val = self.checkpoints[0][0]
        if metric_val == best_val:
            best_path = self.save_dir / "best_model.pth"
            shutil.copyfile(fname, best_path)
            log.info(f"✅ New best model: {self.metric_name}={metric_val:.4f}")

        # Save last checkpoint
        last_path = self.save_dir / "last.pth"
        torch.save(state, last_path)

        return fname

    def best_metric(self) -> float | None:
        return self.checkpoints[0][0] if self.checkpoints else None


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    n_classes: int = 3,
) -> dict:
    """Run evaluation and return full metrics dict."""
    model.eval()
    all_labels, all_preds, all_probs_soft = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        labels = batch["label"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=device.type == "cuda"):
            evidence, alpha, uncertainty, prob = model(x, src_key_padding_mask=mask)
            loss, _ = edl_mse_loss(alpha, labels, epoch, n_classes)

        total_loss += loss.item()
        n_batches += 1

        preds = prob.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs_soft.append(prob.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_soft = np.vstack(all_probs_soft)

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # AUC-ROC (one-vs-rest for multiclass)
    try:
        auc = roc_auc_score(y_true, y_soft, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict) -> None:
    """Main training loop."""
    set_seed(config.get("seed", 42))

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_cfg     = config["data"]
    model_cfg    = config.get("model", {})
    train_cfg    = config["training"]
    opt_cfg      = config["optimizer"]
    sched_cfg    = config["scheduler"]
    ckpt_cfg     = config["checkpoint"]
    log_cfg      = config["logging"]

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    log.info("Loading datasets...")
    train_ds = ConjunctionDataset(data_cfg["train_file"], augment=True)
    val_ds   = ConjunctionDataset(data_cfg["val_file"],   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True) and device.type == "cuda",
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=train_cfg.get("pin_memory", True) and device.type == "cuda",
        collate_fn=collate_fn,
    )

    log.info(f"Train batches: {len(train_loader):,}, Val batches: {len(val_loader):,}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    log.info("Building model...")
    model = build_model(config).to(device)
    n_params = model.count_parameters()
    log.info(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # -----------------------------------------------------------------------
    # Optimizer & Scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
    )

    total_steps = len(train_loader) * train_cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(sched_cfg["max_lr"]),
        total_steps=total_steps,
        pct_start=float(sched_cfg["pct_start"]),
        anneal_strategy=str(sched_cfg["anneal_strategy"]),
        div_factor=float(sched_cfg["div_factor"]),
        final_div_factor=float(sched_cfg["final_div_factor"]),
    )

    scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True) and device.type == "cuda")

    # -----------------------------------------------------------------------
    # Logging & Checkpoints
    # -----------------------------------------------------------------------
    writer = SummaryWriter(log_dir=log_cfg["tensorboard_dir"])
    ckpt_mgr = CheckpointManager(
        save_dir=Path(ckpt_cfg["save_dir"]),
        keep_n=ckpt_cfg["keep_best_n"],
        metric=ckpt_cfg["metric"],
        mode=ckpt_cfg["mode"],
    )

    n_classes = model_cfg.get("output_classes", 3)
    best_val_auc = 0.0
    patience_counter = 0
    patience = train_cfg["early_stopping_patience"]
    global_step = 0

    log.info("Starting training...")

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    for epoch in range(1, train_cfg["num_epochs"] + 1):
        epoch_start = time.time()
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for batch_idx, batch in enumerate(train_loader):
            x      = batch["x"].to(device)
            labels = batch["label"].to(device)
            mask   = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=scaler.is_enabled()):
                evidence, alpha, uncertainty, prob = model(x, src_key_padding_mask=mask)
                loss, loss_dict = edl_mse_loss(alpha, labels, epoch, n_classes)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_max_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * len(labels)
            train_n += len(labels)
            global_step += 1

            if global_step % log_cfg["log_every_n_steps"] == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/loss_mse", loss_dict["mse_loss"], global_step)
                writer.add_scalar("train/loss_kl", loss_dict["kl_loss"], global_step)

        train_loss_avg = train_loss_sum / max(train_n, 1)

        # -----------------------------------------------------------------------
        # Validation
        # -----------------------------------------------------------------------
        val_metrics = evaluate(model, val_loader, device, epoch, n_classes)
        epoch_time = time.time() - epoch_start

        log.info(
            f"Epoch {epoch:3d}/{train_cfg['num_epochs']} | "
            f"train_loss={train_loss_avg:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"time={epoch_time:.0f}s"
        )

        # TensorBoard
        writer.add_scalar("val/loss",     val_metrics["loss"],     epoch)
        writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
        writer.add_scalar("val/auc",      val_metrics["auc"],      epoch)
        writer.add_scalar("val/f1",       val_metrics["f1"],       epoch)
        writer.add_scalar("val/precision",val_metrics["precision"],epoch)
        writer.add_scalar("val/recall",   val_metrics["recall"],   epoch)
        writer.add_scalar("train/loss_epoch", train_loss_avg, epoch)

        # -----------------------------------------------------------------------
        # Checkpointing & Early Stopping
        # -----------------------------------------------------------------------
        val_auc = val_metrics["auc"]

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val_metrics,
            "train_loss": train_loss_avg,
            "config": config,
        }
        ckpt_mgr.save(state, epoch, val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping triggered (patience={patience})")
                break

    writer.close()
    log.info(f"\n✅ Training complete! Best val AUC: {best_val_auc:.4f}")
    log.info(f"   Best model saved to: {Path(ckpt_cfg['save_dir']) / 'best_model.pth'}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Space Debris Collision Risk Transformer")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    if not Path(args.config).exists():
        log.error(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Verify data exists
    for key in ["train_file", "val_file"]:
        p = Path(config["data"][key])
        if not p.exists():
            log.error(f"Data file not found: {p}. Run training/preprocess.py first.")
            sys.exit(1)

    train(config)
