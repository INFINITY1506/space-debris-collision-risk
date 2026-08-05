"""
Download pre-trained model checkpoints from HuggingFace Hub.

Usage:
    python download_models.py

Set HF_REPO_ID env var to override the default repo.
Models are saved to data/models/.
"""

import os
import hashlib
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Change this to your HuggingFace repo ---
HF_REPO_ID = os.getenv("HF_REPO_ID", "infinity1506/space-debris-models")
HF_REVISION = os.getenv("HF_REVISION", "6a0bb8dfd51c244e06fcdf34c6cc88c52b3864ec")
BEST_MODEL_SHA256 = os.getenv(
    "BEST_MODEL_SHA256",
    "4c45b63aab7bca2c9ddb5450590496da0b8d4755a06480ebe3ba12601682cc78",
)

MODEL_FILES = ["best_model.pth"]
ENSEMBLE_FILES = [
    "ckpt_ep039_auc0.9999.pth",
    "ckpt_ep041_auc0.9999.pth",
    "ckpt_ep048_auc0.9999.pth",
    "last.pth",
]
if os.getenv("DOWNLOAD_ENSEMBLE", "false").lower() in {"1", "true", "yes"}:
    MODEL_FILES.extend(ENSEMBLE_FILES)

# Normalization parameters (required for correct inference)
NORM_DIR = Path(__file__).parent / "data" / "processed"
NORM_DIR.mkdir(parents=True, exist_ok=True)
NORM_FILES = [
    "normalization.npz",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_from_huggingface():
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before downloading models") from exc

    print(f"Downloading models from HuggingFace: {HF_REPO_ID}")
    print(f"Saving to: {MODEL_DIR}")
    print()

    for filename in MODEL_FILES:
        dest = MODEL_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  [SKIP] {filename} already exists ({size_mb:.1f} MB)")
            continue

        print(f"  [DOWN] {filename} ...", end=" ", flush=True)
        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                revision=HF_REVISION,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            if filename == "best_model.pth":
                digest = sha256_file(Path(path))
                if digest != BEST_MODEL_SHA256:
                    raise RuntimeError(
                        f"Checksum mismatch for {filename}: expected {BEST_MODEL_SHA256}, got {digest}"
                    )
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"\n  Make sure the repo '{HF_REPO_ID}' exists and is public.")
            print(f"  Create it at: https://huggingface.co/new")
            sys.exit(1)

    # Download normalization parameters
    for filename in NORM_FILES:
        dest = NORM_DIR / filename
        if dest.exists():
            print(f"  [SKIP] {filename} already exists")
            continue

        print(f"  [DOWN] {filename} ...", end=" ", flush=True)
        try:
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                revision=HF_REVISION,
                local_dir=str(NORM_DIR),
                local_dir_use_symlinks=False,
            )
            print(f"OK")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    print("\nAll models downloaded successfully!")


def download_from_url():
    """Fallback: download from direct URL (Google Drive, etc.)."""
    import requests

    model_url = os.getenv("MODEL_DOWNLOAD_URL", "")
    if not model_url:
        print("ERROR: No MODEL_DOWNLOAD_URL set and HuggingFace download failed.")
        print("Set either HF_REPO_ID or MODEL_DOWNLOAD_URL environment variable.")
        sys.exit(1)

    print(f"Downloading models from: {model_url}")
    # Add direct URL download logic if needed


if __name__ == "__main__":
    required_files = [(MODEL_DIR, f) for f in MODEL_FILES] + [(NORM_DIR, f) for f in NORM_FILES]
    existing = [(directory, name) for directory, name in required_files if (directory / name).exists()]
    if len(existing) == len(required_files):
        digest = sha256_file(MODEL_DIR / "best_model.pth")
        if digest != BEST_MODEL_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for best_model.pth: expected {BEST_MODEL_SHA256}, got {digest}"
            )
        print("All model and normalization files already present. Nothing to download.")
        sys.exit(0)

    missing = [name for directory, name in required_files if not (directory / name).exists()]
    print(f"Missing {len(missing)} model file(s): {', '.join(missing)}")
    print()

    download_from_huggingface()
