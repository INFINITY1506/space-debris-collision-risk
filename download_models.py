"""
Download pre-trained model checkpoints from HuggingFace Hub.

Usage:
    python download_models.py

Set HF_REPO_ID env var to override the default repo.
Models are saved to data/models/.
"""

import os
import sys
import hashlib
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Change this to your HuggingFace repo ---
HF_REPO_ID = os.getenv("HF_REPO_ID", "infinity1506/space-debris-models")

MODEL_FILES = [
    "best_model.pth",
    "ckpt_ep039_auc0.9999.pth",
    "ckpt_ep041_auc0.9999.pth",
    "ckpt_ep048_auc0.9999.pth",
    "last.pth",
]


def download_from_huggingface():
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

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
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"\n  Make sure the repo '{HF_REPO_ID}' exists and is public.")
            print(f"  Create it at: https://huggingface.co/new")
            sys.exit(1)

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
    # Check if models already exist
    existing = [f for f in MODEL_FILES if (MODEL_DIR / f).exists()]
    if len(existing) == len(MODEL_FILES):
        print("All model files already present. Nothing to download.")
        sys.exit(0)

    missing = [f for f in MODEL_FILES if not (MODEL_DIR / f).exists()]
    print(f"Missing {len(missing)} model file(s): {', '.join(missing)}")
    print()

    download_from_huggingface()
