# 🛰️ Space Debris Collision Risk Predictor

> Deep learning system for predicting satellite-debris collision risks using a 150M-parameter transformer with evidential uncertainty quantification.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue)](https://reactjs.org)

---

## 🎯 Overview

This system predicts collision risks between satellites and space debris using:
- **TLE data** from CelesTrak (18,000+ objects)
- **SGP4 orbital propagation** over a 7-day horizon
- **Transformer neural network** (150M parameters) trained on conjunction events
- **Evidential Deep Learning** for uncertainty quantification (epistemic + aleatoric)

## 📁 Project Structure

```
space-debris-predictor/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── predictor.py            # Inference pipeline
│   ├── models/transformer.py  # 150M transformer architecture
│   └── utils/
│       ├── sgp4_propagator.py  # Orbital mechanics
│       └── feature_engineering.py
├── frontend/                   # React + Vite + TypeScript UI
├── training/
│   ├── config.yaml             # Training configuration
│   ├── data_download.py        # TLE data acquisition
│   ├── preprocess.py           # Conjunction dataset generation
│   ├── train.py                # Training loop
│   └── evaluate.py             # Model evaluation
├── data/
│   ├── raw/                    # TLE files & catalog
│   ├── processed/              # HDF5 datasets
│   └── models/                 # Checkpoints
└── docs/                       # Evaluation plots & metrics
```

## 🚀 Quickstart

### Option A: Automated Setup

```bash
# Clone the repo
git clone https://github.com/INFINITY1506/space-debris-collision-risk.git
cd space-debris-collision-risk

# Linux/Mac
chmod +x setup.sh && ./setup.sh

# Windows (double-click or run from terminal)
setup.bat
```

### Option B: Manual Setup

#### 1. Install Python Dependencies

```bash
pip install -r backend/requirements.txt
```

For GPU acceleration (optional):
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Install Frontend Dependencies

```bash
cd frontend && npm install && cd ..
```

#### 3. Download Model Checkpoints

The pre-trained model files (~290MB total) exceed GitHub's file size limit and must be downloaded separately.

**Required files** (place in `data/models/`):**

| File | Size | Description |
|------|------|-------------|
| `best_model.pth` | ~60MB | Primary model (required) |
| `ckpt_ep039_auc0.9999.pth` | ~60MB | Ensemble checkpoint |
| `ckpt_ep041_auc0.9999.pth` | ~60MB | Ensemble checkpoint |
| `ckpt_ep048_auc0.9999.pth` | ~60MB | Ensemble checkpoint |
| `last.pth` | ~60MB | Final training checkpoint |

> **Note:** At minimum, `best_model.pth` is required to run the backend. The ensemble checkpoints improve prediction quality.

**Alternatively, train from scratch:**
```bash
python training/data_download.py   # Download TLE data from CelesTrak
python training/preprocess.py      # Generate conjunction dataset
python training/train.py           # Train (~50 epochs, ~15-20 min/epoch on GPU)
python training/evaluate.py        # Evaluate & generate plots
```

#### 4. Start the Application

```bash
# Terminal 1 — Backend API
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173** in your browser.

---

## 🧠 Model Architecture

| Component             | Spec                                |
|-----------------------|-------------------------------------|
| Input                 | [batch, 168 timesteps, 22 features] |
| Embedding             | Linear(22 → 1024) + LayerNorm       |
| Positional Encoding   | Sinusoidal (learned CLS token)      |
| Encoder Layers        | 10 × TransformerEncoderLayer        |
| Attention Heads       | 16                                  |
| Feedforward Dim       | 4096 (GELU activation)              |
| Interaction Module    | Cross-attention pairwise module     |
| Output Head           | Evidential DL (Dirichlet, K=3)      |
| Total Parameters      | ~150M                               |

**Output:** Collision probability + epistemic uncertainty + aleatoric uncertainty

---

## 📊 Performance Targets

| Metric         | Target      |
|---------------|-------------|
| Accuracy       | ≥ 94%       |
| AUC-ROC        | ≥ 0.98      |
| ECE            | < 0.05      |
| Inference time | < 5 seconds |

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                           |
|--------|-----------------------|---------------------------------------|
| POST   | `/predict`            | Satellite → top-10 debris threats     |
| GET    | `/satellites`         | List all catalog objects (w/ search)  |
| GET    | `/satellite/{id}`     | Satellite orbital details             |
| GET    | `/health`             | API status & model info               |

### Example: `/predict`
```json
// POST /predict
{ "satellite_name": "ISS", "top_n": 10 }

// Response
{
  "satellite": { "name": "ISS (ZARYA)", "norad_id": 25544, "altitude_km": 408.3, ... },
  "threats": [
    {
      "rank": 1,
      "debris_name": "COSMOS 2251 DEB",
      "collision_probability_pct": "0.0023%",
      "uncertainty_range": "±0.12%",
      "risk_level": "HIGH",
      "miss_distance_km": 2.15,
      "tca_utc": "2026-03-07T14:22:00Z",
      "relative_velocity_km_s": 14.37
    }, ...
  ],
  "total_time_s": 3.2
}
```

---

## ⚙️ Training Configuration

Edit `training/config.yaml` to adjust:
- Batch size, learning rate, epochs
- Model architecture (d_model, n_heads, n_layers)  
- Data paths and risk thresholds

---

## 📈 Features (30+)

The model uses 30 features computed per conjunction:

- **Orbital (14):** Semi-major axis, eccentricity, inclination, RAAN, arg. perigee, mean anomaly, altitude — for both satellite and debris
- **Relative (10):** Miss distance, relative velocity, TCA time, altitude diff, inclination diff, RAAN diff, RSW components, period ratio
- **Physical (5):** Combined mass/cross-section, kinetic energy, hardness factor, momentum transfer
- **Temporal (3):** Hours to TCA, orbital decay rate, time since epoch

---

## 🚂 Deploy on Railway

### Step 1: Upload Models to HuggingFace (free, one-time)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Create a new model repo: `infinity1506/space-debris-models`
3. Upload the 5 `.pth` files from `data/models/`:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload infinity1506/space-debris-models data/models/ --include="*.pth"
   ```

### Step 2: Deploy on Railway

1. Go to [railway.app](https://railway.app) and connect your GitHub
2. Click **"New Project"** → **"Deploy from GitHub Repo"**
3. Select `INFINITY1506/space-debris-collision-risk`
4. Railway auto-detects the Dockerfile and deploys
5. The build will download models from HuggingFace automatically

### Step 3: Get your URL

Railway assigns a public URL like `your-app.up.railway.app`. The app serves both the API and the frontend from a single service.

---

## ⚠️ Known Limitations

1. **Static data**: Uses daily TLE snapshots, not live updates
2. **7-day window**: Predictions only for the next 7 days
3. **Simplified physics**: Does not account for maneuvers or atmospheric drag variations
4. **Label quality**: Ground truth based on physics approximations, not verified collision records
5. **GPU requirement**: Training requires CUDA-capable GPU (RTX 5080 with nightly PyTorch)

---

## 🔮 Future Work

- Real-time TLE updates from Space-Track.org
- Multi-satellite conjunction analysis
- Maneuver recommendation engine
- Historical trend analysis
- ~~Docker containerization~~ (done)

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙏 Data Sources

- **TLE Data**: [CelesTrak](https://celestrak.org) — public domain
- **SGP4 Library**: [sgp4](https://github.com/brandon-rhodes/python-sgp4)
- **Skyfield**: [skyfield](https://rhodesmill.org/skyfield/)
