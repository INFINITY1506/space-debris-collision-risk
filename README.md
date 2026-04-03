# Debris Sentinel - Space Collision Risk Intelligence

> Real-time satellite-debris collision risk prediction powered by a 150M-parameter transformer with evidential uncertainty quantification.

<p align="center">
  <a href="https://space-debris-collision-risk-production.up.railway.app"><strong>View Live Demo</strong></a>
</p>

[![Live](https://img.shields.io/badge/Status-Live-brightgreen)](https://space-debris-collision-risk-production.up.railway.app)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://reactjs.org)
[![Railway](https://img.shields.io/badge/Deployed_on-Railway-0B0D0E?logo=railway&logoColor=white)](https://railway.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo

**https://space-debris-collision-risk-production.up.railway.app**

Search for any satellite (e.g. ISS, Hubble, Starlink), click it on the interactive 3D globe, and get AI-powered collision risk analysis against 17,000+ tracked objects in real time.

---

## What It Does

This system predicts collision risks between active satellites and space debris by combining orbital mechanics with deep learning:

- **17,000+ tracked objects** from CelesTrak TLE catalog
- **SGP4 orbital propagation** over a 7-day prediction horizon
- **150M-parameter transformer** trained on conjunction events with cross-attention
- **Evidential Deep Learning** providing calibrated uncertainty (epistemic + aleatoric)
- **Interactive 3D globe** with real-time satellite visualization

---

## Architecture

```
TLE Catalog (17K objects)
        |
   SGP4 Propagation (7-day horizon)
        |
   Feature Engineering (30 features per conjunction)
        |
   Transformer Encoder (10 layers, 16 heads, 150M params)
        |
   Evidential Output Head (Dirichlet, K=3)
        |
   Risk Classification: LOW | MEDIUM | HIGH
   + Collision Probability (%) + Uncertainty Bounds
```

### Transformer Specifications

| Component             | Specification                       |
|-----------------------|-------------------------------------|
| Input                 | [batch, 168 timesteps, 22 features] |
| Embedding             | Linear(22 -> 1024) + LayerNorm      |
| Positional Encoding   | Sinusoidal (learned CLS token)      |
| Encoder Layers        | 10 x TransformerEncoderLayer        |
| Attention Heads       | 16                                  |
| Feedforward Dim       | 4096 (GELU activation)              |
| Interaction Module    | Cross-attention pairwise module     |
| Output Head           | Evidential DL (Dirichlet, K=3)      |
| Total Parameters      | ~150M                               |

---

## Project Structure

```
space-debris-collision-risk/
├── backend/
│   ├── main.py                   # FastAPI application & static file serving
│   ├── predictor.py              # End-to-end inference pipeline
│   ├── models/
│   │   └── transformer.py        # 150M transformer architecture
│   └── utils/
│       ├── sgp4_propagator.py    # SGP4 orbital propagation
│       ├── feature_engineering.py # 30-feature conjunction computation
│       ├── bplane.py             # B-plane geometry calculations
│       ├── maneuver.py           # Avoidance maneuver computation
│       └── interpret.py          # Attention & feature importance
├── frontend/                     # React + Vite + TypeScript + Three.js
│   ├── src/
│   │   ├── components/
│   │   │   ├── GlobeView.tsx     # Interactive 3D globe (react-globe.gl)
│   │   │   └── ResultsPanel.tsx  # Threat analysis dashboard
│   │   └── App.tsx
│   └── package.json
├── training/
│   ├── config.yaml               # Hyperparameter configuration
│   ├── data_download.py          # CelesTrak TLE acquisition
│   ├── preprocess.py             # Conjunction dataset generation
│   ├── train.py                  # Training loop with mixed precision
│   └── evaluate.py               # Evaluation & metrics
├── data/
│   ├── raw/                      # TLE files & satellite catalog
│   ├── processed/                # Conjunction datasets
│   └── models/                   # Model checkpoints (~290MB)
├── Dockerfile                    # Multi-stage build (Node + Python)
├── railway.toml                  # Railway deployment config
└── download_models.py            # HuggingFace model downloader
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Node.js 18+
- ~300MB disk space for model checkpoints

### Quick Setup

```bash
# Clone
git clone https://github.com/INFINITY1506/space-debris-collision-risk.git
cd space-debris-collision-risk

# Backend
pip install -r backend/requirements.txt
python download_models.py

# Frontend
cd frontend && npm install && cd ..

# Run
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
cd frontend && npm run dev
```

Open **http://localhost:5173**

### Automated Setup (Alternative)

```bash
# Linux/Mac
chmod +x setup.sh && ./setup.sh

# Windows
setup.bat
```

### GPU Acceleration (Optional)

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only (default)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Model Checkpoints

Pre-trained weights (~290MB) are hosted on HuggingFace and downloaded automatically:

| File | Size | Description |
|------|------|-------------|
| `best_model.pth` | 57MB | Primary model (required) |
| `ckpt_ep039_auc0.9999.pth` | 57MB | Ensemble checkpoint |
| `ckpt_ep041_auc0.9999.pth` | 57MB | Ensemble checkpoint |
| `ckpt_ep048_auc0.9999.pth` | 57MB | Ensemble checkpoint |
| `last.pth` | 57MB | Final training checkpoint |

At minimum, `best_model.pth` is required. Ensemble checkpoints improve prediction robustness.

### Train From Scratch

```bash
python training/data_download.py   # Download TLE data from CelesTrak
python training/preprocess.py      # Generate conjunction dataset
python training/train.py           # Train (~50 epochs, ~15-20 min/epoch on GPU)
python training/evaluate.py        # Evaluate & generate plots
```

---

## API Reference

Base URL: `https://space-debris-collision-risk-production.up.railway.app`

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| POST   | `/predict`        | Satellite -> top-N debris threats  |
| GET    | `/satellites`     | List catalog objects (with search) |
| GET    | `/satellite/{id}` | Satellite orbital details          |
| GET    | `/health`         | API status & model availability    |

### Example Request

```bash
curl -X POST https://space-debris-collision-risk-production.up.railway.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"satellite_name": "ISS", "top_n": 10}'
```

### Example Response

```json
{
  "satellite": {
    "name": "ISS (ZARYA)",
    "norad_id": 25544,
    "altitude_km": 428.6,
    "inclination_deg": 51.63
  },
  "threats": [
    {
      "rank": 1,
      "debris_name": "COSMOS 2251 DEB",
      "collision_probability_pct": "0.0173%",
      "uncertainty_range": "+/-0.52%",
      "risk_level": "LOW",
      "miss_distance_km": 2.15,
      "tca_utc": "2026-03-25T14:22:00Z",
      "relative_velocity_km_s": 14.37
    }
  ],
  "pairs_analyzed": 2726,
  "total_time_s": 77.9
}
```

---

## Feature Engineering (30 Features)

| Category | Count | Features |
|----------|-------|----------|
| Orbital | 14 | Semi-major axis, eccentricity, inclination, RAAN, arg. perigee, mean anomaly, altitude (x2 for satellite & debris) |
| Relative | 10 | Miss distance, relative velocity, TCA time, altitude diff, inclination diff, RAAN diff, RSW components, period ratio |
| Physical | 5 | Combined mass/cross-section, kinetic energy, hardness factor, momentum transfer |
| Temporal | 3 | Hours to TCA, orbital decay rate, time since epoch |

---

## Performance

| Metric         | Value       |
|---------------|-------------|
| Accuracy       | >= 94%      |
| AUC-ROC        | 0.9999      |
| ECE            | < 0.05      |
| Inference time | ~70s (cloud CPU) / <5s (GPU) |

---

## Deployment

### Railway (Current Production)

The app is deployed as a single Docker container on Railway, serving both the FastAPI backend and React frontend.

```toml
# railway.toml
[build]
builder = "DOCKERFILE"

[deploy]
startCommand = "sh -c 'python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}'"
healthcheckPath = "/health"
healthcheckTimeout = 600
```

To deploy your own instance:

1. Fork this repository
2. Upload model weights to HuggingFace (see `download_models.py`)
3. Connect the repo on [railway.app](https://railway.app)
4. Railway auto-builds and deploys from the Dockerfile

### Docker (Self-hosted)

```bash
docker build -t debris-sentinel .
docker run -p 8000:8000 debris-sentinel
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | PyTorch 2.1+ |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + TypeScript + Vite |
| 3D Globe | react-globe.gl + Three.js |
| Orbital Mechanics | SGP4 (python-sgp4) |
| Model Hosting | HuggingFace Hub |
| Deployment | Railway (Docker) |

---

## Known Limitations

1. **Static TLE data** -- uses daily snapshots, not real-time feeds
2. **7-day prediction window** -- beyond this, TLE propagation error grows significantly
3. **Simplified physics** -- does not model maneuvers, atmospheric drag variations, or solar radiation pressure
4. **Cloud CPU inference** -- predictions take ~70s on Railway's shared CPU; <5s with GPU
5. **Label approximation** -- ground truth derived from physics-based thresholds, not verified collision records

---

## Future Roadmap

- [ ] Real-time TLE updates via Space-Track.org API
- [ ] Multi-satellite conjunction analysis
- [ ] Maneuver recommendation engine with delta-v optimization
- [ ] Historical trend analysis and conjunction frequency tracking
- [ ] GPU-accelerated inference tier
- [x] Docker containerization
- [x] Cloud deployment (Railway)
- [x] Interactive 3D satellite visualization

---

## Data Sources

- **TLE Catalog**: [CelesTrak](https://celestrak.org) (public domain)
- **SGP4 Library**: [python-sgp4](https://github.com/brandon-rhodes/python-sgp4)
- **Skyfield**: [rhodesmill.org/skyfield](https://rhodesmill.org/skyfield/)

---

## License

MIT License -- see [LICENSE](LICENSE)

---

<p align="center">
  Built by <a href="https://github.com/INFINITY1506">INFINITY1506</a>
</p>
