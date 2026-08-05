# Debris Sentinel

Debris Sentinel is a research-grade web application for screening close approaches between a selected satellite and public debris TLEs. It combines SGP4 propagation, explicit miss-distance ranking, a lightweight collision-probability estimate, and optional transformer diagnostics in an interactive 3D interface.

Live service: [debris-sentinel-cd536idu7a-as.a.run.app](https://debris-sentinel-cd536idu7a-as.a.run.app)

> **Safety notice:** This is a portfolio and research project, not an operational conjunction-assessment system. Do not use its results to make spacecraft maneuver or safety decisions. Operational decisions require authoritative ephemerides, covariance data, conjunction data messages, and expert review.

## What it does

- Refreshes public active-satellite and debris-group TLEs from CelesTrak at startup.
- Propagates candidate debris over a seven-day horizon with SGP4.
- Excludes active spacecraft and co-orbiting modules from the debris threat list.
- Ranks results by minimum propagated miss distance.
- Shows an approximate physics-based collision probability and explicit risk thresholds.
- Exposes the transformer output only as an advisory diagnostic; it does not control ranking, displayed collision probability, or risk level.
- Serves a React/Three.js frontend and FastAPI API from one container.

The bundled checkpoint is approximately 5 million parameters (6 encoder layers, 8 heads, 256-dimensional embedding). Its reported validation metrics came from threshold-derived training data and should not be interpreted as operational collision-prediction performance.

## Architecture

```text
CelesTrak TLE snapshot
        |
  debris-only filtering
        |
SGP4 propagation (7 days, hourly)
        |
32-element conjunction feature vector
        |
miss-distance ranking + screening probability
        |
advisory transformer diagnostics
        |
FastAPI + React interface
```

## Local setup

Requirements: Python 3.12+, Node.js 20+, and about 1 GB of free disk space.

```bash
git clone https://github.com/INFINITY1506/space-debris-collision-risk.git
cd space-debris-collision-risk

python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python download_models.py
python training/data_download.py

cd frontend
npm ci
npm run build
cd ..

python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Run the checks with:

```bash
pytest -q
cd frontend && npm run build && npm audit
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` or `/api/health` | Readiness, catalog age, and model status |
| `POST` | `/predict` or `/api/predict` | Top debris screening results |
| `POST` | `/predict/detailed` | B-plane and Monte Carlo research views |
| `GET` | `/satellites` | Search the current catalog |
| `GET` | `/satellite/{norad_id}` | Retrieve an object's orbital fields |
| `POST` | `/maneuver` | Experimental maneuver illustration |
| `POST` | `/interpret` | Experimental model interpretation |

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"norad_id": 25544, "top_n": 10}'
```

## Cloud Run deployment

The production container is sized for a small public portfolio deployment:

- 1 vCPU and 2 GiB memory
- request-based billing with scale-to-zero
- concurrency 1
- maximum 1 instance to cap spend
- 300-second request timeout
- Singapore region (`asia-southeast1`)

Deploy with:

```bash
./scripts/deploy_cloud_run.sh YOUR_GOOGLE_CLOUD_PROJECT_ID
```

See [docs/DEPLOY_CLOUD_RUN.md](docs/DEPLOY_CLOUD_RUN.md) for billing, domain mapping, DNS, rollback, and production checks.

## Data and model files

- `data/raw/catalog.csv` is a deployable snapshot and is refreshed when it is more than 48 hours old.
- `best_model.pth` and `normalization.npz` are downloaded from the public Hugging Face repository during the container build.
- Set `DOWNLOAD_ENSEMBLE=true` during a custom build only if the optional checkpoints are needed.
- Startup fails closed if the checkpoint, normalization file, catalog, or required debris sources are unavailable or invalid.

## Known limitations

- Public TLEs do not include the covariance information required for an operational collision probability.
- Hourly sampling can miss a closer approach between samples.
- TLE propagation uncertainty grows with prediction horizon and varies by object.
- The probability estimate uses simplified object assumptions.
- The current transformer repeats static conjunction features over its sequence and is retained only for research diagnostics.
- Rate limiting is per container instance and is not a substitute for an edge security service.

## Technology

Python, PyTorch, FastAPI, SGP4, React, TypeScript, Vite, Three.js, Docker, GitHub Actions, and Google Cloud Run.

## License

MIT — see [LICENSE](LICENSE).
