"""
main.py
-------
FastAPI backend for the Space Debris Collision Risk Predictor.

Endpoints:
  POST /predict            — satellite name → top 10 debris threats
  GET  /satellites         — list all trackable satellites (with search)
  GET  /satellite/{id}     — satellite details
  GET  /health             — API status check

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Literal, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timezone


def _sanitize(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# Import predictor
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.predictor import SatellitePredictor
from backend.catalog import download_catalog_snapshot, ensure_catalog_fresh

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
MODEL_PATH   = Path(os.getenv("MODEL_PATH",   str(_ROOT / "data/models/best_model.pth")))
CATALOG_PATH = Path(os.getenv("CATALOG_PATH", str(_ROOT / "data/raw/catalog.csv")))
NORM_PATH    = Path(os.getenv("NORM_PATH",    str(_ROOT / "data/processed/normalization.npz")))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CATALOG_MAX_AGE_HOURS = float(os.getenv("CATALOG_MAX_AGE_HOURS", "48"))
CATALOG_HARD_MAX_AGE_HOURS = float(os.getenv("CATALOG_HARD_MAX_AGE_HOURS", "168"))
CATALOG_SNAPSHOT_URI = os.getenv("CATALOG_SNAPSHOT_URI", "").strip()
SYNC_CATALOG_SNAPSHOT_ON_STARTUP = os.getenv("SYNC_CATALOG_SNAPSHOT_ON_STARTUP", "true").lower() in {"1", "true", "yes"}
REFRESH_TLE_ON_STARTUP = os.getenv("REFRESH_TLE_ON_STARTUP", "false").lower() in {"1", "true", "yes"}
ALLOW_STALE_CATALOG = os.getenv("ALLOW_STALE_CATALOG", "true").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Global predictor instance (shared across requests)
# ---------------------------------------------------------------------------
predictor: Optional[SatellitePredictor] = None
predictor_state = "not_started"
predictor_error: Optional[str] = None
loaded_catalog_age_hours: Optional[float] = None
catalog_screening_available = False


def _predictor_unavailable_detail() -> str:
    """Return a user-facing readiness message without exposing internals."""
    if predictor_state == "error":
        return "Service startup failed. Please try again later."
    return "Service is warming up while the orbital catalog and model load. Please retry shortly."


def _catalog_state(age_hours: Optional[float]) -> str:
    if age_hours is None:
        return "unavailable"
    if age_hours <= CATALOG_MAX_AGE_HOURS:
        return "fresh"
    if age_hours <= CATALOG_HARD_MAX_AGE_HOURS:
        return "aging"
    return "stale"


def _catalog_detail(age_hours: Optional[float]) -> Optional[str]:
    state = _catalog_state(age_hours)
    if state == "aging":
        return (
            f"Catalog refresh is delayed ({age_hours:.1f} hours old); screening remains "
            f"available within the {CATALOG_HARD_MAX_AGE_HOURS:.0f}-hour safety window."
        )
    if state == "stale":
        return (
            f"Collision screening is paused because the catalog is {age_hours:.1f} hours old. "
            "The scheduled refresh must publish a newer validated snapshot."
        )
    return None


def _require_screening_predictor() -> SatellitePredictor:
    if not predictor or not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail=_predictor_unavailable_detail(),
            headers={"Retry-After": "5"},
        )
    if not catalog_screening_available:
        raise HTTPException(
            status_code=503,
            detail=_catalog_detail(loaded_catalog_age_hours) or "Collision screening is temporarily unavailable.",
            headers={"Retry-After": "3600"},
        )
    return predictor

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP)
# ---------------------------------------------------------------------------
from collections import defaultdict
request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 100       # requests per minute
RATE_WINDOW = 60.0     # seconds


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    times = request_counts[client_ip]
    # Remove old timestamps
    request_counts[client_ip] = [t for t in times if now - t < RATE_WINDOW]
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False
    request_counts[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------
def _load_model():
    """Load the ML model in a background thread so the server can respond to
    healthchecks immediately while the (slow) model load is in progress."""
    global predictor, predictor_state, predictor_error, loaded_catalog_age_hours, catalog_screening_available
    predictor_state = "loading"
    predictor_error = None
    log.info("Background model load started...")
    try:
        if CATALOG_SNAPSHOT_URI and SYNC_CATALOG_SNAPSHOT_ON_STARTUP:
            try:
                download_catalog_snapshot(CATALOG_SNAPSHOT_URI, CATALOG_PATH)
            except Exception:
                log.exception(
                    "Could not sync catalog snapshot %s; retaining bundled known-good catalog",
                    CATALOG_SNAPSHOT_URI,
                )
        loaded_catalog_age_hours = ensure_catalog_fresh(
            CATALOG_PATH,
            max_age_hours=CATALOG_MAX_AGE_HOURS,
            refresh=REFRESH_TLE_ON_STARTUP,
            allow_stale=ALLOW_STALE_CATALOG,
        )
        catalog_screening_available = loaded_catalog_age_hours <= CATALOG_HARD_MAX_AGE_HOURS
        predictor = SatellitePredictor(
            model_path=MODEL_PATH,
            catalog_path=CATALOG_PATH,
            norm_path=NORM_PATH,
            device="auto",
            batch_size=512,
        )
        predictor_state = "ready"
        if not catalog_screening_available:
            log.error(_catalog_detail(loaded_catalog_age_hours))
        log.info("✅ Predictor loaded successfully")
    except Exception as e:
        predictor_state = "error"
        predictor_error = str(e)
        catalog_screening_available = False
        log.exception("Failed to load predictor")
        predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in a background thread — don't block the server
    import threading
    loader = threading.Thread(target=_load_model, daemon=True)
    loader.start()
    log.info("Server started — model loading in background...")
    yield
    # Cleanup on shutdown
    log.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Space Debris Collision Risk Predictor",
    description=(
        "Research-grade orbital screening API using SGP4 propagation, "
        "physics-based ranking, and advisory model diagnostics."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Strict-Transport-Security", "max-age=31536000")
    return response


# ---------------------------------------------------------------------------
# Middleware: strip /api prefix (production — frontend calls /api/predict etc.)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def strip_api_prefix(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        request.scope["path"] = request.url.path[4:]  # /api/health → /health
    return await call_next(request)


# ---------------------------------------------------------------------------
# Middleware: rate limiting
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    client_ip = (
        request.headers.get("cf-connecting-ip")
        or forwarded_for
        or (request.client.host if request.client else "unknown")
    )
    if request.url.path not in ("/health", "/api/health", "/docs", "/redoc", "/openapi.json"):
        if not check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded (100 requests/minute)"}
            )
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    satellite_name: Optional[str] = None
    norad_id: Optional[int] = None
    top_n: int = 10

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v):
        if not 1 <= v <= 50:
            raise ValueError("top_n must be between 1 and 50")
        return v

    @field_validator("satellite_name", mode="before")
    @classmethod
    def validate_name(cls, v):
        if v and len(v) > 100:
            raise ValueError("satellite_name too long (max 100 chars)")
        return v

    model_config = {"json_schema_extra": {
        "examples": [
            {"satellite_name": "ISS"},
            {"norad_id": 25544, "top_n": 10},
        ]
    }}


class ThreatItem(BaseModel):
    rank: int
    debris_name: str
    debris_norad_id: int
    debris_source: Optional[str] = None
    collision_probability: float
    collision_probability_pct: str
    uncertainty_pct: float
    uncertainty_range: str
    risk_level: str
    risk_color: str
    probability_low: float
    probability_medium: float
    probability_high: float
    epistemic_uncertainty: float
    miss_distance_km: float
    tca_utc: str
    tca_timestamp: float
    relative_velocity_km_s: float


class PredictResponse(BaseModel):
    satellite: dict
    threats: list[ThreatItem]
    n_candidates_analyzed: int
    propagation_time_s: float
    inference_time_s: float
    total_time_s: float
    ranking_basis: Optional[str] = None
    message: Optional[str] = None


class DetailedPredictRequest(BaseModel):
    satellite_name: Optional[str] = None
    norad_id: Optional[int] = None
    top_n: int = 10
    n_monte_carlo: int = 50

    @field_validator("top_n")
    @classmethod
    def validate_top_n(cls, v):
        if not 1 <= v <= 50:
            raise ValueError("top_n must be between 1 and 50")
        return v

    @field_validator("n_monte_carlo")
    @classmethod
    def validate_mc(cls, v):
        if not 10 <= v <= 200:
            raise ValueError("n_monte_carlo must be between 10 and 200")
        return v


class ManeuverRequest(BaseModel):
    satellite_name: Optional[str] = None
    norad_id: Optional[int] = None
    debris_norad_id: int
    target_miss_km: float = 5.0

    @field_validator("target_miss_km")
    @classmethod
    def validate_miss(cls, v):
        if not 0.1 <= v <= 100.0:
            raise ValueError("target_miss_km must be between 0.1 and 100.0")
        return v


class InterpretRequest(BaseModel):
    satellite_name: Optional[str] = None
    norad_id: Optional[int] = None
    debris_norad_id: int


class SatelliteItem(BaseModel):
    norad_id: int
    name: str
    altitude_km: float
    inclination_deg: float
    source: str
    line1: Optional[str] = None
    line2: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    catalog_size: int
    version: str
    timestamp: str
    device: str
    catalog_age_hours: Optional[float] = None
    catalog_state: str = "unavailable"
    screening_available: bool = False
    detail: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}}, tags=["System"])
async def health_check():
    """Report service readiness separately from catalog screening safety."""
    loaded = predictor is not None and predictor.loaded
    state = _catalog_state(loaded_catalog_age_hours)
    screening_available = loaded and catalog_screening_available
    payload = HealthResponse(
        status=("ok" if state == "fresh" else "degraded") if loaded else predictor_state,
        model_loaded=loaded,
        catalog_size=len(predictor.catalog) if predictor else 0,
        version="1.0.0",
        timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        device=str(predictor.device) if predictor else "none",
        catalog_age_hours=round(loaded_catalog_age_hours, 2) if loaded_catalog_age_hours is not None else None,
        catalog_state=state,
        screening_available=screening_available,
        detail=(
            (predictor_error if predictor_state == "error" else _predictor_unavailable_detail())
            if not loaded
            else _catalog_detail(loaded_catalog_age_hours)
        ),
    )
    if not loaded:
        return JSONResponse(
            status_code=503,
            content=payload.model_dump(),
            headers={"Retry-After": "5"},
        )
    return payload


@app.post("/predict", tags=["Prediction"])
async def predict(request: PredictRequest, req: Request):
    """
    Predict collision risks for a given satellite.

    Provide either `satellite_name` (e.g., "ISS", "STARLINK-1234") or
    `norad_id` (e.g., 25544). Returns top-N debris threats ranked by
    collision risk with uncertainty estimates.

    - **satellite_name**: Satellite name (partial match supported)
    - **norad_id**: NORAD ID (integer)
    - **top_n**: Number of threats to return (1–50, default 10)
    """
    active_predictor = _require_screening_predictor()

    if request.satellite_name is None and request.norad_id is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'satellite_name' or 'norad_id'"
        )

    try:
        result = active_predictor.predict(
            satellite_name=request.satellite_name,
            norad_id=request.norad_id,
            top_n=request.top_n,
        )
    except Exception as e:
        log.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.post("/predict/detailed", tags=["Prediction"])
async def predict_detailed(request: DetailedPredictRequest):
    """
    Detailed prediction with temporal risk profiles, B-plane geometry,
    and Monte Carlo conjunction analysis.
    """
    active_predictor = _require_screening_predictor()

    if request.satellite_name is None and request.norad_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'satellite_name' or 'norad_id'")

    try:
        result = active_predictor.predict_detailed(
            satellite_name=request.satellite_name,
            norad_id=request.norad_id,
            top_n=request.top_n,
            n_monte_carlo=request.n_monte_carlo,
        )
    except Exception as e:
        log.exception(f"Detailed prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _sanitize(result)


@app.post("/maneuver", tags=["Analysis"])
async def compute_maneuver(request: ManeuverRequest):
    """
    Compute collision avoidance maneuver options (R/S/W directions)
    using Clohessy-Wiltshire linearized relative motion equations.
    """
    active_predictor = _require_screening_predictor()

    if request.satellite_name is None and request.norad_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'satellite_name' or 'norad_id'")

    try:
        result = active_predictor.compute_maneuver(
            satellite_name=request.satellite_name,
            norad_id=request.norad_id,
            debris_norad_id=request.debris_norad_id,
            target_miss_km=request.target_miss_km,
        )
    except Exception as e:
        log.exception(f"Maneuver computation error: {e}")
        raise HTTPException(status_code=500, detail="Internal computation error")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _sanitize(result)


@app.post("/interpret", tags=["Analysis"])
async def interpret_prediction(request: InterpretRequest):
    """
    Model interpretability: attention weights, gradient-based feature importance,
    and ensemble predictions from multiple checkpoints.
    """
    active_predictor = _require_screening_predictor()

    if request.satellite_name is None and request.norad_id is None:
        raise HTTPException(status_code=422, detail="Provide either 'satellite_name' or 'norad_id'")

    try:
        result = active_predictor.interpret(
            satellite_name=request.satellite_name,
            norad_id=request.norad_id,
            debris_norad_id=request.debris_norad_id,
        )
    except Exception as e:
        log.exception(f"Interpret error: {e}")
        raise HTTPException(status_code=500, detail="Internal interpretation error")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return _sanitize(result)


@app.get("/satellites", response_model=list[SatelliteItem], tags=["Catalog"])
async def list_satellites(
    search: Optional[str] = Query(default=None, max_length=100, description="Filter by name"),
    limit:  int           = Query(default=100, ge=1, le=5000, description="Max results"),
    include_tle: bool     = Query(default=False, description="Include TLE data"),
    kind: Optional[Literal["active", "debris"]] = Query(
        default=None,
        description="Limit results to active spacecraft or debris/inactive objects",
    ),
):
    """
    List all trackable satellites in the catalog.
    Optionally filter by name substring.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Catalog not loaded")

    satellites = predictor.list_satellites(
        limit=limit,
        search=search,
        include_tle=include_tle,
        kind=kind,
    )
    return satellites


@app.get("/satellite/{norad_id}", tags=["Catalog"])
async def get_satellite(norad_id: int):
    """
    Get detailed information about a specific satellite by NORAD ID.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Catalog not loaded")

    sat = predictor.find_satellite(norad_id=norad_id)
    if sat is None:
        raise HTTPException(status_code=404, detail=f"Satellite with NORAD ID {norad_id} not found")

    # Return all available fields
    return {
        "norad_id": int(sat["norad_id"]),
        "name": str(sat["name"]),
        "inclination_deg": round(float(sat.get("inclination_deg", 0)), 5),
        "raan_deg": round(float(sat.get("raan_deg", 0)), 5),
        "eccentricity": round(float(sat.get("eccentricity", 0)), 7),
        "arg_perigee_deg": round(float(sat.get("arg_perigee_deg", 0)), 5),
        "mean_anomaly_deg": round(float(sat.get("mean_anomaly_deg", 0)), 5),
        "mean_motion_rev_per_day": round(float(sat.get("mean_motion_rev_per_day", 0)), 6),
        "semi_major_axis_km": round(float(sat.get("semi_major_axis_km", 0)), 3),
        "altitude_km": round(float(sat.get("altitude_km", 0)), 1),
        "orbital_period_min": round(float(sat.get("orbital_period_min", 0)), 2),
        "bstar_drag": float(sat.get("bstar_drag", 0)),
        "source": str(sat.get("source", "unknown")),
        "line1": str(sat.get("line1", "")),
        "line2": str(sat.get("line2", "")),
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"},
    )


# ---------------------------------------------------------------------------
# Serve frontend static files (production / Railway deployment)
# ---------------------------------------------------------------------------
_FRONTEND_DIST = _ROOT / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse, Response

    @app.head("/", include_in_schema=False)
    async def head_frontend():
        """Allow uptime checks and link validators to probe the homepage."""
        return Response(status_code=200, media_type="text/html")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend SPA — all non-API routes return index.html."""
        file_path = _FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")

    log.info(f"Serving frontend from {_FRONTEND_DIST}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
