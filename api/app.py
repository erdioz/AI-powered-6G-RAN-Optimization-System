"""FastAPI application exposing RAN optimization endpoints.

Models are loaded once during the application lifespan (not at import time), so
importing this module is cheap and side-effect free. If no artifacts are found,
they are trained on startup.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from api.metrics import registry
from pipeline.inference import RANInferenceService
from pipeline.trainer import train_all
from ran6g.config import get_paths
from ran6g.logging_utils import get_logger

logger = get_logger("ran6g.api")

MAX_BATCH_SIZE = 1024


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class QoSInput(BaseModel):
    rsrp: float = Field(..., examples=[-70.0])
    sinr: float = Field(..., examples=[12.5])
    cqi: int = Field(..., ge=0, le=15, examples=[10])
    distance_to_cell: float = Field(..., ge=0, examples=[250.0])
    beam_index: int = Field(..., ge=0, examples=[3])
    interference_level: float = Field(..., examples=[-100.0])
    speed: float = Field(..., ge=0, examples=[5.0])


class BeamInput(BaseModel):
    x: float = Field(..., examples=[400.0])
    y: float = Field(..., examples=[600.0])
    speed: float = Field(..., ge=0, examples=[5.0])
    distance_to_cell: float = Field(..., ge=0, examples=[250.0])
    azimuth_to_cell: float = Field(..., ge=-3.1416, le=3.1416, examples=[1.2])
    rsrp: float = Field(..., examples=[-70.0])
    sinr: float = Field(..., examples=[12.5])
    cqi: int = Field(..., ge=0, le=15, examples=[10])
    interference_level: float = Field(..., examples=[-100.0])
    cell_id: int = Field(..., ge=0, examples=[1])


class AnomalyInput(BaseModel):
    rsrp: float = Field(..., examples=[-70.0])
    sinr: float = Field(..., examples=[12.5])
    cqi: int = Field(..., ge=0, le=15, examples=[10])
    interference_level: float = Field(..., examples=[-100.0])
    distance_to_cell: float = Field(..., ge=0, examples=[250.0])
    speed: float = Field(..., ge=0, examples=[5.0])
    throughput_mbps: float = Field(..., ge=0, examples=[350.0])
    latency_ms: float = Field(..., ge=0, examples=[8.0])


# --------------------------------------------------------------------------- #
# Lifespan: load (or train) models once at startup
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = get_paths().model_dir
    if not model_dir.exists() or not (model_dir / "qos_model.joblib").exists():
        logger.info("Model artifacts missing at %s; training on startup.", model_dir)
        train_all()
    app.state.model_dir = model_dir
    app.state.service = RANInferenceService(model_dir=model_dir)
    logger.info("Inference service ready (models from %s).", model_dir)
    yield
    app.state.service = None


app = FastAPI(title="AI-powered 6G RAN Optimization API", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    """Record per-route request counts, error counts, and average latency."""
    start = time.perf_counter()
    is_error = False
    try:
        response = await call_next(request)
    except Exception:
        is_error = True
        raise
    else:
        is_error = response.status_code >= 500
        return response
    finally:
        route = request.scope.get("route")
        label = getattr(route, "path", request.url.path)
        registry.observe(label, (time.perf_counter() - start) * 1000.0, is_error)


def get_service(request: Request) -> RANInferenceService:
    """Return the loaded inference service or 503 if it is not ready."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")
    return service


ServiceDep = Annotated[RANInferenceService, Depends(get_service)]


def _check_batch(items: list) -> None:
    if not items:
        raise HTTPException(status_code=400, detail="Batch must contain at least one item.")
    if len(items) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail=f"Batch exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}.")


# --------------------------------------------------------------------------- #
# Operational endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health(request: Request) -> dict:
    """Report service liveness and which model artifacts are loaded."""
    service = getattr(request.app.state, "service", None)
    return {
        "status": "ok" if service is not None else "starting",
        "model_dir": str(getattr(request.app.state, "model_dir", "")),
        "models_loaded": {
            "qos": service is not None and service.qos is not None,
            "beam": service is not None and service.beam is not None,
            "anomaly": service is not None and service.anomaly is not None,
        },
    }


@app.get("/model_info")
def model_info(request: Request) -> dict:
    """Return training metrics/metadata for the currently loaded models."""
    model_dir = getattr(request.app.state, "model_dir", get_paths().model_dir)
    metrics_file = model_dir / "training_metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="training_metrics.json not found.")
    return json.loads(metrics_file.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics() -> dict:
    """Return in-process request metrics (counts, errors, average latency)."""
    return registry.snapshot()


# --------------------------------------------------------------------------- #
# Single-item prediction endpoints
# --------------------------------------------------------------------------- #
@app.post("/predict_qos")
def predict_qos(payload: QoSInput, service: ServiceDep) -> dict:
    return service.predict_qos(payload.model_dump())


@app.post("/select_beam")
def select_beam(payload: BeamInput, service: ServiceDep) -> dict:
    return service.select_beam(payload.model_dump())


@app.post("/detect_anomaly")
def detect_anomaly(payload: AnomalyInput, service: ServiceDep) -> dict:
    return service.detect_anomaly(payload.model_dump())


# --------------------------------------------------------------------------- #
# Batch prediction endpoints (vectorized)
# --------------------------------------------------------------------------- #
@app.post("/predict_qos/batch")
def predict_qos_batch(payload: list[QoSInput], service: ServiceDep) -> dict:
    _check_batch(payload)
    return {"results": service.predict_qos_batch([p.model_dump() for p in payload])}


@app.post("/select_beam/batch")
def select_beam_batch(payload: list[BeamInput], service: ServiceDep) -> dict:
    _check_batch(payload)
    return {"results": service.select_beam_batch([p.model_dump() for p in payload])}


@app.post("/detect_anomaly/batch")
def detect_anomaly_batch(payload: list[AnomalyInput], service: ServiceDep) -> dict:
    _check_batch(payload)
    return {"results": service.detect_anomaly_batch([p.model_dump() for p in payload])}
