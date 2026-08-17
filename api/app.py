"""FastAPI application exposing RAN optimization endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline.inference import RANInferenceService
from pipeline.trainer import train_all
from ran6g.config import get_paths
from ran6g.logging_utils import get_logger

logger = get_logger("ran6g.api")


class QoSInput(BaseModel):
    rsrp: float
    sinr: float
    cqi: int
    distance_to_cell: float
    beam_index: int
    interference_level: float
    speed: float


class BeamInput(BaseModel):
    x: float
    y: float
    speed: float
    distance_to_cell: float
    azimuth_to_cell: float
    rsrp: float
    sinr: float
    cqi: int
    interference_level: float
    cell_id: int


class AnomalyInput(BaseModel):
    rsrp: float
    sinr: float
    cqi: int
    interference_level: float
    distance_to_cell: float
    speed: float
    throughput_mbps: float
    latency_ms: float


app = FastAPI(title="AI-powered 6G RAN Optimization API", version="1.0.0")

model_dir = get_paths().model_dir
if not model_dir.exists() or not (model_dir / "qos_model.joblib").exists():
    logger.info("Model artifacts missing at %s; training on startup.", model_dir)
    train_all()

service = RANInferenceService(model_dir=model_dir)


@app.get("/health")
def health() -> dict:
    """Report service liveness and which model artifacts are loaded."""
    return {
        "status": "ok",
        "model_dir": str(model_dir),
        "models_loaded": {
            "qos": service.qos is not None,
            "beam": service.beam is not None,
            "anomaly": service.anomaly is not None,
        },
    }


@app.post("/predict_qos")
def predict_qos(payload: QoSInput) -> dict:
    try:
        return service.predict_qos(payload.model_dump())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/select_beam")
def select_beam(payload: BeamInput) -> dict:
    try:
        return service.select_beam(payload.model_dump())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/detect_anomaly")
def detect_anomaly(payload: AnomalyInput) -> dict:
    try:
        return service.detect_anomaly(payload.model_dump())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=str(exc)) from exc
