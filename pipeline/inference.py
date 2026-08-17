"""Unified inference utilities for trained RAN AI models."""

from __future__ import annotations

from pathlib import Path

from models.anomaly_model import RadioAnomalyDetector
from models.beam_model import BeamSelector
from models.qos_model import QoSPredictor
from ran6g.config import get_paths


class RANInferenceService:
    """Load model artifacts and provide single-call prediction APIs."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        model_path = Path(model_dir) if model_dir is not None else get_paths().model_dir
        self.qos = QoSPredictor.load(str(model_path / "qos_model.joblib"))
        self.beam = BeamSelector.load(str(model_path / "beam_model.joblib"))
        self.anomaly = RadioAnomalyDetector.load(str(model_path / "anomaly_model.joblib"))

    def predict_qos(self, payload: dict) -> dict:
        return {"qos_class": self.qos.predict(payload)}

    def select_beam(self, payload: dict) -> dict:
        return {"optimal_beam_index": self.beam.predict(payload)}

    def detect_anomaly(self, payload: dict) -> dict:
        return self.anomaly.predict(payload)

    def predict_qos_batch(self, payloads: list[dict]) -> list[dict]:
        return [{"qos_class": c} for c in self.qos.predict_batch(payloads)]

    def select_beam_batch(self, payloads: list[dict]) -> list[dict]:
        return [{"optimal_beam_index": b} for b in self.beam.predict_batch(payloads)]

    def detect_anomaly_batch(self, payloads: list[dict]) -> list[dict]:
        return self.anomaly.predict_batch(payloads)
