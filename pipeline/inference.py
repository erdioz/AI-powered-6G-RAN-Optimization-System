"""Unified inference utilities for trained RAN AI models."""

from __future__ import annotations

from pathlib import Path

from models.anomaly_model import RadioAnomalyDetector
from models.beam_model import BeamSelector
from models.qos_model import QoSPredictor


class RANInferenceService:
    """Load model artifacts and provide single-call prediction APIs."""

    def __init__(self, model_dir: str = "outputs/models") -> None:
        model_path = Path(model_dir)
        self.qos = QoSPredictor.load(str(model_path / "qos_model.joblib"))
        self.beam = BeamSelector.load(str(model_path / "beam_model.joblib"))
        self.anomaly = RadioAnomalyDetector.load(str(model_path / "anomaly_model.joblib"))

    def predict_qos(self, payload: dict) -> dict:
        return {"qos_class": self.qos.predict(payload)}

    def select_beam(self, payload: dict) -> dict:
        return {"optimal_beam_index": self.beam.predict(payload)}

    def detect_anomaly(self, payload: dict) -> dict:
        return self.anomaly.predict(payload)
