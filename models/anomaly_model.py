"""Anomaly detection model module for radio conditions."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyModelConfig:
    feature_columns: tuple[str, ...] = (
        "rsrp",
        "sinr",
        "cqi",
        "interference_level",
        "distance_to_cell",
        "speed",
        "throughput_mbps",
        "latency_ms",
    )


class RadioAnomalyDetector:
    """Isolation Forest detector for abnormal radio behavior."""

    def __init__(self, config: AnomalyModelConfig | None = None) -> None:
        self.config = config or AnomalyModelConfig()
        self.model = IsolationForest(n_estimators=200, contamination=0.06, random_state=42)

    def train(self, df: pd.DataFrame) -> None:
        X = df[list(self.config.feature_columns)]
        self.model.fit(X)

    def predict(self, features: dict) -> dict:
        sample = pd.DataFrame([features])[list(self.config.feature_columns)]
        pred = int(self.model.predict(sample)[0])
        score = float(self.model.decision_function(sample)[0])
        return {"is_anomaly": pred == -1, "anomaly_score": score}

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """Vectorized prediction over many feature rows in a single call."""
        frame = pd.DataFrame(rows)[list(self.config.feature_columns)]
        preds = self.model.predict(frame)
        scores = self.model.decision_function(frame)
        return [
            {"is_anomaly": int(preds[i]) == -1, "anomaly_score": float(scores[i])}
            for i in range(len(preds))
        ]

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> RadioAnomalyDetector:
        payload = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.model = payload["model"]
        return instance
