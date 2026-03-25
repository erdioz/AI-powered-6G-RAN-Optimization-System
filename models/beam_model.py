"""Beam selection model module."""

from __future__ import annotations

from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class BeamModelConfig:
    feature_columns: tuple[str, ...] = (
        "x",
        "y",
        "speed",
        "distance_to_cell",
        "rsrp",
        "sinr",
        "cqi",
        "interference_level",
        "cell_id",
    )
    label_column: str = "optimal_beam_index"


class BeamSelector:
    """Multi-class model predicting optimal beam index."""

    def __init__(self, config: BeamModelConfig | None = None) -> None:
        self.config = config or BeamModelConfig()
        self.model = RandomForestClassifier(n_estimators=220, max_depth=16, random_state=42)

    def train(self, df: pd.DataFrame) -> dict:
        X = df[list(self.config.feature_columns)]
        y = df[self.config.label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        return {"accuracy": float(accuracy_score(y_test, preds))}

    def predict(self, features: dict) -> int:
        sample = pd.DataFrame([features])[list(self.config.feature_columns)]
        return int(self.model.predict(sample)[0])

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "BeamSelector":
        payload = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.model = payload["model"]
        return instance
