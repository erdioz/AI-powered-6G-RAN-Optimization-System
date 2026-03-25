"""QoS classification model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@dataclass
class QoSModelConfig:
    feature_columns: tuple[str, ...] = (
        "rsrp",
        "sinr",
        "cqi",
        "distance_to_cell",
        "beam_index",
        "interference_level",
        "speed",
    )
    label_column: str = "qos_class"


class QoSPredictor:
    """RandomForest-based QoS class predictor."""

    def __init__(self, config: QoSModelConfig | None = None) -> None:
        self.config = config or QoSModelConfig()
        self.model = RandomForestClassifier(n_estimators=180, max_depth=14, random_state=42)

    def train(self, df: pd.DataFrame) -> dict:
        X = df[list(self.config.feature_columns)]
        y = df[self.config.label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        return report

    def predict(self, features: dict) -> str:
        sample = pd.DataFrame([features])[list(self.config.feature_columns)]
        return str(self.model.predict(sample)[0])

    def save(self, path: str) -> None:
        joblib.dump({"model": self.model, "config": self.config}, path)

    @classmethod
    def load(cls, path: str) -> "QoSPredictor":
        payload = joblib.load(path)
        instance = cls(config=payload["config"])
        instance.model = payload["model"]
        return instance
