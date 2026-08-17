"""Training pipeline for the 6G RAN optimization system."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.generator import SyntheticRANDataGenerator
from models.anomaly_model import RadioAnomalyDetector
from models.beam_model import BeamSelector
from models.qos_model import QoSPredictor
from ran6g.config import get_paths
from ran6g.logging_utils import get_logger

logger = get_logger("ran6g.trainer")


def train_all(data_path: str | Path | None = None, model_dir: str | Path | None = None) -> dict:
    """Train all models and persist artifacts.

    ``data_path`` and ``model_dir`` default to the project-configured
    locations (overridable via ``RAN6G_*`` environment variables).
    """
    paths = get_paths()
    data_file = Path(data_path) if data_path is not None else paths.data_path
    model_path = Path(model_dir) if model_dir is not None else paths.model_dir
    model_path.mkdir(parents=True, exist_ok=True)

    if not data_file.exists():
        logger.info("Dataset not found at %s; generating synthetic data.", data_file)
        generator = SyntheticRANDataGenerator()
        generator.to_csv(data_file)

    df = pd.read_csv(data_file)
    logger.info("Loaded %d rows for training.", len(df))

    qos_model = QoSPredictor()
    qos_report = qos_model.train(df)
    qos_model.save(str(model_path / "qos_model.joblib"))

    beam_model = BeamSelector()
    beam_metrics = beam_model.train(df)
    beam_model.save(str(model_path / "beam_model.joblib"))

    anomaly_model = RadioAnomalyDetector()
    anomaly_model.train(df)
    anomaly_model.save(str(model_path / "anomaly_model.joblib"))

    metrics = {
        "qos_model": {
            "accuracy": qos_report["accuracy"],
            "macro_f1": qos_report["macro avg"]["f1-score"],
        },
        "beam_model": beam_metrics,
        "anomaly_model": {"status": "trained"},
    }

    with open(model_path / "training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    logger.info("Training complete. Artifacts saved to %s", model_path)
    return metrics


if __name__ == "__main__":
    results = train_all()
    print(json.dumps(results, indent=2))
