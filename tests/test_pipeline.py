"""Tests for the training + inference pipeline wiring."""

from __future__ import annotations

import json

import pandas as pd

from pipeline.inference import RANInferenceService
from pipeline.trainer import train_all


def _train_into(tmp_path, sample_df: pd.DataFrame):
    data_path = tmp_path / "data.csv"
    model_dir = tmp_path / "models"
    sample_df.to_csv(data_path, index=False)
    metrics = train_all(data_path=data_path, model_dir=model_dir)
    return data_path, model_dir, metrics


def test_train_all_produces_artifacts_and_metrics(tmp_path, sample_df: pd.DataFrame) -> None:
    _, model_dir, metrics = _train_into(tmp_path, sample_df)

    for name in ("qos_model.joblib", "beam_model.joblib", "anomaly_model.joblib"):
        assert (model_dir / name).exists()

    saved = json.loads((model_dir / "training_metrics.json").read_text())
    assert saved["qos_model"]["accuracy"] == metrics["qos_model"]["accuracy"]
    assert "accuracy" in metrics["beam_model"]


def test_train_all_generates_data_when_missing(tmp_path) -> None:
    # No CSV at the target path -> trainer should synthesize one.
    data_path = tmp_path / "auto.csv"
    model_dir = tmp_path / "models"
    train_all(data_path=data_path, model_dir=model_dir)
    assert data_path.exists()


def test_inference_service_end_to_end(tmp_path, sample_df: pd.DataFrame) -> None:
    _, model_dir, _ = _train_into(tmp_path, sample_df)
    service = RANInferenceService(model_dir=model_dir)
    row = sample_df.iloc[0].to_dict()

    qos = service.predict_qos({k: row[k] for k in service.qos.config.feature_columns})
    beam = service.select_beam({k: row[k] for k in service.beam.config.feature_columns})
    anomaly = service.detect_anomaly({k: row[k] for k in service.anomaly.config.feature_columns})

    assert qos["qos_class"] in {"good", "medium", "poor"}
    assert isinstance(beam["optimal_beam_index"], int)
    assert isinstance(anomaly["is_anomaly"], bool)
