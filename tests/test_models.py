"""Tests for the QoS, beam, and anomaly models: train/predict/persist."""

from __future__ import annotations

import pandas as pd

from models.anomaly_model import RadioAnomalyDetector
from models.beam_model import BeamSelector
from models.qos_model import QoSPredictor


def _feature_row(df: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    return {c: df.iloc[0][c] for c in columns}


def test_qos_train_and_predict(sample_df: pd.DataFrame) -> None:
    model = QoSPredictor()
    report = model.train(sample_df)
    assert "accuracy" in report
    pred = model.predict(_feature_row(sample_df, model.config.feature_columns))
    assert pred in {"good", "medium", "poor"}


def test_qos_save_load_roundtrip(tmp_path, sample_df: pd.DataFrame) -> None:
    model = QoSPredictor()
    model.train(sample_df)
    path = tmp_path / "qos.joblib"
    model.save(str(path))
    loaded = QoSPredictor.load(str(path))
    row = _feature_row(sample_df, model.config.feature_columns)
    assert loaded.predict(row) == model.predict(row)


def test_beam_train_and_predict(sample_df: pd.DataFrame) -> None:
    model = BeamSelector()
    metrics = model.train(sample_df)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    pred = model.predict(_feature_row(sample_df, model.config.feature_columns))
    assert isinstance(pred, int)


def test_beam_save_load_roundtrip(tmp_path, sample_df: pd.DataFrame) -> None:
    model = BeamSelector()
    model.train(sample_df)
    path = tmp_path / "beam.joblib"
    model.save(str(path))
    loaded = BeamSelector.load(str(path))
    row = _feature_row(sample_df, model.config.feature_columns)
    assert loaded.predict(row) == model.predict(row)


def test_anomaly_train_and_predict(sample_df: pd.DataFrame) -> None:
    model = RadioAnomalyDetector()
    model.train(sample_df)
    result = model.predict(_feature_row(sample_df, model.config.feature_columns))
    assert set(result) == {"is_anomaly", "anomaly_score"}
    assert isinstance(result["is_anomaly"], bool)


def test_anomaly_save_load_roundtrip(tmp_path, sample_df: pd.DataFrame) -> None:
    model = RadioAnomalyDetector()
    model.train(sample_df)
    path = tmp_path / "anomaly.joblib"
    model.save(str(path))
    loaded = RadioAnomalyDetector.load(str(path))
    row = _feature_row(sample_df, model.config.feature_columns)
    assert loaded.predict(row)["is_anomaly"] == model.predict(row)["is_anomaly"]
