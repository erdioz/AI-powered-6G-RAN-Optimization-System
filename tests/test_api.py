"""End-to-end tests for the FastAPI serving layer.

The app trains models at import time, so we point the project paths at a
temporary directory pre-seeded with a small dataset before importing it.
"""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

import ran6g.config as config


@pytest.fixture(scope="module")
def client(tmp_path_factory, request):
    tmp = tmp_path_factory.mktemp("api")
    data_path = tmp / "data.csv"
    model_dir = tmp / "models"

    # Build a small dataset so import-time training stays fast.
    from data.generator import GenerationConfig, SyntheticRANDataGenerator

    df = SyntheticRANDataGenerator(GenerationConfig(num_users=8, time_steps=15)).generate()
    df.to_csv(data_path, index=False)

    old_env = {k: os.environ.get(k) for k in ("RAN6G_DATA_PATH", "RAN6G_MODEL_DIR")}
    os.environ["RAN6G_DATA_PATH"] = str(data_path)
    os.environ["RAN6G_MODEL_DIR"] = str(model_dir)
    config.get_paths.cache_clear()

    import api.app as app_module

    app_module = importlib.reload(app_module)
    test_client = TestClient(app_module.app)

    yield test_client, df

    # Restore environment for other tests.
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    config.get_paths.cache_clear()


def test_health(client) -> None:
    test_client, _ = client
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert all(body["models_loaded"].values())


def test_predict_qos(client) -> None:
    test_client, df = client
    row = df.iloc[0]
    payload = {
        "rsrp": float(row["rsrp"]), "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "distance_to_cell": float(row["distance_to_cell"]), "beam_index": int(row["beam_index"]),
        "interference_level": float(row["interference_level"]), "speed": float(row["speed"]),
    }
    resp = test_client.post("/predict_qos", json=payload)
    assert resp.status_code == 200
    assert resp.json()["qos_class"] in {"good", "medium", "poor"}


def test_select_beam(client) -> None:
    test_client, df = client
    row = df.iloc[0]
    payload = {
        "x": float(row["x"]), "y": float(row["y"]), "speed": float(row["speed"]),
        "distance_to_cell": float(row["distance_to_cell"]),
        "azimuth_to_cell": float(row["azimuth_to_cell"]), "rsrp": float(row["rsrp"]),
        "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "interference_level": float(row["interference_level"]), "cell_id": int(row["cell_id"]),
    }
    resp = test_client.post("/select_beam", json=payload)
    assert resp.status_code == 200
    assert isinstance(resp.json()["optimal_beam_index"], int)


def test_detect_anomaly(client) -> None:
    test_client, df = client
    row = df.iloc[0]
    payload = {
        "rsrp": float(row["rsrp"]), "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "interference_level": float(row["interference_level"]),
        "distance_to_cell": float(row["distance_to_cell"]), "speed": float(row["speed"]),
        "throughput_mbps": float(row["throughput_mbps"]), "latency_ms": float(row["latency_ms"]),
    }
    resp = test_client.post("/detect_anomaly", json=payload)
    assert resp.status_code == 200
    assert "is_anomaly" in resp.json()


def test_validation_rejects_bad_input(client) -> None:
    test_client, _ = client
    resp = test_client.post("/predict_qos", json={"rsrp": "not-a-number"})
    assert resp.status_code == 422
