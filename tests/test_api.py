"""End-to-end tests for the FastAPI serving layer.

Models load during the app lifespan, so the ``TestClient`` is entered as a
context manager to trigger startup. Project paths are pointed at a temporary
directory pre-seeded with a small dataset to keep startup training fast.
"""

from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

import ran6g.config as config


def _qos_payload(row) -> dict:
    return {
        "rsrp": float(row["rsrp"]), "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "distance_to_cell": float(row["distance_to_cell"]), "beam_index": int(row["beam_index"]),
        "interference_level": float(row["interference_level"]), "speed": float(row["speed"]),
    }


def _beam_payload(row) -> dict:
    return {
        "x": float(row["x"]), "y": float(row["y"]), "speed": float(row["speed"]),
        "distance_to_cell": float(row["distance_to_cell"]),
        "azimuth_to_cell": float(row["azimuth_to_cell"]), "rsrp": float(row["rsrp"]),
        "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "interference_level": float(row["interference_level"]), "cell_id": int(row["cell_id"]),
    }


def _anomaly_payload(row) -> dict:
    return {
        "rsrp": float(row["rsrp"]), "sinr": float(row["sinr"]), "cqi": int(row["cqi"]),
        "interference_level": float(row["interference_level"]),
        "distance_to_cell": float(row["distance_to_cell"]), "speed": float(row["speed"]),
        "throughput_mbps": float(row["throughput_mbps"]), "latency_ms": float(row["latency_ms"]),
    }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    data_path = tmp / "data.csv"
    model_dir = tmp / "models"

    from data.generator import GenerationConfig, SyntheticRANDataGenerator

    df = SyntheticRANDataGenerator(GenerationConfig(num_users=8, time_steps=15)).generate()
    df.to_csv(data_path, index=False)

    old_env = {k: os.environ.get(k) for k in ("RAN6G_DATA_PATH", "RAN6G_MODEL_DIR")}
    os.environ["RAN6G_DATA_PATH"] = str(data_path)
    os.environ["RAN6G_MODEL_DIR"] = str(model_dir)
    config.get_paths.cache_clear()

    import api.app as app_module

    app_module = importlib.reload(app_module)
    with TestClient(app_module.app) as test_client:  # `with` triggers lifespan startup
        yield test_client, df

    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    config.get_paths.cache_clear()


def test_health(client) -> None:
    test_client, _ = client
    body = test_client.get("/health").json()
    assert body["status"] == "ok"
    assert all(body["models_loaded"].values())


def test_model_info(client) -> None:
    test_client, _ = client
    resp = test_client.get("/model_info")
    assert resp.status_code == 200
    info = resp.json()
    assert "beam_model" in info
    assert info["metadata"]["num_rows"] > 0


def test_predict_qos(client) -> None:
    test_client, df = client
    resp = test_client.post("/predict_qos", json=_qos_payload(df.iloc[0]))
    assert resp.status_code == 200
    assert resp.json()["qos_class"] in {"good", "medium", "poor"}


def test_select_beam(client) -> None:
    test_client, df = client
    resp = test_client.post("/select_beam", json=_beam_payload(df.iloc[0]))
    assert resp.status_code == 200
    assert isinstance(resp.json()["optimal_beam_index"], int)


def test_detect_anomaly(client) -> None:
    test_client, df = client
    resp = test_client.post("/detect_anomaly", json=_anomaly_payload(df.iloc[0]))
    assert resp.status_code == 200
    assert "is_anomaly" in resp.json()


def test_predict_qos_batch(client) -> None:
    test_client, df = client
    payload = [_qos_payload(df.iloc[i]) for i in range(5)]
    resp = test_client.post("/predict_qos/batch", json=payload)
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 5
    assert all(r["qos_class"] in {"good", "medium", "poor"} for r in results)


def test_select_beam_batch(client) -> None:
    test_client, df = client
    payload = [_beam_payload(df.iloc[i]) for i in range(5)]
    resp = test_client.post("/select_beam/batch", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5


def test_detect_anomaly_batch(client) -> None:
    test_client, df = client
    payload = [_anomaly_payload(df.iloc[i]) for i in range(5)]
    resp = test_client.post("/detect_anomaly/batch", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 5


def test_empty_batch_rejected(client) -> None:
    test_client, _ = client
    resp = test_client.post("/predict_qos/batch", json=[])
    assert resp.status_code == 400


def test_metrics_endpoint_tracks_requests(client) -> None:
    test_client, df = client
    test_client.post("/predict_qos", json=_qos_payload(df.iloc[0]))
    snap = test_client.get("/metrics").json()
    assert snap["total_requests"] >= 1
    assert "/predict_qos" in snap["routes"]


def test_validation_rejects_bad_input(client) -> None:
    test_client, _ = client
    resp = test_client.post("/predict_qos", json={"rsrp": "not-a-number"})
    assert resp.status_code == 422


def test_validation_rejects_out_of_range_cqi(client) -> None:
    test_client, df = client
    payload = _qos_payload(df.iloc[0])
    payload["cqi"] = 99  # exceeds le=15
    resp = test_client.post("/predict_qos", json=payload)
    assert resp.status_code == 422
