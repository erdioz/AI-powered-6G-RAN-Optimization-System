"""Tests for the in-process metrics registry."""

from __future__ import annotations

from api.metrics import MetricsRegistry


def test_registry_counts_and_averages() -> None:
    reg = MetricsRegistry()
    reg.observe("/predict_qos", 10.0, is_error=False)
    reg.observe("/predict_qos", 20.0, is_error=True)
    snap = reg.snapshot()

    assert snap["total_requests"] == 2
    assert snap["total_errors"] == 1
    route = snap["routes"]["/predict_qos"]
    assert route["requests"] == 2
    assert route["errors"] == 1
    assert route["avg_latency_ms"] == 15.0


def test_registry_reset() -> None:
    reg = MetricsRegistry()
    reg.observe("/health", 1.0, is_error=False)
    reg.reset()
    snap = reg.snapshot()
    assert snap["total_requests"] == 0
    assert snap["routes"] == {}
