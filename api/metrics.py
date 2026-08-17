"""Minimal in-process request metrics.

A dependency-free counter/latency tracker so the API can expose basic
observability without pulling in Prometheus. For production, swap this for
``prometheus-client`` and a ``/metrics`` exposition endpoint.
"""

from __future__ import annotations

import threading
from collections import defaultdict


class MetricsRegistry:
    """Thread-safe counters and latency accumulators keyed by route."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._latency_ms_sum: dict[str, float] = defaultdict(float)

    def observe(self, route: str, latency_ms: float, is_error: bool) -> None:
        with self._lock:
            self._requests[route] += 1
            self._latency_ms_sum[route] += latency_ms
            if is_error:
                self._errors[route] += 1

    def snapshot(self) -> dict:
        with self._lock:
            routes = sorted(self._requests)
            total = sum(self._requests.values())
            return {
                "total_requests": total,
                "total_errors": sum(self._errors.values()),
                "routes": {
                    route: {
                        "requests": self._requests[route],
                        "errors": self._errors[route],
                        "avg_latency_ms": round(
                            self._latency_ms_sum[route] / self._requests[route], 3
                        ),
                    }
                    for route in routes
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latency_ms_sum.clear()


registry = MetricsRegistry()
