"""Tests for the deterministic radio-channel physics helpers."""

from __future__ import annotations

import math

import pytest

from data.radio_channel import RadioChannel, RadioParams


@pytest.fixture
def channel() -> RadioChannel:
    return RadioChannel()


def test_path_loss_increases_with_distance(channel: RadioChannel) -> None:
    assert channel.path_loss_db(500) < channel.path_loss_db(2000)


def test_path_loss_handles_zero_distance(channel: RadioChannel) -> None:
    # Minimum-distance clamp must keep the result finite.
    value = channel.path_loss_db(0.0)
    assert math.isfinite(value)


def test_rsrp_decreases_with_distance(channel: RadioChannel) -> None:
    near = channel.rsrp_dbm(100, beam_gain_db=10, shadowing_db=0)
    far = channel.rsrp_dbm(3000, beam_gain_db=10, shadowing_db=0)
    assert near > far


def test_sinr_monotonic_in_interference(channel: RadioChannel) -> None:
    low = channel.sinr_db(signal_dbm=-70, interference_dbm=-110)
    high = channel.sinr_db(signal_dbm=-70, interference_dbm=-80)
    assert low > high


@pytest.mark.parametrize(
    "sinr,expected",
    [(30.0, "good"), (10.0, "medium"), (-5.0, "poor")],
)
def test_qos_class_thresholds(sinr: float, expected: str) -> None:
    assert RadioChannel.qos_class_from_sinr(sinr) == expected


def test_sinr_to_cqi_bounds() -> None:
    assert RadioChannel.sinr_to_cqi(-100) == 1
    assert RadioChannel.sinr_to_cqi(100) == 15
    assert 1 <= RadioChannel.sinr_to_cqi(5.0) <= 15


def test_throughput_increases_with_sinr() -> None:
    assert RadioChannel.throughput_mbps_from_sinr(20) > RadioChannel.throughput_mbps_from_sinr(0)


def test_latency_has_floor_and_decreases_with_sinr() -> None:
    assert RadioChannel.latency_ms_from_sinr(100) == 1.0  # floored
    assert RadioChannel.latency_ms_from_sinr(0) > RadioChannel.latency_ms_from_sinr(10)


def test_custom_params_respected() -> None:
    params = RadioParams(noise_floor_dbm=-90.0)
    channel = RadioChannel(params)
    assert channel.params.noise_floor_dbm == -90.0
