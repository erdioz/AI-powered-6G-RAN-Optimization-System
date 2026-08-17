"""Tests for the synthetic dataset generator."""

from __future__ import annotations

import pandas as pd

from data.generator import GenerationConfig, SyntheticRANDataGenerator

EXPECTED_COLUMNS = {
    "time_step", "user_id", "cell_id", "x", "y", "speed", "distance_to_cell",
    "beam_index", "optimal_beam_index", "beam_gain_db", "rsrp", "sinr", "cqi",
    "interference_level", "noise_floor", "latency_ms", "throughput_mbps",
    "qos_class", "is_anomaly",
}


def test_generate_row_count(small_config: GenerationConfig, sample_df: pd.DataFrame) -> None:
    assert len(sample_df) == small_config.num_users * small_config.time_steps


def test_generate_has_expected_schema(sample_df: pd.DataFrame) -> None:
    assert EXPECTED_COLUMNS.issubset(set(sample_df.columns))


def test_generation_is_deterministic(small_config: GenerationConfig) -> None:
    a = SyntheticRANDataGenerator(small_config).generate()
    b = SyntheticRANDataGenerator(small_config).generate()
    pd.testing.assert_frame_equal(a, b)


def test_qos_classes_are_valid(sample_df: pd.DataFrame) -> None:
    assert set(sample_df["qos_class"].unique()).issubset({"good", "medium", "poor"})


def test_beam_indices_within_range(small_config: GenerationConfig, sample_df: pd.DataFrame) -> None:
    assert sample_df["beam_index"].between(0, small_config.num_beams - 1).all()
    assert sample_df["optimal_beam_index"].between(0, small_config.num_beams - 1).all()


def test_anomaly_flag_binary(sample_df: pd.DataFrame) -> None:
    assert set(sample_df["is_anomaly"].unique()).issubset({0, 1})


def test_to_csv_writes_file(tmp_path, small_config: GenerationConfig) -> None:
    out = tmp_path / "nested" / "dataset.csv"
    result = SyntheticRANDataGenerator(small_config).to_csv(out)
    assert result.exists()
    assert len(pd.read_csv(out)) > 0
