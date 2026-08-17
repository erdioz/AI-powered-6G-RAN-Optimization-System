"""Tests for the map visualization helpers (site recovery + rendering)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering for CI

import numpy as np  # noqa: E402

from data.generator import GenerationConfig, SyntheticRANDataGenerator  # noqa: E402
from visualization.plots import infer_num_beams, plot_ue_movement, recover_sites  # noqa: E402


def test_infer_num_beams(sample_df) -> None:
    assert infer_num_beams(sample_df) == 8


def test_recover_sites_matches_true_positions() -> None:
    # Recovered site positions should match the environment's base stations.
    gen = SyntheticRANDataGenerator(GenerationConfig(num_users=8, time_steps=15))
    df = gen.generate()
    recovered = recover_sites(df)
    truth = {bs.cell_id: (bs.x, bs.y) for bs in gen.environment.base_stations}

    assert set(recovered) == set(truth)
    for cell_id, (tx, ty) in truth.items():
        rx, ry = recovered[cell_id]
        assert np.isclose(rx, tx, atol=1e-6)
        assert np.isclose(ry, ty, atol=1e-6)


def test_recover_sites_missing_columns_returns_empty(sample_df) -> None:
    assert recover_sites(sample_df.drop(columns=["azimuth_to_cell"])) == {}


def test_plot_ue_movement_writes_file(tmp_path, sample_df) -> None:
    out = plot_ue_movement(sample_df, user_id=0, save_path=str(tmp_path / "map.png"))
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_ue_movement_without_sectors(tmp_path, sample_df) -> None:
    out = plot_ue_movement(
        sample_df, user_id=0, save_path=str(tmp_path / "nosectors.png"), show_sectors=False
    )
    assert out.exists()
