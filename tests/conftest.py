"""Shared pytest fixtures for the 6G RAN optimization test suite."""

from __future__ import annotations

import pandas as pd
import pytest

from data.generator import GenerationConfig, SyntheticRANDataGenerator


@pytest.fixture(scope="session")
def small_config() -> GenerationConfig:
    """A small, fast configuration for tests."""
    return GenerationConfig(num_users=8, num_cells=3, num_beams=8, time_steps=15, random_seed=13)


@pytest.fixture(scope="session")
def sample_df(small_config: GenerationConfig) -> pd.DataFrame:
    """A small deterministic synthetic dataset reused across tests."""
    return SyntheticRANDataGenerator(small_config).generate()
