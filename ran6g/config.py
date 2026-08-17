"""Centralized path configuration.

Paths are resolved relative to the project root by default but can be
overridden with environment variables so the same code runs unchanged in
Colab, containers, and CI:

- ``RAN6G_DATA_PATH``    -> location of the synthetic dataset CSV
- ``RAN6G_MODEL_DIR``    -> directory for trained model artifacts
- ``RAN6G_PLOT_DIR``     -> directory for generated plots
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# The project root is two levels up from this file: <root>/ran6g/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem locations used across the project."""

    project_root: Path
    data_path: Path
    model_dir: Path
    plot_dir: Path

    def ensure_dirs(self) -> Paths:
        """Create output directories if they do not yet exist."""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Return the resolved project paths, honoring environment overrides."""
    root = PROJECT_ROOT
    data_path = Path(os.environ.get("RAN6G_DATA_PATH", root / "data" / "sample_dataset.csv"))
    model_dir = Path(os.environ.get("RAN6G_MODEL_DIR", root / "outputs" / "models"))
    plot_dir = Path(os.environ.get("RAN6G_PLOT_DIR", root / "outputs" / "plots"))
    return Paths(project_root=root, data_path=data_path, model_dir=model_dir, plot_dir=plot_dir)
