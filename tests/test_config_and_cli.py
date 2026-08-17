"""Tests for path configuration and the CLI argument parser."""

from __future__ import annotations

import ran6g.config as config
from ran6g.cli import build_parser


def test_paths_honor_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAN6G_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("RAN6G_DATA_PATH", str(tmp_path / "d.csv"))
    config.get_paths.cache_clear()
    try:
        paths = config.get_paths()
        assert paths.model_dir == tmp_path / "m"
        assert paths.data_path == tmp_path / "d.csv"
    finally:
        config.get_paths.cache_clear()


def test_ensure_dirs_creates_directories(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAN6G_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("RAN6G_PLOT_DIR", str(tmp_path / "p"))
    monkeypatch.setenv("RAN6G_DATA_PATH", str(tmp_path / "sub" / "d.csv"))
    config.get_paths.cache_clear()
    try:
        paths = config.get_paths().ensure_dirs()
        assert paths.model_dir.is_dir()
        assert paths.plot_dir.is_dir()
        assert paths.data_path.parent.is_dir()
    finally:
        config.get_paths.cache_clear()


def test_cli_parses_subcommands() -> None:
    parser = build_parser()
    args = parser.parse_args(["realtime", "--steps", "3", "--sleep", "0.0"])
    assert args.command == "realtime"
    assert args.steps == 3
    assert args.sleep == 0.0


def test_cli_serve_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.reload is False
