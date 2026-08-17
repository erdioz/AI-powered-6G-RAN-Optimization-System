"""Shared cross-cutting utilities for the 6G RAN optimization system.

This package centralizes configuration (paths), logging, and the CLI entry
point so the domain modules (``data``, ``models``, ``pipeline`` …) don't have
to hard-code filesystem locations or duplicate logging setup.
"""

from ran6g.config import Paths, get_paths
from ran6g.logging_utils import get_logger

__all__ = ["Paths", "get_paths", "get_logger"]
