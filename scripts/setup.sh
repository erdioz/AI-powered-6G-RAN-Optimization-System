#!/usr/bin/env bash
# Prepare a development environment for the 6G RAN optimization system.
# Installs the package in editable mode with dev extras (pytest, ruff, httpx).
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Installing ran6g in editable mode with dev extras..."
if ! pip install -e ".[dev]"; then
  echo "Editable install failed; falling back to requirements.txt + dev tools."
  pip install -r requirements.txt pytest ruff httpx
fi

echo "Environment ready. Run 'make test' or 'pytest' to verify."
