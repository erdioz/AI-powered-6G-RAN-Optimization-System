# Contributing

Thanks for your interest in improving the AI-powered 6G RAN Optimization System.

## Development setup

```bash
git clone https://github.com/erdioz/AI-powered-6G-RAN-Optimization-System.git
cd AI-powered-6G-RAN-Optimization-System
pip install -e ".[dev]"    # or: ./scripts/setup.sh
```

## Workflow

1. Create a feature branch off `main`.
2. Make your change with matching tests under `tests/`.
3. Run the checks locally:

   ```bash
   make lint    # ruff
   make test    # pytest
   make cov     # coverage report
   ```

4. Ensure `ruff check .` is clean and the full suite passes.
5. Open a pull request describing the change and its motivation.

## Code style

- Formatting and linting are enforced by **ruff** (config in `pyproject.toml`).
  Run `make format` to auto-fix.
- Keep functions typed and documented, matching the surrounding code.
- Prefer the central `ran6g.config` paths and `ran6g.logging_utils.get_logger`
  over hard-coded paths or `print`.

## Adding a feature or column

- New dataset columns go in `data/generator.py`; update the affected model
  `feature_columns`, the API input schema in `api/app.py`, and the tests.
- New models follow the existing `train / predict / predict_batch / save / load`
  interface so they slot into `pipeline/trainer.py` and `pipeline/inference.py`.

## Continuous integration

`.github/workflows/ci.yml` runs ruff and the pytest suite across Python
3.10–3.12 on every push and pull request. PRs must be green before merge.
