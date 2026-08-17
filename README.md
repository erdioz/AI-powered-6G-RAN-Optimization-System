# AI-powered 6G RAN Optimization System

A lightweight, modular Python project for simulating a 6G-like Radio Access Network (RAN) and training AI models for:

- **QoS prediction** (`good` / `medium` / `poor`)
- **Beam selection optimization** (best beam index)
- **Radio anomaly detection** (interference / degradation)

The solution is synthetic-data driven and **Google Colab-compatible**.

Tested on Python 3.10–3.12, with a pytest suite, ruff linting, and CI on every push/PR.

---

## Project Structure

```text
AI-powered-6G-RAN-Optimization-System/
├── api/
│   ├── app.py            # FastAPI app (lifespan loading, single + batch endpoints)
│   └── metrics.py        # in-process request metrics
├── data/
│   ├── generator.py
│   ├── radio_channel.py
│   └── sample_dataset.csv
├── models/
│   ├── anomaly_model.py
│   ├── beam_model.py
│   └── qos_model.py
├── pipeline/
│   ├── inference.py
│   └── trainer.py
├── simulation/
│   ├── beam_rl.py
│   ├── mobility.py
│   ├── ran_environment.py
│   └── realtime_loop.py
├── ran6g/                # config (paths), logging, and the `ran6g` CLI
├── visualization/
│   └── plots.py
├── tests/                # pytest suite (~81% coverage)
├── docs/architecture.md  # component + data-flow diagram
├── notebooks/
│   └── 6g_ran_colab_demo.ipynb
├── outputs/              # generated model + plot artifacts (gitignored)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

See [`docs/architecture.md`](docs/architecture.md) for a component and data-flow
diagram.

---

## 1) Data Simulation

`data/generator.py` creates a time-series dataset containing:

- UE state: `user_id`, `x`, `y`, `speed`
- Serving cell info: `cell_id`, `distance_to_cell`, `azimuth_to_cell`
- Radio metrics: `RSRP`, `SINR`, `CQI`, `interference_level`, `noise_floor`
- Beam fields: `beam_index`, `optimal_beam_index`, `beam_gain_db`
- QoS outcomes: `qos_class`, `latency_ms`, `throughput_mbps`
- Anomaly annotation: `is_anomaly`

Synthetic realism includes:

- Path loss and shadowing
- Multi-cell interference
- UE mobility over time
- Occasional injected interference/degradation anomalies

Generate dataset:

```bash
python -m data.generator
```

This writes `data/sample_dataset.csv`.

---

## 2) AI Models

### QoS Prediction
- File: `models/qos_model.py`
- Model: `RandomForestClassifier`
- Input features: `rsrp, sinr, cqi, distance_to_cell, beam_index, interference_level, speed`
- Output: `qos_class`

> **Note on QoS accuracy.** In the synthetic dataset `qos_class` is defined as a
> threshold on SINR (`RadioChannel.qos_class_from_sinr`), and SINR is itself a
> feature. The task is therefore *near-deterministic* and the model reaches ~100%
> accuracy — this is expected, not overfitting. The training report includes
> per-class F1 and feature importances so this relationship is transparent. To
> make QoS a genuinely harder learning problem, redefine the label as a function
> of latency/throughput/noise rather than SINR alone.

### Beam Selection
- File: `models/beam_model.py`
- Model: `RandomForestClassifier` (multi-class)
- Input features: `UE position + azimuth_to_cell + radio metrics + cell_id`
- Output: `optimal_beam_index`

> The optimal beam is physically the beam whose boresight best aligns with the
> **angle-of-departure from the serving cell** to the UE. Exposing that angle as
> the `azimuth_to_cell` feature raises beam-selection accuracy from ~0.18 to
> ~0.99, and it dominates the feature-importance ranking as expected.

### Anomaly Detection
- File: `models/anomaly_model.py`
- Model: `IsolationForest`
- Output: anomaly flag and score

---

## 3) Training and Inference Pipelines

Train all models and save artifacts:

```bash
python -m pipeline.trainer
```

Model artifacts are saved in `outputs/models/`.

Inference service usage:

```python
from pipeline.inference import RANInferenceService
service = RANInferenceService()
```

---

## 4) API Endpoints (FastAPI)

Start server:

```bash
ran6g serve --reload        # or: uvicorn api.app:app --reload
```

Models are loaded once during the application **lifespan** (not at import time);
if no artifacts are found they are trained on startup.

**Prediction endpoints** (single item):

- `POST /predict_qos`
- `POST /select_beam`
- `POST /detect_anomaly`

**Batch endpoints** (accept a JSON list, vectorized, up to 1024 items):

- `POST /predict_qos/batch`
- `POST /select_beam/batch`
- `POST /detect_anomaly/batch`

**Operational endpoints:**

- `GET /health` — liveness and which models are loaded
- `GET /model_info` — training metrics/metadata of the loaded models
- `GET /metrics` — in-process request counts, errors, and average latency

Inputs are validated by pydantic (e.g. `cqi` must be `0–15`). Interactive docs
are available at `/docs` when the server is running. Example:

```bash
curl -X POST http://127.0.0.1:8000/predict_qos \
  -H 'Content-Type: application/json' \
  -d '{"rsrp":-70,"sinr":12.5,"cqi":10,"distance_to_cell":250,"beam_index":3,"interference_level":-100,"speed":5}'
```

### Docker

```bash
docker compose up --build        # serves on http://127.0.0.1:8000
# or:
docker build -t ran6g-optimizer . && docker run -p 8000:8000 ran6g-optimizer
```

The image trains the models at build time so the container starts ready to serve.

---

## 5) Visualization

`visualization/plots.py` provides:

- UE movement trajectory
- SINR over time
- Selected vs optimal beam indices
- Anomaly scatter plot

Example:

```python
import pandas as pd
from visualization.plots import plot_ue_movement

df = pd.read_csv("data/sample_dataset.csv")
plot_ue_movement(df, user_id=0)
```

Plots are stored in `outputs/plots/`.

---

## 6) Colab Notebook

`notebooks/6g_ran_colab_demo.ipynb` includes an end-to-end workflow:

1. Install dependencies
2. Generate synthetic dataset
3. Train all models
4. Run sample inference
5. Visualize movement, SINR, beam decisions, and anomalies

---

## 7) Advanced Features Included

- **Optional RL beam optimizer**: `simulation/beam_rl.py` (tabular Q-learning)
- **Multi-cell interference simulation**: in `simulation/ran_environment.py`
- **Real-time simulation loop**: `simulation/realtime_loop.py`

Run real-time demo:

```bash
python -m simulation.realtime_loop
```

---

## Installation

Runtime only:

```bash
pip install -r requirements.txt
```

Editable install with dev tools (pytest, ruff), recommended for development:

```bash
pip install -e ".[dev]"
# or:
./scripts/setup.sh
```

---

## Command-line interface

Installing the package exposes a `ran6g` console command:

```bash
ran6g generate            # write the synthetic dataset CSV
ran6g train               # train all models and print metrics
ran6g realtime --steps 15 # run the online inference demo
ran6g serve --port 8000   # launch the FastAPI server
```

### Configuration via environment variables

Filesystem locations are resolved from the project root but can be overridden so
the same code runs unchanged in Colab, containers, and CI:

| Variable            | Default                         | Purpose                     |
|---------------------|---------------------------------|-----------------------------|
| `RAN6G_DATA_PATH`   | `data/sample_dataset.csv`       | Synthetic dataset location  |
| `RAN6G_MODEL_DIR`   | `outputs/models`                | Trained model artifacts     |
| `RAN6G_PLOT_DIR`    | `outputs/plots`                 | Generated plots             |
| `RAN6G_LOG_LEVEL`   | `INFO`                          | Logging verbosity           |

---

## Development

Common workflows are wrapped in a `Makefile`:

```bash
make install-dev   # editable install with dev extras
make test          # run the pytest suite
make cov           # tests with a coverage report
make lint          # ruff lint checks
make format        # auto-format and fix with ruff
```

The test suite lives in `tests/` and covers the radio-channel physics, mobility,
the RAN environment, all three models (train/predict/save/load), the training and
inference pipelines, and the FastAPI endpoints. Continuous integration
(`.github/workflows/ci.yml`) runs ruff and the test suite across Python 3.10–3.12
on every push and pull request.

---

## Extensibility for future 6G AI use-cases

The modular architecture is designed to support next-phase research:

- **Digital Twin Integration**: plug in higher-fidelity channel or ray-tracing simulators
- **RAN Slicing Intelligence**: add per-slice QoS targets and slice-aware policies
- **Online/Continual Learning**: replace static training with streaming updates
- **Policy Optimization**: integrate actor-critic or contextual bandits for scheduling and beam control

Each module can be swapped independently without changing API contracts.
