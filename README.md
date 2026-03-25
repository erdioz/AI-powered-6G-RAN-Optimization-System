# AI-powered 6G RAN Optimization System

A lightweight, modular Python project for simulating a 6G-like Radio Access Network (RAN) and training AI models for:

- **QoS prediction** (`good` / `medium` / `poor`)
- **Beam selection optimization** (best beam index)
- **Radio anomaly detection** (interference / degradation)

The solution is synthetic-data driven and **Google Colab-compatible**.

---

## Project Structure

```text
AI-powered-6G-RAN-Optimization-System/
├── api/
│   └── app.py
├── data/
│   ├── generator.py
│   ├── radio_channel.py
│   └── sample_dataset.csv
├── models/
│   ├── anomaly_model.py
│   ├── beam_model.py
│   └── qos_model.py
├── notebooks/
│   └── 6g_ran_colab_demo.ipynb
├── pipeline/
│   ├── inference.py
│   └── trainer.py
├── simulation/
│   ├── beam_rl.py
│   ├── mobility.py
│   ├── ran_environment.py
│   └── realtime_loop.py
├── visualization/
│   └── plots.py
├── outputs/
│   ├── models/
│   └── plots/
├── requirements.txt
└── README.md
```

---

## 1) Data Simulation

`data/generator.py` creates a time-series dataset containing:

- UE state: `user_id`, `x`, `y`, `speed`
- Serving cell info: `cell_id`, `distance_to_cell`
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

### Beam Selection
- File: `models/beam_model.py`
- Model: `RandomForestClassifier` (multi-class)
- Input features: `radio metrics + UE position + cell_id`
- Output: `optimal_beam_index`

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
uvicorn api.app:app --reload
```

Endpoints:

- `POST /predict_qos`
- `POST /select_beam`
- `POST /detect_anomaly`

Each endpoint accepts JSON input and returns model predictions.

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

```bash
pip install -r requirements.txt
```

---

## Extensibility for future 6G AI use-cases

The modular architecture is designed to support next-phase research:

- **Digital Twin Integration**: plug in higher-fidelity channel or ray-tracing simulators
- **RAN Slicing Intelligence**: add per-slice QoS targets and slice-aware policies
- **Online/Continual Learning**: replace static training with streaming updates
- **Policy Optimization**: integrate actor-critic or contextual bandits for scheduling and beam control

Each module can be swapped independently without changing API contracts.
