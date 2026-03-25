"""Simple real-time simulation loop for online inference demos."""

from __future__ import annotations

import time

import pandas as pd

from data.generator import SyntheticRANDataGenerator, GenerationConfig
from pipeline.inference import RANInferenceService
from pipeline.trainer import train_all


def run_realtime_demo(steps: int = 10, sleep_s: float = 0.2) -> None:
    train_all()
    service = RANInferenceService()
    generator = SyntheticRANDataGenerator(GenerationConfig(num_users=5, time_steps=steps))
    df = generator.generate()

    for t, t_df in df.groupby("time_step"):
        sample = t_df.iloc[0].to_dict()

        qos_input = {k: sample[k] for k in ["rsrp", "sinr", "cqi", "distance_to_cell", "beam_index", "interference_level", "speed"]}
        beam_input = {k: sample[k] for k in ["x", "y", "speed", "distance_to_cell", "rsrp", "sinr", "cqi", "interference_level", "cell_id"]}
        anomaly_input = {k: sample[k] for k in ["rsrp", "sinr", "cqi", "interference_level", "distance_to_cell", "speed", "throughput_mbps", "latency_ms"]}

        qos_pred = service.predict_qos(qos_input)
        beam_pred = service.select_beam(beam_input)
        anom_pred = service.detect_anomaly(anomaly_input)

        print(f"t={t:03d} qos={qos_pred['qos_class']} beam={beam_pred['optimal_beam_index']} anomaly={anom_pred['is_anomaly']}")
        time.sleep(sleep_s)


if __name__ == "__main__":
    run_realtime_demo()
