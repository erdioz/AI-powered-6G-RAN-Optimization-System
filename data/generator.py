"""Synthetic dataset generation for AI-powered 6G RAN optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from data.radio_channel import RadioChannel
from simulation.mobility import MobilityModel
from simulation.ran_environment import RANEnvironment


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""

    num_users: int = 60
    num_cells: int = 3
    num_beams: int = 8
    time_steps: int = 120
    area_size: float = 1200.0
    random_seed: int = 13


class SyntheticRANDataGenerator:
    """Generates a time-series synthetic dataset for 6G RAN tasks."""

    def __init__(self, config: GenerationConfig | None = None) -> None:
        self.config = config or GenerationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.mobility = MobilityModel(area_size=self.config.area_size, rng_seed=self.config.random_seed)
        self.environment = RANEnvironment(
            num_cells=self.config.num_cells,
            area_size=self.config.area_size,
            num_beams=self.config.num_beams,
            seed=self.config.random_seed,
        )
        self.radio = RadioChannel()

    def generate(self) -> pd.DataFrame:
        """Generate synthetic per-UE/per-timestep observations."""
        users = self.mobility.initialize_users(self.config.num_users)
        rows: list[dict] = []

        for t in range(self.config.time_steps):
            new_states = []
            for user in users:
                state = self.mobility.step(user)
                serving_bs, dist_m = self.environment.nearest_cell(state.x, state.y)

                optimal_beam = self.environment.optimal_beam(state.x, state.y, serving_bs)
                explored_beam = int((optimal_beam + self.rng.integers(-1, 2)) % self.config.num_beams)

                beam_gain = self.environment.beam_gain_db(state.x, state.y, serving_bs, explored_beam)
                shadowing = self.rng.normal(0, self.radio.params.shadowing_std_db)
                interference = self.environment.interference_dbm(serving_bs.cell_id, state.x, state.y)

                # Random anomaly injection for robust anomaly detection
                anomaly_flag = int(self.rng.random() < 0.05)
                if anomaly_flag:
                    interference += self.rng.uniform(8, 20)
                    shadowing -= self.rng.uniform(4, 10)

                rsrp = self.radio.rsrp_dbm(dist_m, beam_gain_db=beam_gain, shadowing_db=float(shadowing))
                sinr = self.radio.sinr_db(signal_dbm=rsrp, interference_dbm=interference)
                cqi = self.radio.sinr_to_cqi(sinr)
                qos_class = self.radio.qos_class_from_sinr(sinr)

                throughput = self.radio.throughput_mbps_from_sinr(sinr)
                latency = self.radio.latency_ms_from_sinr(sinr)

                rows.append(
                    {
                        "time_step": t,
                        "user_id": state.user_id,
                        "cell_id": serving_bs.cell_id,
                        "x": state.x,
                        "y": state.y,
                        "speed": state.speed_mps,
                        "distance_to_cell": dist_m,
                        "beam_index": explored_beam,
                        "optimal_beam_index": optimal_beam,
                        "beam_gain_db": beam_gain,
                        "rsrp": rsrp,
                        "sinr": sinr,
                        "cqi": cqi,
                        "interference_level": interference,
                        "noise_floor": self.radio.params.noise_floor_dbm,
                        "latency_ms": latency,
                        "throughput_mbps": throughput,
                        "qos_class": qos_class,
                        "is_anomaly": anomaly_flag,
                    }
                )
                new_states.append(state)
            users = new_states

        return pd.DataFrame(rows)

    def to_csv(self, output_path: str | Path) -> Path:
        """Generate dataset and write to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.generate()
        df.to_csv(output_path, index=False)
        return output_path


if __name__ == "__main__":
    generator = SyntheticRANDataGenerator()
    path = generator.to_csv("data/sample_dataset.csv")
    print(f"Synthetic dataset generated at: {path}")
