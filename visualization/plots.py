"""Visualization helpers for the synthetic RAN dataset and model outputs."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_ue_movement(df: pd.DataFrame, user_id: int = 0, save_path: str = "outputs/plots/ue_movement.png") -> Path:
    user_df = df[df["user_id"] == user_id].sort_values("time_step")
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(user_df["x"], user_df["y"], marker="o", ms=2)
    plt.title(f"UE Movement Path (user_id={user_id})")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def plot_sinr_over_time(df: pd.DataFrame, user_id: int = 0, save_path: str = "outputs/plots/sinr_over_time.png") -> Path:
    user_df = df[df["user_id"] == user_id].sort_values("time_step")
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(user_df["time_step"], user_df["sinr"], color="tab:blue")
    plt.title(f"SINR Over Time (user_id={user_id})")
    plt.xlabel("Time step")
    plt.ylabel("SINR (dB)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def plot_beam_selection(df: pd.DataFrame, user_id: int = 0, save_path: str = "outputs/plots/beam_selection.png") -> Path:
    user_df = df[df["user_id"] == user_id].sort_values("time_step")
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.step(user_df["time_step"], user_df["beam_index"], label="Selected beam", where="post")
    plt.step(user_df["time_step"], user_df["optimal_beam_index"], label="Optimal beam", where="post", alpha=0.7)
    plt.title(f"Beam Selection Decisions (user_id={user_id})")
    plt.xlabel("Time step")
    plt.ylabel("Beam index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def plot_anomalies(df: pd.DataFrame, save_path: str = "outputs/plots/anomalies.png") -> Path:
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    normal = df[df["is_anomaly"] == 0]
    anomalous = df[df["is_anomaly"] == 1]
    plt.scatter(normal["sinr"], normal["interference_level"], s=8, alpha=0.4, label="Normal")
    plt.scatter(anomalous["sinr"], anomalous["interference_level"], s=16, alpha=0.8, label="Anomaly", color="tab:red")
    plt.title("Detected/Injected Radio Anomalies")
    plt.xlabel("SINR (dB)")
    plt.ylabel("Interference level (dBm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output
