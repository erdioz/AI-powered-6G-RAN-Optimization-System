"""Visualization helpers for the synthetic RAN dataset and model outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Wedge


def infer_num_beams(df: pd.DataFrame) -> int:
    """Infer the beam-codebook size from the beam-index columns."""
    cols = [c for c in ("beam_index", "optimal_beam_index") if c in df.columns]
    if not cols:
        return 8
    return int(max(int(df[c].max()) for c in cols)) + 1


def recover_sites(df: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Recover each cell's (x, y) location from the UE geometry.

    Since ``azimuth_to_cell = atan2(y - by, x - bx)`` and ``distance_to_cell`` is
    the UE-to-cell range, the serving cell sits at
    ``(x - dist*cos(az), y - dist*sin(az))``. Averaging over every observation of
    a cell cancels floating-point noise.
    """
    sites: dict[int, tuple[float, float]] = {}
    if not {"azimuth_to_cell", "distance_to_cell", "x", "y", "cell_id"}.issubset(df.columns):
        return sites
    bx = df["x"] - df["distance_to_cell"] * np.cos(df["azimuth_to_cell"])
    by = df["y"] - df["distance_to_cell"] * np.sin(df["azimuth_to_cell"])
    for cell_id, group in pd.DataFrame({"cell_id": df["cell_id"], "bx": bx, "by": by}).groupby("cell_id"):
        sites[int(cell_id)] = (float(group["bx"].mean()), float(group["by"].mean()))
    return sites


def plot_ue_movement(
    df: pd.DataFrame,
    user_id: int = 0,
    save_path: str = "outputs/plots/ue_movement.png",
    show_sectors: bool = True,
    sector_radius: float | None = None,
) -> Path:
    """Plot a UE trajectory over the cell layout, with colored beam sectors.

    Cell sites are drawn as markers, each cell's beams are shown as color-coded
    angular sectors, and the UE track markers are colored by the optimal serving
    beam so beam handovers are visible against the sectors.
    """
    user_df = df[df["user_id"] == user_id].sort_values("time_step")
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_beams = infer_num_beams(df)
    cmap = plt.get_cmap("hsv", num_beams)  # cyclic map: beams wrap around 360 deg
    norm = Normalize(vmin=-0.5, vmax=num_beams - 0.5)
    sites = recover_sites(df) if show_sectors else {}

    # Sector radius defaults to a fraction of the map's diagonal.
    if sector_radius is None:
        span_x = float(df["x"].max() - df["x"].min())
        span_y = float(df["y"].max() - df["y"].min())
        sector_radius = 0.42 * float(np.hypot(span_x, span_y))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    beam_width_deg = 360.0 / num_beams

    # Colored beam sectors radiating from each site.
    for cell_id, (sx, sy) in sites.items():
        for b in range(num_beams):
            ax.add_patch(
                Wedge(
                    (sx, sy),
                    sector_radius,
                    b * beam_width_deg,
                    (b + 1) * beam_width_deg,
                    facecolor=cmap(b),
                    alpha=0.13,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=1,
                )
            )
        ax.plot(sx, sy, marker="^", color="black", markersize=13, zorder=6)
        ax.annotate(
            f"Cell {cell_id}",
            (sx, sy),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            zorder=7,
        )

    # UE trajectory: faint connecting line + markers colored by optimal beam.
    ax.plot(user_df["x"], user_df["y"], color="0.35", linewidth=1.0, alpha=0.6, zorder=3)
    beam_col = "optimal_beam_index" if "optimal_beam_index" in user_df.columns else None
    if beam_col is not None:
        sc = ax.scatter(
            user_df["x"], user_df["y"],
            c=user_df[beam_col], cmap=cmap, norm=norm,
            s=32, edgecolor="black", linewidth=0.3, zorder=4,
        )
        cbar = fig.colorbar(sc, ax=ax, ticks=range(num_beams), fraction=0.046, pad=0.04)
        cbar.set_label("Optimal beam index")
    else:
        ax.scatter(user_df["x"], user_df["y"], s=20, color="tab:blue", zorder=4)

    # Mark start and end of the track.
    if len(user_df):
        ax.scatter(user_df["x"].iloc[0], user_df["y"].iloc[0], marker="o", s=90,
                   facecolor="none", edgecolor="lime", linewidth=2, zorder=5, label="Start")
        ax.scatter(user_df["x"].iloc[-1], user_df["y"].iloc[-1], marker="X", s=90,
                   color="black", zorder=5, label="End")

    ax.plot([], [], marker="^", color="black", linestyle="none", label="Cell site")
    ax.set_title(f"UE Movement over Cell Layout (user_id={user_id})\nsectors colored by beam index")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
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
