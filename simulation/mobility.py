"""UE mobility models for RAN simulations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class UEState:
    """UE motion state in a 2D map."""

    user_id: int
    x: float
    y: float
    speed_mps: float
    direction_rad: float


class MobilityModel:
    """Simple random-waypoint-like mobility with reflective boundaries."""

    def __init__(self, area_size: float = 1000.0, rng_seed: int = 42) -> None:
        self.area_size = area_size
        self.rng = np.random.default_rng(rng_seed)

    def initialize_users(self, num_users: int, speed_min: float = 0.5, speed_max: float = 20.0) -> list[UEState]:
        states: list[UEState] = []
        for user_id in range(num_users):
            x, y = self.rng.uniform(0, self.area_size, size=2)
            speed = self.rng.uniform(speed_min, speed_max)
            direction = self.rng.uniform(0, 2 * np.pi)
            states.append(UEState(user_id=user_id, x=float(x), y=float(y), speed_mps=float(speed), direction_rad=float(direction)))
        return states

    def step(self, state: UEState, dt_s: float = 1.0) -> UEState:
        """Advance state by one step with directional jitter."""
        direction = state.direction_rad + self.rng.normal(0, 0.2)
        speed = max(0.1, state.speed_mps + self.rng.normal(0, 0.5))

        new_x = state.x + speed * np.cos(direction) * dt_s
        new_y = state.y + speed * np.sin(direction) * dt_s

        if new_x < 0 or new_x > self.area_size:
            direction = np.pi - direction
            new_x = np.clip(new_x, 0, self.area_size)
        if new_y < 0 or new_y > self.area_size:
            direction = -direction
            new_y = np.clip(new_y, 0, self.area_size)

        return UEState(
            user_id=state.user_id,
            x=float(new_x),
            y=float(new_y),
            speed_mps=float(speed),
            direction_rad=float(direction),
        )
