"""RAN environment for multi-cell synthetic simulations."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BaseStation:
    """Represents a gNodeB and its beam codebook."""

    cell_id: int
    x: float
    y: float
    num_beams: int


class RANEnvironment:
    """Maintains cell placement and beam utility logic."""

    def __init__(self, num_cells: int = 3, area_size: float = 1000.0, num_beams: int = 8, seed: int = 7) -> None:
        self.num_cells = num_cells
        self.area_size = area_size
        self.num_beams = num_beams
        self.rng = np.random.default_rng(seed)
        self.base_stations = self._create_cells()

    def _create_cells(self) -> list[BaseStation]:
        stations: list[BaseStation] = []
        for cell_id in range(self.num_cells):
            x, y = self.rng.uniform(0.1 * self.area_size, 0.9 * self.area_size, size=2)
            stations.append(BaseStation(cell_id=cell_id, x=float(x), y=float(y), num_beams=self.num_beams))
        return stations

    def nearest_cell(self, x: float, y: float) -> tuple[BaseStation, float]:
        """Return serving cell and distance in meters."""
        dists = [(bs, float(np.hypot(x - bs.x, y - bs.y))) for bs in self.base_stations]
        return min(dists, key=lambda item: item[1])

    def beam_gain_db(self, user_x: float, user_y: float, bs: BaseStation, beam_index: int) -> float:
        """Compute directional beam gain; best alignment yields higher gain."""
        angle = np.arctan2(user_y - bs.y, user_x - bs.x) % (2 * np.pi)
        beam_width = 2 * np.pi / bs.num_beams
        beam_center = (beam_index + 0.5) * beam_width
        offset = min(abs(angle - beam_center), 2 * np.pi - abs(angle - beam_center))
        max_gain = 18.0
        min_gain = 2.0
        normalized = max(0.0, 1 - offset / (np.pi))
        return float(min_gain + (max_gain - min_gain) * normalized)

    def optimal_beam(self, user_x: float, user_y: float, bs: BaseStation) -> int:
        gains = [self.beam_gain_db(user_x, user_y, bs, i) for i in range(bs.num_beams)]
        return int(np.argmax(gains))

    def interference_dbm(self, serving_cell_id: int, user_x: float, user_y: float) -> float:
        """Aggregate interference from non-serving cells."""
        powers_mw = []
        for bs in self.base_stations:
            if bs.cell_id == serving_cell_id:
                continue
            dist = max(np.hypot(user_x - bs.x, user_y - bs.y), 1.0)
            # Coarse attenuation model for interferers
            recv_dbm = 36.0 - 32.0 * np.log10(dist)
            powers_mw.append(10 ** (recv_dbm / 10.0))

        if not powers_mw:
            return -120.0
        return float(10 * np.log10(sum(powers_mw)))
