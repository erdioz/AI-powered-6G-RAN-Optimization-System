"""Optional lightweight reinforcement learning beam optimizer."""

from __future__ import annotations

import numpy as np


class QLearningBeamOptimizer:
    """Tabular Q-learning for coarse beam decision based on binned state."""

    def __init__(self, num_beams: int, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.1, seed: int = 42) -> None:
        self.num_beams = num_beams
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple[int, int], np.ndarray] = {}

    def _state_key(self, sinr: float, distance: float) -> tuple[int, int]:
        return (int(np.clip((sinr + 20) // 5, 0, 20)), int(np.clip(distance // 100, 0, 20)))

    def act(self, sinr: float, distance: float) -> int:
        key = self._state_key(sinr, distance)
        self.q_table.setdefault(key, np.zeros(self.num_beams))
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_beams))
        return int(np.argmax(self.q_table[key]))

    def update(self, sinr: float, distance: float, action: int, reward: float, next_sinr: float, next_distance: float) -> None:
        key = self._state_key(sinr, distance)
        next_key = self._state_key(next_sinr, next_distance)
        self.q_table.setdefault(key, np.zeros(self.num_beams))
        self.q_table.setdefault(next_key, np.zeros(self.num_beams))

        best_next = np.max(self.q_table[next_key])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[key][action]
        self.q_table[key][action] += self.alpha * td_error
