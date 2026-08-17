"""Tests for mobility, the RAN environment, and the RL beam optimizer."""

from __future__ import annotations

import numpy as np

from simulation.beam_rl import QLearningBeamOptimizer
from simulation.mobility import MobilityModel
from simulation.ran_environment import RANEnvironment


def test_mobility_users_within_bounds() -> None:
    model = MobilityModel(area_size=1000.0, rng_seed=1)
    users = model.initialize_users(20)
    assert len(users) == 20
    for u in users:
        assert 0 <= u.x <= 1000
        assert 0 <= u.y <= 1000


def test_mobility_step_keeps_users_in_area() -> None:
    model = MobilityModel(area_size=500.0, rng_seed=2)
    state = model.initialize_users(1)[0]
    for _ in range(200):
        state = model.step(state)
        assert 0 <= state.x <= 500
        assert 0 <= state.y <= 500


def test_environment_creates_requested_cells() -> None:
    env = RANEnvironment(num_cells=4, area_size=1000, num_beams=8, seed=3)
    assert len(env.base_stations) == 4


def test_nearest_cell_returns_closest() -> None:
    env = RANEnvironment(num_cells=3, area_size=1000, num_beams=8, seed=3)
    bs, dist = env.nearest_cell(env.base_stations[0].x, env.base_stations[0].y)
    assert bs.cell_id == 0
    assert dist < 1e-6


def test_optimal_beam_maximizes_gain() -> None:
    env = RANEnvironment(num_cells=1, area_size=1000, num_beams=8, seed=3)
    bs = env.base_stations[0]
    best = env.optimal_beam(bs.x + 100, bs.y, bs)
    gains = [env.beam_gain_db(bs.x + 100, bs.y, bs, i) for i in range(bs.num_beams)]
    assert gains[best] == max(gains)


def test_interference_finite() -> None:
    env = RANEnvironment(num_cells=3, area_size=1000, num_beams=8, seed=3)
    val = env.interference_dbm(0, 500, 500)
    assert np.isfinite(val)


def test_qlearning_updates_q_value() -> None:
    opt = QLearningBeamOptimizer(num_beams=8, seed=0)
    action = opt.act(sinr=5.0, distance=200.0)
    before = opt.q_table[opt._state_key(5.0, 200.0)][action].copy()
    opt.update(5.0, 200.0, action, reward=1.0, next_sinr=6.0, next_distance=180.0)
    after = opt.q_table[opt._state_key(5.0, 200.0)][action]
    assert after > before
