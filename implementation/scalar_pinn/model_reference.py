"""Finite-generator model and one-day semigroup for the daily reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.linalg import expm
from scipy.sparse import lil_matrix


LOWER = 0.35
W0 = 0.60
BUDGETS = np.asarray((0.0, 0.06, 0.14, 0.22, 0.30), dtype=np.float64)
ACTION_SIZE = 0.08
DRIFT_INTERCEPT = 0.04
DRIFT_SLOPE = 0.08
DIFFUSION = 0.06
MARKS = np.asarray((-0.6, 0.4), dtype=np.float64)
MARK_WEIGHTS = np.asarray((0.7, 0.3), dtype=np.float64)
JUMP_SCALE = 0.30
INTENSITIES = (0.0, 0.12)
EPSILON = 0.20
HORIZON_DAYS = 20
N_DECISIONS = 20


@dataclass(frozen=True)
class ModelRow:
    upper: float
    h: float
    intensity: float
    grid: np.ndarray
    Q: object
    P: np.ndarray
    left_rates: np.ndarray
    right_rates: np.ndarray

    @property
    def state_count(self) -> int:
        return int(self.grid.size)

    @property
    def row_key(self) -> str:
        return f"lambda_{self.intensity:g}_upper_{self.upper:g}_h_{self.h:g}"


def drift(moisture: np.ndarray | float) -> np.ndarray:
    return DRIFT_INTERCEPT - DRIFT_SLOPE * np.asarray(moisture, dtype=np.float64)


def jump_map(moisture: float) -> np.ndarray:
    return float(moisture) * np.exp(JUMP_SCALE * MARKS)


def _interpolation_nodes(mapped: float, grid: np.ndarray) -> tuple[tuple[int, float], ...]:
    if mapped <= float(grid[0]):
        return ((0, 1.0),)
    if mapped >= float(grid[-1]):
        return ((grid.size - 1, 1.0),)
    position = (mapped - float(grid[0])) / float(grid[1] - grid[0])
    lower = int(np.floor(position))
    upper = lower + 1
    fraction = position - lower
    if fraction <= 1.0e-14:
        return ((lower, 1.0),)
    return ((lower, 1.0 - fraction), (upper, fraction))


def build_generator(upper: float, h: float, intensity: float) -> tuple[np.ndarray, object, np.ndarray, np.ndarray]:
    """Build Q without clipping rates or normalizing rows after construction."""

    upper = float(upper)
    h = float(h)
    intensity = float(intensity)
    intervals = int(round((upper - LOWER) / h))
    if intervals < 2 or not np.isclose(LOWER + intervals * h, upper, atol=1.0e-12, rtol=0.0):
        raise ValueError("upper and h must define an integer uniform grid")
    grid = np.linspace(LOWER, upper, intervals + 1, dtype=np.float64)
    actual_h = float(grid[1] - grid[0])
    if not np.isclose(actual_h, h, atol=1.0e-12, rtol=0.0):
        raise ValueError("uniform grid spacing mismatch")

    n_states = grid.size
    Q = lil_matrix((n_states, n_states), dtype=np.float64)
    left_rates = np.zeros(n_states, dtype=np.float64)
    right_rates = np.zeros(n_states, dtype=np.float64)
    diffusion_sq = DIFFUSION**2
    for index in range(1, n_states - 1):
        w = float(grid[index])
        b = float(drift(w))
        left = 0.5 * diffusion_sq / actual_h**2 - b / (2.0 * actual_h)
        right = 0.5 * diffusion_sq / actual_h**2 + b / (2.0 * actual_h)
        if left < -1.0e-12 or right < -1.0e-12:
            raise ValueError(f"negative diffusion/drift rate at grid index {index}")
        left_rates[index] = left
        right_rates[index] = right
        Q[index, index - 1] += left
        Q[index, index + 1] += right
        Q[index, index] -= left + right
        if intensity != 0.0:
            for mark, probability in zip(MARKS, MARK_WEIGHTS):
                mapped = w * float(np.exp(JUMP_SCALE * mark))
                for node, weight in _interpolation_nodes(mapped, grid):
                    Q[index, node] += intensity * float(probability) * weight
                Q[index, index] -= intensity * float(probability)

    csr = Q.tocsr()
    off_diagonal = csr.toarray().copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if float(np.min(off_diagonal)) < -1.0e-12:
        raise ValueError("negative off-diagonal generator entry")
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    if float(np.max(np.abs(row_sums))) >= 1.0e-10:
        raise ValueError("generator row sums exceed tolerance")
    return grid, csr, left_rates, right_rates


def build_model_row(upper: float, h: float, intensity: float) -> ModelRow:
    grid, Q, left_rates, right_rates = build_generator(upper, h, intensity)
    P = np.asarray(expm(Q.toarray()), dtype=np.float64)
    if not np.isfinite(P).all():
        raise ValueError("nonfinite daily transition")
    if float(np.min(P)) < -1.0e-12:
        raise ValueError("daily transition has a negative probability")
    row_sums = P.sum(axis=1)
    if float(np.max(np.abs(row_sums - 1.0))) >= 1.0e-10:
        raise ValueError("daily transition row sums exceed tolerance")
    return ModelRow(
        upper=float(upper),
        h=float(h),
        intensity=float(intensity),
        grid=grid,
        Q=Q,
        P=P,
        left_rates=left_rates,
        right_rates=right_rates,
    )


def interpolate_column(values: np.ndarray, locations: np.ndarray | float, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation with explicit absorbing boundary sentinels."""

    values = np.asarray(values, dtype=np.float64)
    locations_array = np.asarray(locations, dtype=np.float64)
    result = np.interp(locations_array, grid, values)
    result = np.asarray(result, dtype=np.float64)
    result = np.where(locations_array >= float(grid[-1]), 0.0, result)
    return np.where(locations_array <= float(grid[0]), values[0], result)


def actions_for_budget(budget_index: int) -> tuple[float, ...]:
    if budget_index < 0 or budget_index >= BUDGETS.size:
        raise IndexError(budget_index)
    if budget_index == 0:
        return (0.0,)
    if budget_index == 1:
        return (0.0, 0.06)
    return (0.0, ACTION_SIZE)


def successor_budget_index(budget_index: int, action: float) -> int:
    if abs(float(action)) <= 1.0e-14:
        return int(budget_index)
    if budget_index == 1 and np.isclose(action, 0.06, atol=1.0e-12, rtol=0.0):
        return 0
    if budget_index >= 2 and np.isclose(action, ACTION_SIZE, atol=1.0e-12, rtol=0.0):
        return int(budget_index - 1)
    raise ValueError(f"invalid action {action} at budget index {budget_index}")


def row_specifications() -> tuple[tuple[float, float], ...]:
    return (
        (1.55, 0.01),
        (1.55, 0.005),
        (1.55, 0.0025),
        (1.15, 0.005),
        (1.95, 0.005),
    )

