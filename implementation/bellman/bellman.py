"""Daily finite-chain Bellman solver and independent policy evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import model


@dataclass
class Solution:
    mu: float | None
    risk_only: bool
    V: np.ndarray
    C: np.ndarray
    R: np.ndarray
    continuation_V: np.ndarray
    continuation_C: np.ndarray
    continuation_R: np.ndarray
    actions: np.ndarray
    successor_budget: np.ndarray
    bellman_gap: float
    identity_gap: float


@dataclass
class PolicyEvaluation:
    C: np.ndarray
    R: np.ndarray
    budget_feasible: bool
    invalid_action: str | None


def _candidate_value(
    continuation: np.ndarray,
    grid: np.ndarray,
    state_shift: float,
    successor_index: int,
) -> np.ndarray:
    return model.interpolate_column(
        continuation[successor_index], grid + state_shift, grid
    )


def solve(
    mu: float | None,
    P: np.ndarray,
    grid: np.ndarray,
    *,
    risk_only: bool = False,
) -> Solution:
    """Solve the complete 20-day Bellman chain for one multiplier."""

    if not risk_only and (mu is None or not np.isfinite(mu) or mu < 0.0):
        raise ValueError("finite nonnegative mu required")
    P = np.asarray(P, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    n_states = grid.size
    shape = (model.N_DECISIONS + 1, model.BUDGETS.size, n_states)
    V = np.zeros(shape, dtype=np.float64)
    C = np.zeros(shape, dtype=np.float64)
    R = np.zeros(shape, dtype=np.float64)
    continuation_V = np.zeros((model.N_DECISIONS, model.BUDGETS.size, n_states), dtype=np.float64)
    continuation_C = np.zeros_like(continuation_V)
    continuation_R = np.zeros_like(continuation_V)
    actions = np.zeros_like(continuation_V)
    successor_budget = np.zeros(continuation_V.shape, dtype=np.int64)

    # Failure and safe sentinel conventions apply at terminal and every prior
    # decision time.  A risk-only value is the probability sentinel 1.
    failure_value = 1.0 if risk_only else float(mu)
    V[:, :, 0] = failure_value
    R[:, :, 0] = 1.0
    V[:, :, -1] = 0.0
    C[:, :, 0] = 0.0
    C[:, :, -1] = 0.0
    R[:, :, -1] = 0.0
    for time_index in range(model.N_DECISIONS - 1, -1, -1):
        cont_v = (P @ V[time_index + 1].T).T
        cont_c = (P @ C[time_index + 1].T).T
        cont_r = (P @ R[time_index + 1].T).T
        continuation_V[time_index] = cont_v
        continuation_C[time_index] = cont_c
        continuation_R[time_index] = cont_r

        for budget_index in range(model.BUDGETS.size):
            # q=0 is always considered first, so strict improvement is the
            # exact-tie no-op rule.
            selected_score = cont_r[budget_index].copy() if risk_only else cont_v[budget_index].copy()
            selected_c = cont_c[budget_index].copy()
            selected_r = cont_r[budget_index].copy()
            selected_q = np.zeros(n_states, dtype=np.float64)
            selected_successor = np.full(n_states, budget_index, dtype=np.int64)

            for action in model.actions_for_budget(budget_index):
                if abs(action) <= 1.0e-14:
                    continue
                next_budget = model.successor_budget_index(budget_index, action)
                continuation_v_action = _candidate_value(cont_v, grid, action, next_budget)
                continuation_c_action = _candidate_value(cont_c, grid, action, next_budget)
                continuation_r_action = _candidate_value(cont_r, grid, action, next_budget)
                candidate_score = (
                    continuation_r_action
                    if risk_only
                    else float(action) + continuation_v_action
                )
                improve = candidate_score < selected_score
                selected_score = np.where(improve, candidate_score, selected_score)
                selected_c = np.where(
                    improve,
                    float(action) + continuation_c_action,
                    selected_c,
                )
                selected_r = np.where(improve, continuation_r_action, selected_r)
                selected_q = np.where(improve, float(action), selected_q)
                selected_successor = np.where(improve, next_budget, selected_successor)

            # Sentinels never act and retain their declared values.
            selected_score[0] = failure_value
            selected_score[-1] = 0.0
            selected_c[0] = 0.0
            selected_c[-1] = 0.0
            selected_r[0] = 1.0
            selected_r[-1] = 0.0
            selected_q[[0, -1]] = 0.0
            selected_successor[[0, -1]] = budget_index
            V[time_index, budget_index] = selected_score
            C[time_index, budget_index] = selected_c
            R[time_index, budget_index] = selected_r
            actions[time_index, budget_index] = selected_q
            successor_budget[time_index, budget_index] = selected_successor

    identity_gap = 0.0
    if not risk_only:
        identity_gap = float(np.max(np.abs(V - (C + float(mu) * R))))
    solution = Solution(
        mu=None if risk_only else float(mu),
        risk_only=bool(risk_only),
        V=V,
        C=C,
        R=R,
        continuation_V=continuation_V,
        continuation_C=continuation_C,
        continuation_R=continuation_R,
        actions=actions,
        successor_budget=successor_budget,
        bellman_gap=0.0,
        identity_gap=float(identity_gap),
    )
    solution.bellman_gap = recompute_bellman_gap(solution, grid)
    return solution


def recompute_bellman_gap(solution: Solution, grid: np.ndarray) -> float:
    """Recompute the minimization residual from stored continuations."""

    grid = np.asarray(grid, dtype=np.float64)
    max_gap = 0.0
    for time_index in range(model.N_DECISIONS):
        for budget_index in range(model.BUDGETS.size):
            cont_v = solution.continuation_V[time_index]
            cont_r = solution.continuation_R[time_index]
            selected = cont_r[budget_index].copy() if solution.risk_only else cont_v[budget_index].copy()
            for action in model.actions_for_budget(budget_index):
                if abs(action) <= 1.0e-14:
                    continue
                next_budget = model.successor_budget_index(budget_index, action)
                candidate = _candidate_value(
                    cont_r if solution.risk_only else cont_v,
                    grid,
                    action,
                    next_budget,
                )
                if not solution.risk_only:
                    candidate = float(action) + candidate
                selected = np.minimum(selected, candidate)
            selected[0] = solution.V[time_index, budget_index, 0]
            selected[-1] = solution.V[time_index, budget_index, -1]
            max_gap = max(
                max_gap,
                float(np.max(np.abs(selected[1:-1] - solution.V[time_index, budget_index, 1:-1]))),
            )
    return float(max_gap)


def evaluate_policy(actions: np.ndarray, P: np.ndarray, grid: np.ndarray) -> PolicyEvaluation:
    """Evaluate fixed stored actions without calling the Bellman solver."""

    actions = np.asarray(actions, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    expected_shape = (model.N_DECISIONS, model.BUDGETS.size, grid.size)
    if actions.shape != expected_shape:
        raise ValueError(f"actions shape {actions.shape} != {expected_shape}")
    C = np.zeros((model.N_DECISIONS + 1, model.BUDGETS.size, grid.size), dtype=np.float64)
    R = np.zeros_like(C)
    C[:, :, 0] = 0.0
    R[:, :, 0] = 1.0
    C[:, :, -1] = 0.0
    R[:, :, -1] = 0.0
    invalid: str | None = None
    feasible = True
    for time_index in range(model.N_DECISIONS - 1, -1, -1):
        cont_c = (P @ C[time_index + 1].T).T
        cont_r = (P @ R[time_index + 1].T).T
        for budget_index in range(model.BUDGETS.size):
            for state_index, moisture in enumerate(grid):
                if state_index == 0 or state_index == grid.size - 1:
                    continue
                action = float(actions[time_index, budget_index, state_index])
                try:
                    next_budget = model.successor_budget_index(budget_index, action)
                except ValueError as error:
                    feasible = False
                    invalid = str(error)
                    continue
                if action < -1.0e-10 or action > float(model.BUDGETS[budget_index]) + 1.0e-10:
                    feasible = False
                    invalid = f"budget violation {action} at {budget_index}"
                    continue
                shifted = float(moisture) + action
                C[time_index, budget_index, state_index] = action + float(
                    model.interpolate_column(
                        cont_c[next_budget], shifted, grid
                    )
                )
                R[time_index, budget_index, state_index] = float(
                    model.interpolate_column(cont_r[next_budget], shifted, grid)
                )
    return PolicyEvaluation(C=C, R=R, budget_feasible=feasible, invalid_action=invalid)


def solution_summary(solution: Solution, evaluation: PolicyEvaluation) -> dict[str, Any]:
    discrepancy_r = float(np.max(np.abs(solution.R - evaluation.R)))
    discrepancy_c = float(np.max(np.abs(solution.C - evaluation.C)))
    return {
        "mu": solution.mu,
        "risk_only": solution.risk_only,
        "bellman_gap": solution.bellman_gap,
        "identity_gap": solution.identity_gap,
        "policy_eval_R_gap": discrepancy_r,
        "policy_eval_C_gap": discrepancy_c,
        "policy_budget_feasible": bool(evaluation.budget_feasible),
        "policy_invalid_action": evaluation.invalid_action,
        "finite": bool(
            np.isfinite(solution.V).all()
            and np.isfinite(solution.C).all()
            and np.isfinite(solution.R).all()
            and np.isfinite(solution.actions).all()
        ),
        "risk_min": float(np.min(solution.R)),
        "risk_max": float(np.max(solution.R)),
        "cost_min": float(np.min(solution.C)),
        "cost_max": float(np.max(solution.C)),
    }
