"""Constrained Lagrange/Bellman reference with fixed bisection protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import bellman
import model


BRACKET_MUS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
BISECTION_STEPS = 24
MECHANICAL_TOL = 1.0e-10


@dataclass
class TrialRecord:
    label: str
    solution: bellman.Solution
    evaluation: bellman.PolicyEvaluation
    summary: dict[str, Any]


@dataclass
class RowResult:
    row: model.ModelRow
    status: str
    trials: list[TrialRecord] = field(default_factory=list)
    initial_risk_min: float = float("nan")
    initial_risk_no_cost: float = float("nan")
    initial_cost: float = float("nan")
    initial_risk: float = float("nan")
    risky_label: str | None = None
    safe_label: str | None = None
    mixture_alpha: float = 0.0
    mixture_cost: float = float("nan")
    mixture_risk: float = float("nan")
    dual_bound: float = float("nan")
    duality_gap: float = float("nan")
    gate_summary: dict[str, Any] = field(default_factory=dict)


def _initial_scalar(array: np.ndarray, grid: np.ndarray) -> float:
    return float(model.interpolate_column(array[0, -1], model.W0, grid))


def _record_trial(
    records: list[TrialRecord],
    label: str,
    solution: bellman.Solution,
    row: model.ModelRow,
    callback: Callable[[TrialRecord], None] | None,
) -> TrialRecord:
    evaluation = bellman.evaluate_policy(solution.actions, row.P, row.grid)
    summary = bellman.solution_summary(solution, evaluation)
    summary.update(
        {
            "label": label,
            "initial_V": _initial_scalar(solution.V, row.grid),
            "initial_C": _initial_scalar(solution.C, row.grid),
            "initial_R": _initial_scalar(solution.R, row.grid),
        }
    )
    record = TrialRecord(label=label, solution=solution, evaluation=evaluation, summary=summary)
    records.append(record)
    if callback is not None:
        callback(record)
    return record


def _finite_trial_dual(records: list[TrialRecord]) -> float:
    values = [
        float(record.summary["initial_V"]) - float(record.solution.mu) * model.EPSILON
        for record in records
        if not record.solution.risk_only and record.solution.mu is not None
    ]
    return max(values) if values else float("nan")


def _row_mechanical_gate(records: list[TrialRecord], mixture_risk: float, duality_gap: float) -> dict[str, Any]:
    finite = all(bool(record.summary["finite"]) for record in records)
    bellman = all(float(record.summary["bellman_gap"]) <= 1.0e-9 for record in records)
    identity = all(
        record.solution.risk_only or float(record.summary["identity_gap"]) <= 1.0e-8
        for record in records
    )
    evaluator = all(
        float(record.summary["policy_eval_R_gap"]) <= 1.0e-10
        and float(record.summary["policy_eval_C_gap"]) <= 1.0e-10
        for record in records
    )
    risk_bounds = all(
        -1.0e-10 <= float(record.summary["risk_min"]) <= 1.0 + 1.0e-10
        and -1.0e-10 <= float(record.summary["risk_max"]) <= 1.0 + 1.0e-10
        for record in records
    )
    cost_bounds = all(
        -1.0e-10 <= float(record.summary["cost_min"]) <= 0.30 + 1.0e-10
        and -1.0e-10 <= float(record.summary["cost_max"]) <= 0.30 + 1.0e-10
        for record in records
    )
    action_bounds = all(bool(record.summary["policy_budget_feasible"]) for record in records)
    mixture = (
        np.isfinite(mixture_risk)
        and mixture_risk <= model.EPSILON + 1.0e-10
        and np.isfinite(duality_gap)
        and -1.0e-8 <= duality_gap <= 1.0e-5
    )
    return {
        "all_trials_finite": finite,
        "bellman_residual_le_1e-9": bellman,
        "identity_le_1e-8": identity,
        "fixed_policy_discrepancy_le_1e-10": evaluator,
        "risk_bounds": risk_bounds,
        "cost_bounds": cost_bounds,
        "executed_actions_budget_feasible": action_bounds,
        "mixture_risk_and_duality": bool(mixture),
        "pass": bool(
            finite
            and bellman
            and identity
            and evaluator
            and risk_bounds
            and cost_bounds
            and action_bounds
            and mixture
        ),
    }


def solve_row(
    row: model.ModelRow,
    on_trial: Callable[[TrialRecord], None] | None = None,
) -> RowResult:
    records: list[TrialRecord] = []
    risk_only = _record_trial(
        records,
        "risk_only",
        bellman.solve(None, row.P, row.grid, risk_only=True),
        row,
        on_trial,
    )
    initial_risk_min = float(risk_only.summary["initial_R"])
    result = RowResult(row=row, status="PENDING", trials=records, initial_risk_min=initial_risk_min)
    if initial_risk_min > model.EPSILON + 1.0e-10:
        result.status = "INFEASIBLE"
        result.gate_summary = _row_mechanical_gate(records, float("nan"), float("nan"))
        result.gate_summary["minimum_risk_feasible"] = False
        result.gate_summary["pass"] = False
        return result

    mu_zero = _record_trial(
        records,
        "mu_0",
        bellman.solve(0.0, row.P, row.grid),
        row,
        on_trial,
    )
    result.initial_risk_no_cost = float(mu_zero.summary["initial_R"])
    result.initial_cost = float(mu_zero.summary["initial_C"])
    result.initial_risk = result.initial_risk_no_cost

    risky: TrialRecord | None = None
    safe: TrialRecord | None = None
    if result.initial_risk_no_cost <= model.EPSILON + 1.0e-10:
        safe = mu_zero
        result.safe_label = safe.label
    else:
        previous = mu_zero
        for mu in BRACKET_MUS:
            trial = _record_trial(
                records,
                f"mu_{mu:g}",
                bellman.solve(mu, row.P, row.grid),
                row,
                on_trial,
            )
            if float(trial.summary["initial_R"]) <= model.EPSILON + 1.0e-10:
                risky = previous
                safe = trial
                break
            previous = trial
        if safe is None:
            result.status = "BLOCKED_LAGRANGE_BRACKET"
            result.gate_summary = _row_mechanical_gate(records, float("nan"), float("nan"))
            result.gate_summary["minimum_risk_feasible"] = True
            result.gate_summary["pass"] = False
            return result

        low = risky
        high = safe
        assert low is not None and high is not None
        for iteration in range(BISECTION_STEPS):
            assert low.solution.mu is not None and high.solution.mu is not None
            midpoint = 0.5 * (float(low.solution.mu) + float(high.solution.mu))
            trial = _record_trial(
                records,
                f"bisection_{iteration + 1:02d}_mu_{midpoint:.17g}",
                bellman.solve(midpoint, row.P, row.grid),
                row,
                on_trial,
            )
            if float(trial.summary["initial_R"]) <= model.EPSILON + 1.0e-10:
                high = trial
            else:
                low = trial
        risky = low
        safe = high

    assert safe is not None
    result.risky_label = None if risky is None else risky.label
    result.safe_label = safe.label
    safe_risk = float(safe.summary["initial_R"])
    safe_cost = float(safe.summary["initial_C"])
    if risky is None:
        alpha = 0.0
        mixture_risk = safe_risk
        mixture_cost = safe_cost
    else:
        risky_risk = float(risky.summary["initial_R"])
        risky_cost = float(risky.summary["initial_C"])
        denominator = risky_risk - safe_risk
        if denominator <= 1.0e-14 or risky_risk <= model.EPSILON + 1.0e-10:
            alpha = 0.0
            mixture_risk = safe_risk
            mixture_cost = safe_cost
        else:
            alpha = (model.EPSILON - safe_risk) / denominator
            mixture_risk = alpha * risky_risk + (1.0 - alpha) * safe_risk
            mixture_cost = alpha * risky_cost + (1.0 - alpha) * safe_cost
    result.mixture_alpha = float(alpha)
    result.mixture_risk = float(mixture_risk)
    result.mixture_cost = float(mixture_cost)
    result.dual_bound = _finite_trial_dual(records)
    result.duality_gap = float(result.mixture_cost - result.dual_bound)
    result.status = "FEASIBLE"
    result.gate_summary = _row_mechanical_gate(records, result.mixture_risk, result.duality_gap)
    result.gate_summary["minimum_risk_feasible"] = True
    result.gate_summary["pass"] = bool(result.gate_summary["pass"])
    return result


def row_summary(result: RowResult) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "row_key": result.row.row_key,
        "upper": result.row.upper,
        "h": result.row.h,
        "intensity": result.row.intensity,
        "state_count": result.row.state_count,
        "status": result.status,
        "trial_count": len(result.trials),
        "initial_risk_min": finite_or_none(result.initial_risk_min),
        "initial_risk_no_cost": finite_or_none(result.initial_risk_no_cost),
        "initial_cost": finite_or_none(result.initial_cost),
        "initial_risk": finite_or_none(result.initial_risk),
        "risky_label": result.risky_label,
        "safe_label": result.safe_label,
        "mixture_alpha": finite_or_none(result.mixture_alpha),
        "mixture_cost": finite_or_none(result.mixture_cost),
        "mixture_risk": finite_or_none(result.mixture_risk),
        "dual_bound": finite_or_none(result.dual_bound),
        "duality_gap": finite_or_none(result.duality_gap),
        "gate_summary": result.gate_summary,
        "trials": [record.summary for record in result.trials],
    }
