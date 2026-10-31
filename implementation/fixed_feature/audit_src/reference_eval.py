"""Independent central-generator fixed-policy reference and CPU-f64 audit."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import expm_multiply

from model import (
    DAYS,
    TRACE_MAX_ERROR,
    FixedPolicyPINN,
    apply_policy_at_day_start,
    pide_residual,
)
from operators import BUDGETS, WD, DIFFUSION, action_probability
from reference_model import build_model_row

UPPER = 1.55
GRID_H = 0.0025
BUDGET_LEVELS = np.asarray(BUDGETS, dtype=np.float64)
POLICIES = ("zero", "schedule", "buffer")
INTENSITIES = (0.0, 0.12)
SEEDS = (20260928, 20260929)
PIDE_MEDIAN_GATE = 0.005
PIDE_P95_GATE = 0.02
RISK_P95_GATE = 0.02
RISK_LOWER_P95_GATE = 0.05
COST_P95_GATE = 0.03
INITIAL_GATE = 0.02
INTERFACE_GATE = 1.0e-8
OUTPUT_MIN = -0.02
OUTPUT_MAX = 1.02
TRACE_GATE = TRACE_MAX_ERROR
TAU_AUDIT = (0.0, 0.25, 0.5, 0.75, 1.0)
PIDE_AUDIT_PER_DAY = 320
PIDE_AUDIT_SEED = 991831
VALUE_CHUNK = 8192
PIDE_CHUNK = 128


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=32)
def _cached_row(upper: float, h: float, intensity: float):
    return build_model_row(float(upper), float(h), float(intensity))


def action_numpy(policy: str, day: int, water: float, budget_index: int) -> tuple[float, float, int]:
    beta = float(BUDGETS[budget_index])
    q = min(0.08, beta)
    if policy == "zero":
        p = 0.0
    elif policy == "schedule":
        p = 1.0 if day <= 3 and beta > 0.0 else 0.0
    elif policy == "buffer":
        p = float(np.clip((WD + 2.0 * DIFFUSION - water) / DIFFUSION, 0.0, 1.0)) if beta > 0.0 else 0.0
    else:
        raise ValueError(policy)
    if water <= WD or water >= UPPER:
        p = 0.0
    if p <= 0.0:
        q = 0.0
    return p, q, max(0, budget_index - 1) if q > 0.0 else budget_index


def _interpolate(grid: np.ndarray, values: np.ndarray, x: float) -> float:
    return float(np.interp(float(x), grid, values))


def apply_reference_policy(
    policy: str, day: int, grid: np.ndarray, post_values: np.ndarray
) -> np.ndarray:
    """Map post-action values `[budget, water, (cost,risk)]` to pre-action."""
    result = np.zeros_like(post_values)
    for bi in range(len(BUDGETS)):
        for wi, water in enumerate(grid):
            probability, amount, successor = action_numpy(policy, day, float(water), bi)
            c0 = _interpolate(grid, post_values[bi, :, 0], float(water))
            r0 = _interpolate(grid, post_values[bi, :, 1], float(water))
            c1 = _interpolate(grid, post_values[successor, :, 0], float(water + amount))
            r1 = _interpolate(grid, post_values[successor, :, 1], float(water + amount))
            result[bi, wi, 0] = (1.0 - probability) * c0 + probability * (amount + c1)
            result[bi, wi, 1] = (1.0 - probability) * r0 + probability * r1
    return result


class FixedPolicyReference:
    def __init__(self, policy: str, intensity: float, upper: float = UPPER, h: float = GRID_H):
        self.policy = str(policy)
        self.intensity = float(intensity)
        self.upper = float(upper)
        self.h = float(h)
        self.row = _cached_row(self.upper, self.h, self.intensity)
        self.grid = np.asarray(self.row.grid, dtype=np.float64)
        self._tau_cache: dict[tuple[float, ...], dict[float, np.ndarray]] = {}
        self.post = np.zeros((DAYS + 1, len(BUDGETS), self.grid.size, 2), dtype=np.float64)
        self.post[DAYS, :, 0, 1] = 1.0
        self.day_terminal = np.zeros((DAYS, len(BUDGETS), self.grid.size, 2), dtype=np.float64)
        for day in range(DAYS - 1, -1, -1):
            terminal = (
                self.post[DAYS]
                if day == DAYS - 1
                else apply_reference_policy(self.policy, day + 1, self.grid, self.post[day + 1])
            )
            self.day_terminal[day] = terminal
            terminal_matrix = terminal.transpose(1, 0, 2).reshape(self.grid.size, -1)
            propagated = self.row.P @ terminal_matrix
            self.post[day] = propagated.reshape(self.grid.size, len(BUDGETS), 2).transpose(1, 0, 2)
        self.pre = np.zeros_like(self.post)
        self.pre[DAYS] = self.post[DAYS]
        for day in range(DAYS):
            self.pre[day] = apply_reference_policy(self.policy, day, self.grid, self.post[day])

    def initial(self, water: float = 0.60, budget: float = 0.30) -> tuple[float, float]:
        bi = int(np.argmin(np.abs(BUDGET_LEVELS - float(budget))))
        cost = _interpolate(self.grid, self.pre[0, bi, :, 0], water)
        risk = _interpolate(self.grid, self.pre[0, bi, :, 1], water)
        return cost, risk

    def values_at_taus(self, taus: tuple[float, ...] = TAU_AUDIT) -> dict[float, np.ndarray]:
        cache_key = tuple(float(value) for value in taus)
        if cache_key in self._tau_cache:
            return self._tau_cache[cache_key]
        terminal = self.day_terminal.transpose(2, 0, 1, 3).reshape(self.grid.size, -1)
        result: dict[float, np.ndarray] = {}
        for tau in taus:
            if np.isclose(tau, 0.0):
                propagated = self.row.P @ terminal
            elif np.isclose(tau, 1.0):
                propagated = terminal
            else:
                propagated = expm_multiply((1.0 - float(tau)) * self.row.Q, terminal)
            result[float(tau)] = np.asarray(propagated).reshape(
                self.grid.size, DAYS, len(BUDGETS), 2
            ).transpose(1, 2, 0, 3)
        self._tau_cache[cache_key] = result
        return result

    def initial_sensitivities(self) -> list[dict[str, object]]:
        specifications = (
            (1.55, 0.01, "grid_h_0.01"),
            (1.55, 0.005, "grid_h_0.005"),
            (1.15, 0.005, "upper_1.15_h_0.005"),
            (1.95, 0.005, "upper_1.95_h_0.005"),
        )
        primary = self.initial()
        rows = []
        for upper, h, label in specifications:
            alternative = get_reference(self.policy, self.intensity, upper, h)
            value = alternative.initial()
            rows.append(
                {
                    "label": label,
                    "upper": upper,
                    "h": h,
                    "expected_water": value[0],
                    "risk": value[1],
                    "abs_difference_expected_water": abs(value[0] - primary[0]),
                    "abs_difference_risk": abs(value[1] - primary[1]),
                }
            )
        return rows


def _deposit_mass(grid: np.ndarray, target: np.ndarray, budget: int, water: float, mass: float) -> None:
    if mass == 0.0:
        return
    if water <= float(grid[0]):
        target[int(budget), 0] += mass
        return
    if water >= float(grid[-1]):
        target[int(budget), -1] += mass
        return
    position = (float(water) - float(grid[0])) / float(grid[1] - grid[0])
    lower = int(np.floor(position))
    fraction = position - lower
    target[int(budget), lower] += mass * (1.0 - fraction)
    target[int(budget), lower + 1] += mass * fraction


def forward_policy_ledger(reference: FixedPolicyReference) -> dict[str, object]:
    """Propagate the reference state law forward and audit C/R duality."""
    grid = reference.grid
    mass = np.zeros((len(BUDGETS), grid.size), dtype=np.float64)
    initial_position = (0.60 - float(grid[0])) / float(grid[1] - grid[0])
    initial_lower = int(np.floor(initial_position))
    initial_fraction = initial_position - initial_lower
    initial_budget = int(np.argmin(np.abs(BUDGET_LEVELS - 0.30)))
    mass[initial_budget, initial_lower] += 1.0 - initial_fraction
    mass[initial_budget, initial_lower + 1] += initial_fraction
    daily_rows = []
    cumulative_water = 0.0
    for day in range(DAYS):
        post_action = np.zeros_like(mass)
        daily_water = 0.0
        daily_pulse_count = 0.0
        for budget_index in range(len(BUDGETS)):
            for water_index, water in enumerate(grid):
                state_mass = float(mass[budget_index, water_index])
                if state_mass == 0.0:
                    continue
                probability, amount, successor = action_numpy(
                    reference.policy, day, float(water), budget_index
                )
                daily_water += state_mass * probability * amount
                daily_pulse_count += state_mass * probability
                _deposit_mass(
                    grid,
                    post_action,
                    budget_index,
                    float(water),
                    state_mass * (1.0 - probability),
                )
                _deposit_mass(
                    grid,
                    post_action,
                    successor,
                    float(water + amount),
                    state_mass * probability,
                )
        mass = post_action @ reference.row.P
        cumulative_water += daily_water
        daily_rows.append(
            {
                "day": day,
                "expected_water_pulse": daily_water,
                "expected_positive_action_count": daily_pulse_count,
                "cumulative_expected_water": cumulative_water,
                "failure_mass_after_dynamics": float(np.sum(mass[:, 0])),
                "safe_sentinel_mass_after_dynamics": float(np.sum(mass[:, -1])),
                "total_mass": float(np.sum(mass)),
            }
        )
    backward_cost, backward_risk = reference.initial()
    water_error = abs(cumulative_water - backward_cost)
    risk_error = abs(float(np.sum(mass[:, 0])) - backward_risk)
    return {
        "daily": daily_rows,
        "forward_expected_water": cumulative_water,
        "backward_expected_water": backward_cost,
        "expected_water_abs_difference": water_error,
        "forward_terminal_failure_mass": float(np.sum(mass[:, 0])),
        "backward_terminal_failure_risk": backward_risk,
        "risk_abs_difference": risk_error,
        "closure_tolerance": 1.0e-10,
        "pass": bool(water_error <= 1.0e-10 and risk_error <= 1.0e-10),
    }


_REFERENCE_CACHE: dict[tuple[str, float, float, float], FixedPolicyReference] = {}


def get_reference(
    policy: str, intensity: float, upper: float = UPPER, h: float = GRID_H
) -> FixedPolicyReference:
    key = (str(policy), float(intensity), float(upper), float(h))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = FixedPolicyReference(*key)
    return _REFERENCE_CACHE[key]


def load_model(export_path: Path) -> tuple[FixedPolicyPINN, dict[str, object]]:
    payload = torch.load(export_path, map_location="cpu", weights_only=True)
    model = FixedPolicyPINN(str(payload["policy"]), float(payload["intensity"])).double()
    model.load_state_dict(payload["model"])
    model.eval()
    if any(parameter.dtype != torch.float64 or parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError(f"audit model is not CPU float64: {export_path}")
    if any(buffer.device.type != "cpu" for _, buffer in model.named_buffers()):
        raise RuntimeError(f"audit buffers are not on CPU: {export_path}")
    return model, payload


def model_values(model: FixedPolicyPINN, day: int, states: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), VALUE_CHUNK):
            state = torch.from_numpy(np.asarray(states[start : start + VALUE_CHUNK], dtype=np.float64))
            budget = torch.from_numpy(np.asarray(budgets[start : start + VALUE_CHUNK], dtype=np.int64))
            outputs.append(model.forward_day(day, state, budget).numpy())
    return np.concatenate(outputs, axis=0)


def daily_value_errors(
    model: FixedPolicyPINN, reference: FixedPolicyReference
) -> dict[str, object]:
    ref_by_tau = reference.values_at_taus()
    all_cost_errors: list[np.ndarray] = []
    all_risk_errors: list[np.ndarray] = []
    lower_cost_errors: list[np.ndarray] = []
    lower_risk_errors: list[np.ndarray] = []
    all_model_values: list[np.ndarray] = []
    max_record: dict[str, object] = {"abs_error": -1.0}
    lower_offsets = np.asarray((0.0025, 0.005, 0.01, 0.02, 0.04), dtype=np.float64)

    for day in range(DAYS):
        for tau in TAU_AUDIT:
            for budget_index in range(len(BUDGETS)):
                water = reference.grid
                states = np.column_stack((np.full(water.size, tau), water))
                budgets = np.full(water.size, budget_index, dtype=np.int64)
                values = model_values(model, day, states, budgets)
                target = ref_by_tau[float(tau)][day, budget_index]
                error = np.abs(values - target)
                all_cost_errors.append(error[:, 0])
                all_risk_errors.append(error[:, 1])
                all_model_values.append(values)
                where = np.unravel_index(int(np.argmax(error)), error.shape)
                candidate = float(error[where])
                if candidate > float(max_record["abs_error"]):
                    max_record = {
                        "abs_error": candidate,
                        "channel": "expected_water" if where[1] == 0 else "risk",
                        "day": day,
                        "local_time": float(tau),
                        "budget": float(BUDGETS[budget_index]),
                        "water": float(water[where[0]]),
                    }

                lower_water = WD + lower_offsets
                lower_states = np.column_stack((np.full(lower_water.size, tau), lower_water))
                lower_budgets = np.full(lower_water.size, budget_index, dtype=np.int64)
                lower_values = model_values(model, day, lower_states, lower_budgets)
                target_cost = np.interp(lower_water, water, target[:, 0])
                target_risk = np.interp(lower_water, water, target[:, 1])
                lower_cost_errors.append(np.abs(lower_values[:, 0] - target_cost))
                lower_risk_errors.append(np.abs(lower_values[:, 1] - target_risk))

    cost = np.concatenate(all_cost_errors)
    risk = np.concatenate(all_risk_errors)
    lower_cost = np.concatenate(lower_cost_errors)
    lower_risk = np.concatenate(lower_risk_errors)
    all_values = np.concatenate([values.reshape(-1) for values in all_model_values])
    return {
        "grid": {
            "day_count": DAYS,
            "tau_values": list(TAU_AUDIT),
            "budget_values": list(BUDGETS),
            "water_points_per_slice": int(reference.grid.size),
            "compared_value_count_per_channel": int(cost.size),
        },
        "expected_water": {
            "p95_abs_error": float(np.quantile(cost, 0.95)),
            "max_abs_error": float(np.max(cost)),
            "initial_abs_error": None,
        },
        "risk": {
            "p95_abs_error": float(np.quantile(risk, 0.95)),
            "max_abs_error": float(np.max(risk)),
            "initial_abs_error": None,
        },
        "lower_layer": {
            "p95_abs_error_expected_water": float(np.quantile(lower_cost, 0.95)),
            "max_abs_error_expected_water": float(np.max(lower_cost)),
            "p95_abs_error_risk": float(np.quantile(lower_risk, 0.95)),
            "max_abs_error_risk": float(np.max(lower_risk)),
        },
        "maximum_error_location": max_record,
        "output_range": {"min": float(np.min(all_values)), "max": float(np.max(all_values))},
        "finite": bool(np.isfinite(all_values).all()),
    }


def interface_metrics(model: FixedPolicyPINN, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(int(seed) + 909090)
    water = rng.uniform(WD + 0.001, UPPER - 0.001, 1024)
    water_t = torch.from_numpy(water.astype(np.float64))
    budgets = np.arange(water.size, dtype=np.int64) % len(BUDGETS)
    budget_t = torch.from_numpy(budgets)
    by_day: dict[str, float] = {}
    with torch.no_grad():
        for day in range(DAYS - 1):
            states = torch.stack((torch.ones_like(water_t), water_t), dim=1)
            values = model.forward_day(day, states, budget_t)
            traces = model.interpolate_trace(day, water_t, budget_t)
            by_day[str(day)] = float(torch.max(torch.abs(values - traces)))
    maximum = max(by_day.values())
    return {"by_day_max_abs_error": by_day, "max_abs_error": maximum, "pass": maximum <= INTERFACE_GATE}


def terminal_and_boundary_metrics(model: FixedPolicyPINN) -> dict[str, float]:
    times = torch.tensor((0.0, 0.2, 0.5, 0.8, 1.0), dtype=torch.float64).repeat(2)
    water = torch.cat(
        (torch.full((5,), WD, dtype=torch.float64), torch.full((5,), UPPER, dtype=torch.float64))
    )
    states = torch.stack((times, water), dim=1)
    budgets = torch.tensor((0, 1, 2, 3, 4, 0, 1, 2, 3, 4), dtype=torch.long)
    maxima = {"boundary": 0.0, "terminal": 0.0}
    with torch.no_grad():
        for day in range(DAYS):
            values = model.forward_day(day, states, budgets)
            expected_cost = torch.zeros_like(times)
            expected_risk = torch.cat((torch.ones(5), torch.zeros(5)))
            expected = torch.stack((expected_cost, expected_risk), dim=1)
            maxima["boundary"] = max(
                maxima["boundary"], float(torch.max(torch.abs(values - expected)))
            )
        terminal_water = torch.linspace(WD + 0.001, UPPER - 0.001, 1024, dtype=torch.float64)
        terminal_states = torch.stack((torch.ones_like(terminal_water), terminal_water), dim=1)
        budgets = torch.arange(terminal_water.numel(), dtype=torch.long) % len(BUDGETS)
        terminal = model.forward_day(DAYS - 1, terminal_states, budgets)
        maxima["terminal"] = float(torch.max(torch.abs(terminal)))
    return maxima


def budget_feasibility_metrics(model: FixedPolicyPINN) -> dict[str, object]:
    water_values = torch.tensor(
        (WD, WD + 1.0e-5, 0.41, 0.44, 0.47, 0.60, 1.54999, UPPER),
        dtype=torch.float64,
    )
    violations = []
    rows = 0
    budget_levels = torch.as_tensor(BUDGETS, dtype=torch.float64)
    for day in range(DAYS):
        for budget_index in range(len(BUDGETS)):
            water = water_values
            index = torch.full((water.numel(),), budget_index, dtype=torch.long)
            probability, amount, successor = action_probability(model.policy, day, water, index)
            available = budget_levels[index]
            remaining = budget_levels[successor]
            violations.append(torch.clamp(amount - available, min=0.0).max())
            expected_remaining = torch.where(amount > 0.0, available - amount, available)
            violations.append(torch.abs(remaining - expected_remaining).max())
            violations.append(torch.where(probability <= 0.0, amount, torch.zeros_like(amount)).max())
            rows += int(water.numel())
    maximum = float(torch.stack(violations).max()) if violations else 0.0
    return {"evaluated_decision_states": rows, "max_abs_violation": maximum,
            "pass": maximum <= 1.0e-12}


def pide_residual_metrics(model: FixedPolicyPINN, seed: int) -> dict[str, object]:
    abs_values: list[np.ndarray] = []
    lower_values: list[np.ndarray] = []
    rng = np.random.default_rng(PIDE_AUDIT_SEED)
    for day in range(DAYS):
        ordinary = np.column_stack(
            (rng.uniform(0.0, 1.0, PIDE_AUDIT_PER_DAY), rng.uniform(WD + 1.0e-4, UPPER - 1.0e-4, PIDE_AUDIT_PER_DAY))
        )
        offsets = np.asarray((0.0025, 0.005, 0.01, 0.02, 0.04), dtype=np.float64)
        lower = np.column_stack(
            (rng.uniform(0.0, 1.0, PIDE_AUDIT_PER_DAY), WD + rng.choice(offsets, PIDE_AUDIT_PER_DAY))
        )
        budgets = np.arange(PIDE_AUDIT_PER_DAY, dtype=np.int64) % len(BUDGETS)
        day_residual = []
        day_lower = []
        for states_np, target in ((ordinary, day_residual), (lower, day_lower)):
            for start in range(0, len(states_np), PIDE_CHUNK):
                states = torch.from_numpy(states_np[start : start + PIDE_CHUNK].copy()).requires_grad_(True)
                budget = torch.from_numpy(budgets[start : start + PIDE_CHUNK])
                with torch.enable_grad():
                    residual = pide_residual(model, day, states, budget)
                target.append(np.abs(residual.detach().numpy()).reshape(-1))
        abs_values.append(np.concatenate(day_residual))
        lower_values.append(np.concatenate(day_lower))
    ordinary_abs = np.concatenate(abs_values)
    lower_abs = np.concatenate(lower_values)
    return {
        "sample_seed": PIDE_AUDIT_SEED,
        "training_seed_not_used_for_audit_sampling": int(seed),
        "ordinary_count": int(ordinary_abs.size),
        "lower_layer_count": int(lower_abs.size),
        "ordinary_median_abs": float(np.median(ordinary_abs)),
        "ordinary_p95_abs": float(np.quantile(ordinary_abs, 0.95)),
        "ordinary_max_abs": float(np.max(ordinary_abs)),
        "lower_layer_p95_abs": float(np.quantile(lower_abs, 0.95)),
        "lower_layer_max_abs": float(np.max(lower_abs)),
        "finite": bool(np.isfinite(ordinary_abs).all() and np.isfinite(lower_abs).all()),
        "ordinary_abs_values": ordinary_abs,
        "lower_layer_abs_values": lower_abs,
    }


def branch_audit(export_path: Path, branch_dir: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model, payload = load_model(export_path)
    policy = str(payload["policy"])
    intensity = float(payload["intensity"])
    seed = int(payload["seed"])
    reference = get_reference(policy, intensity)
    values = daily_value_errors(model, reference)
    reference_initial = reference.initial()
    budget_index = int(np.argmin(np.abs(BUDGET_LEVELS - 0.30)))
    water = torch.tensor((0.60,), dtype=torch.float64)
    budgets = torch.tensor((budget_index,), dtype=torch.long)
    with torch.no_grad():
        initial = apply_policy_at_day_start(model, 0, water, budgets)[0].numpy()
    initial_errors = {
        "expected_water": abs(float(initial[0]) - reference_initial[0]),
        "risk": abs(float(initial[1]) - reference_initial[1]),
    }
    values["expected_water"]["initial_abs_error"] = initial_errors["expected_water"]
    values["risk"]["initial_abs_error"] = initial_errors["risk"]
    interface = interface_metrics(model, seed)
    traces = payload.get("trace_records", [])
    trace_max = max(
        [float(row[f"max_abs_error_{channel}"]) for row in traces for channel in ("cost", "risk")]
        or [0.0]
    )
    endpoints = terminal_and_boundary_metrics(model)
    budget = budget_feasibility_metrics(model)
    pide_all = pide_residual_metrics(model, seed)
    residual_distribution = {
        "ordinary_abs_values": pide_all.pop("ordinary_abs_values"),
        "lower_layer_abs_values": pide_all.pop("lower_layer_abs_values"),
    }
    pide = pide_all
    ledger = forward_policy_ledger(reference)
    sensitivity = reference.initial_sensitivities()
    trace_arrays = {"water_grid": model.trace_grid.detach().cpu().numpy()}
    for day in range(DAYS - 1):
        trace_arrays[f"day_{day:02d}"] = getattr(model, f"terminal_trace_{day}").detach().cpu().numpy()
    return ({
        "policy": policy,
        "intensity": intensity,
        "seed": seed,
        "export_sha256": _sha256(export_path),
        "model_values": values,
        "initial": {
            "model_expected_water": float(initial[0]),
            "reference_expected_water": reference_initial[0],
            "model_risk": float(initial[1]),
            "reference_risk": reference_initial[1],
            "absolute_errors": initial_errors,
        },
        "interface": interface,
        "terminal_boundary": endpoints,
        "budget_feasibility": budget,
        "forward_policy_ledger": ledger,
        "interface_trace_shape": {
            "day_traces": DAYS - 1,
            "budget_levels": len(BUDGETS),
            "water_nodes": int(model.trace_grid.numel()),
            "channels": ["expected_water", "risk"],
        },
        "trace_interpolation": {
            "day_count": len(traces),
            "max_abs_error_all_channels": trace_max,
            "gate": TRACE_GATE,
            "pass": trace_max <= TRACE_GATE,
        },
        "pide_residual": pide,
        "reference_sensitivity": sensitivity,
        "fallback": None,
        "stderr": "",
    }, {"residuals": residual_distribution, "traces": trace_arrays})


def _gate_branch(row: dict[str, object]) -> dict[str, object]:
    values = row["model_values"]
    pide = row["pide_residual"]
    endpoints = row["terminal_boundary"]
    checks = {
        "finite_outputs": bool(values["finite"] and pide["finite"]),
        "expected_water_p95": values["expected_water"]["p95_abs_error"] <= COST_P95_GATE,
        "expected_water_initial": values["expected_water"]["initial_abs_error"] <= INITIAL_GATE,
        "risk_p95": values["risk"]["p95_abs_error"] <= RISK_P95_GATE,
        "risk_lower_layer_p95": values["lower_layer"]["p95_abs_error_risk"] <= RISK_LOWER_P95_GATE,
        "risk_initial": values["risk"]["initial_abs_error"] <= INITIAL_GATE,
        "pide_median": pide["ordinary_median_abs"] <= PIDE_MEDIAN_GATE,
        "pide_p95": pide["ordinary_p95_abs"] <= PIDE_P95_GATE,
        "boundary_exact": endpoints["boundary"] <= INTERFACE_GATE,
        "terminal_exact": endpoints["terminal"] <= INTERFACE_GATE,
        "budget_feasible": row["budget_feasibility"]["pass"],
        "forward_backward_ledger_closure": row["forward_policy_ledger"]["pass"],
        "daily_interface": row["interface"]["max_abs_error"] <= INTERFACE_GATE,
        "trace_interpolation": row["trace_interpolation"]["pass"],
        "output_range": values["output_range"]["min"] >= OUTPUT_MIN and values["output_range"]["max"] <= OUTPUT_MAX,
    }
    return {"branch": f"{row['policy']}_lambda_{row['intensity']:g}_seed_{row['seed']}",
            "checks": checks, "pass": bool(all(checks.values()))}


def paired_prediction_difference(
    first_path: Path, second_path: Path, reference: FixedPolicyReference
) -> dict[str, object]:
    first, first_payload = load_model(first_path)
    second, second_payload = load_model(second_path)
    if first_payload["policy"] != second_payload["policy"] or first_payload["intensity"] != second_payload["intensity"]:
        raise ValueError("paired-seed models do not share a frozen policy and intensity")
    differences = {"expected_water": [], "risk": []}
    maxima = {
        "expected_water": {"abs_error": -1.0},
        "risk": {"abs_error": -1.0},
    }
    for day in range(DAYS):
        for tau in TAU_AUDIT:
            for budget_index, budget_value in enumerate(BUDGETS):
                water = reference.grid
                states = np.column_stack((np.full(water.size, tau), water))
                budgets = np.full(water.size, budget_index, dtype=np.int64)
                first_values = model_values(first, day, states, budgets)
                second_values = model_values(second, day, states, budgets)
                error = np.abs(first_values - second_values)
                differences["expected_water"].append(error[:, 0])
                differences["risk"].append(error[:, 1])
                for channel_index, channel in enumerate(("expected_water", "risk")):
                    location = int(np.argmax(error[:, channel_index]))
                    candidate = float(error[location, channel_index])
                    if candidate > float(maxima[channel]["abs_error"]):
                        maxima[channel] = {
                            "abs_error": candidate,
                            "day": day,
                            "local_time": float(tau),
                            "budget": float(budget_value),
                            "water": float(water[location]),
                        }
    return {
        "expected_water": {
            "p95_abs_difference": float(np.quantile(np.concatenate(differences["expected_water"]), 0.95)),
            "max_abs_difference": float(maxima["expected_water"]["abs_error"]),
            "maximum_location": maxima["expected_water"],
        },
        "risk": {
            "p95_abs_difference": float(np.quantile(np.concatenate(differences["risk"]), 0.95)),
            "max_abs_difference": float(maxima["risk"]["abs_error"]),
            "maximum_location": maxima["risk"],
        },
        "audit_grid": {
            "day_count": DAYS,
            "tau_values": list(TAU_AUDIT),
            "budget_values": list(BUDGETS),
            "water_points_per_slice": int(reference.grid.size),
        },
        "interpretation": "diagnostic only; neither seed is selected",
    }


def run_audit(matrix_root: Path, audit_dir: Path) -> dict[str, object]:
    started = time.time()
    branches = []
    exports: dict[tuple[str, float, int], Path] = {}
    for policy in POLICIES:
        for intensity in INTENSITIES:
            for seed in SEEDS:
                branch_dir = matrix_root / f"{policy}_lambda_{intensity:g}_seed_{seed}" / "branch"
                manifest_path = branch_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if manifest["fallback"] is not None or manifest["stderr"] != "":
                    raise RuntimeError(f"branch fallback/stderr metadata invalid: {branch_dir}")
                execution = json.loads((branch_dir.parent / "execution_status.json").read_text())
                if execution["exit_code"] != 0 or not execution["stderr_empty"]:
                    raise RuntimeError(f"execution status invalid: {branch_dir}")
                export_path = branch_dir / manifest["export"]
                row, artifacts = branch_audit(export_path, branch_dir)
                key = (policy, float(intensity), int(seed))
                exports[key] = export_path
                residual_path = audit_dir / f"residuals_{policy}_lambda_{intensity:g}_seed_{seed}.npz"
                trace_path = audit_dir / f"interfaces_{policy}_lambda_{intensity:g}_seed_{seed}.npz"
                np.savez_compressed(residual_path, **artifacts["residuals"])
                np.savez_compressed(trace_path, **artifacts["traces"])
                row["pide_residual_distribution"] = {
                    "path": residual_path.name,
                    "sha256": _sha256(residual_path),
                    "ordinary_abs_count": int(artifacts["residuals"]["ordinary_abs_values"].size),
                    "lower_layer_abs_count": int(artifacts["residuals"]["lower_layer_abs_values"].size),
                }
                row["interface_trace_file"] = {
                    "path": trace_path.name,
                    "sha256": _sha256(trace_path),
                    "bytes": trace_path.stat().st_size,
                }
                row["manifest_sha256"] = _sha256(manifest_path)
                row["execution_status_sha256"] = _sha256(branch_dir.parent / "execution_status.json")
                row["source_hashes"] = manifest["source_hashes"]
                branches.append(row)
                write_json(audit_dir / f"branch_{row['policy']}_lambda_{row['intensity']:g}_seed_{row['seed']}.json", row)
                print(
                    json.dumps(
                        {"event": "branch_audited", "policy": row["policy"],
                         "intensity": row["intensity"], "seed": row["seed"],
                         "initial_errors": row["initial"]["absolute_errors"],
                         "risk_p95": row["model_values"]["risk"]["p95_abs_error"],
                         "water_p95": row["model_values"]["expected_water"]["p95_abs_error"],
                         "pide_p95": row["pide_residual"]["ordinary_p95_abs"],
                         "trace_max": row["trace_interpolation"]["max_abs_error_all_channels"],
                         "elapsed_seconds": time.time() - started},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    gates = [_gate_branch(row) for row in branches]
    paired = []
    for policy in POLICIES:
        for intensity in INTENSITIES:
            first = next(row for row in branches if row["policy"] == policy and row["intensity"] == intensity and row["seed"] == SEEDS[0])
            second = next(row for row in branches if row["policy"] == policy and row["intensity"] == intensity and row["seed"] == SEEDS[1])
            reference = get_reference(policy, intensity)
            full_difference = paired_prediction_difference(
                exports[(policy, float(intensity), int(SEEDS[0]))],
                exports[(policy, float(intensity), int(SEEDS[1]))],
                reference,
            )
            paired.append(
                {
                    "policy": policy,
                    "intensity": intensity,
                    "initial_expected_water_abs_difference": abs(first["initial"]["model_expected_water"] - second["initial"]["model_expected_water"]),
                    "initial_risk_abs_difference": abs(first["initial"]["model_risk"] - second["initial"]["model_risk"]),
                    "max_daily_risk_p95_difference": abs(first["model_values"]["risk"]["p95_abs_error"] - second["model_values"]["risk"]["p95_abs_error"]),
                    "full_daily_prediction_difference": full_difference,
                }
            )
    result = {
        "format": "fixed-policy-pide-daily-audit-v5",
        "matrix_root": str(matrix_root),
        "reference": {"generator": "central drift finite chain", "h": GRID_H, "upper": UPPER,
                      "policy_seeds_not_used_in_reference": True},
        "branches": branches,
        "branch_gates": gates,
        "paired_seed_diagnostics": paired,
        "overall_pass": bool(len(branches) == 12 and all(gate["pass"] for gate in gates)),
        "fallback": None,
        "stderr": "",
        "elapsed_seconds": time.time() - started,
    }
    write_json(audit_dir / "audit.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.audit_dir.exists():
        raise FileExistsError(f"audit root exists; no overwrite/retry: {args.audit_dir}")
    args.audit_dir.mkdir(parents=True)
    result = run_audit(args.matrix_root, args.audit_dir)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
