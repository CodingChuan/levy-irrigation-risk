"""Daily-slab physics-informed value evaluator for frozen policies.

Daily continuation traces are materialized once after each backward slab. This
keeps evaluation linear in the horizon while preserving the same analytic
policy mixture at every daily interface.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from operators import BUDGETS, DIFFUSION, WD, action_probability

UPPER = 1.55
DAYS = 20
SIGMA = 0.06
DRIFT_INTERCEPT = 0.04
DRIFT_SLOPE = 0.08
MARKS = (-0.6, 0.4)
MARK_WEIGHTS = (0.7, 0.3)
JUMP_SCALE = 0.30
WIDTH = 64
HIDDEN_LAYERS = 4
TRACE_STEP = 0.000625
TRACE_FIRST = WD + TRACE_STEP
# Store only open-domain nodes: both absorbing sentinels are evaluated exactly
# and must not enter the one-sided interior interpolation polynomial.
TRACE_INTERVALS = int(round((UPPER - TRACE_FIRST) / TRACE_STEP)) - 1
TRACE_POINTS = TRACE_INTERVALS + 1
TRACE_AUDIT_SEED = 801247
TRACE_AUDIT_POINTS = 4096
TRACE_MAX_ERROR = 1.0e-4
BUFFER_KINKS = (WD + DIFFUSION, WD + 2.0 * DIFFUSION)
BUFFER_KINK_INDICES = tuple(
    int(round((kink - TRACE_FIRST) / TRACE_STEP)) for kink in BUFFER_KINKS
)


class PolicySlabNet(nn.Module):
    """One day, one frozen policy; input `(local_time, water, budget index)`."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        dims = [7] + [width] * HIDDEN_LAYERS + [2]
        layers = []
        for i in range(len(dims) - 2):
            layers.extend((nn.Linear(dims[i], dims[i + 1]), nn.Tanh()))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def features(states: torch.Tensor, budget_index: torch.Tensor) -> torch.Tensor:
        tau = states[:, 0:1]
        moisture = 2.0 * (states[:, 1:2] - WD) / (UPPER - WD) - 1.0
        onehot = torch.nn.functional.one_hot(
            budget_index.to(torch.long), num_classes=len(BUDGETS)
        ).to(states.dtype)
        return torch.cat((tau, moisture, onehot), dim=1)

    def raw(self, states: torch.Tensor, budget_index: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(states, budget_index))


class FixedPolicyPINN(nn.Module):
    def __init__(self, policy: str, intensity: float):
        super().__init__()
        if policy not in {"zero", "schedule", "buffer"}:
            raise ValueError(policy)
        self.policy = str(policy)
        self.intensity = float(intensity)
        self.nets = nn.ModuleList([PolicySlabNet() for _ in range(DAYS)])
        # Generate the same float32 interior coordinates as v2, then omit its
        # final safe-sentinel node; this makes checkpoint trace migration exact.
        trace_grid = torch.linspace(
            TRACE_FIRST, UPPER, TRACE_POINTS + 1, dtype=torch.float32
        )[:-1]
        self.register_buffer("trace_grid", trace_grid)
        self.register_buffer("trace_ready", torch.zeros(DAYS - 1, dtype=torch.bool))
        for day in range(DAYS - 1):
            self.register_buffer(
                f"terminal_trace_{day}",
                torch.zeros((len(BUDGETS), TRACE_POINTS, 2), dtype=torch.float32),
            )

    @staticmethod
    def _trace_name(day: int) -> str:
        return f"terminal_trace_{int(day)}"

    def set_terminal_trace(self, day: int, values: torch.Tensor) -> None:
        day = int(day)
        if day < 0 or day >= DAYS - 1:
            raise IndexError(day)
        target = getattr(self, self._trace_name(day))
        if tuple(values.shape) != tuple(target.shape):
            raise ValueError(f"trace shape {tuple(values.shape)} != {tuple(target.shape)}")
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"nonfinite trace for day {day}")
        target.copy_(values.to(device=target.device, dtype=target.dtype))
        self.trace_ready[day] = True

    def trace_slopes(self, day: int) -> torch.Tensor:
        trace = getattr(self, self._trace_name(day))
        grid = self.trace_grid
        slope = torch.empty_like(trace)
        slope[:, 0, :] = (trace[:, 1, :] - trace[:, 0, :]) / (grid[1] - grid[0])
        slope[:, -1, :] = (trace[:, -1, :] - trace[:, -2, :]) / (grid[-1] - grid[-2])
        slope[:, 1:-1, :] = (trace[:, 2:, :] - trace[:, :-2, :]) / (
            grid[2:] - grid[:-2]
        )[None, :, None]
        return slope

    def interpolate_trace(
        self, day: int, water: torch.Tensor, budget_index: torch.Tensor
    ) -> torch.Tensor:
        day = int(day)
        if day < 0 or day >= DAYS - 1:
            raise IndexError(day)
        if not bool(self.trace_ready[day]):
            raise RuntimeError(f"daily terminal trace {day} has not been materialized")
        trace = getattr(self, self._trace_name(day))
        slope = self.trace_slopes(day)
        grid = self.trace_grid
        raw_water = water.reshape(-1)
        lower_sentinel = raw_water <= WD
        upper_sentinel = raw_water >= UPPER
        w = raw_water.clamp(min=WD, max=UPPER)
        budget_index = budget_index.to(device=w.device, dtype=torch.long).reshape(-1)
        position = (w - TRACE_FIRST) / TRACE_STEP
        position = position.clamp(min=-1.0, max=float(TRACE_POINTS - 1))
        index = torch.floor(position).to(torch.long).clamp(min=0, max=TRACE_POINTS - 2)
        x0 = grid[index]
        x1 = grid[index + 1]
        cell_width = x1 - x0
        fraction = (w - x0) / cell_width
        y0 = trace[budget_index, index]
        y1 = trace[budget_index, index + 1]
        m0 = slope[budget_index, index]
        m1 = slope[budget_index, index + 1]
        if self.policy == "buffer":
            # The buffer action map is continuous but has derivative breaks at
            # its two frozen water thresholds. A single centered slope at a
            # kink averages the left and right derivatives and corrupts the
            # adjacent Hermite cells. Use each cell's own one-sided secant.
            positive_budget = budget_index > 0
            for kink_index in BUFFER_KINK_INDICES:
                left_width = grid[kink_index] - grid[kink_index - 1]
                right_width = grid[kink_index + 1] - grid[kink_index]
                left_secant = (
                    trace[:, kink_index, :] - trace[:, kink_index - 1, :]
                ) / left_width
                right_secant = (
                    trace[:, kink_index + 1, :] - trace[:, kink_index, :]
                ) / right_width
                starts_at_kink = positive_budget & (index == kink_index)
                ends_at_kink = positive_budget & (index + 1 == kink_index)
                m0 = torch.where(
                    starts_at_kink[:, None], right_secant[budget_index], m0
                )
                m1 = torch.where(
                    ends_at_kink[:, None], left_secant[budget_index], m1
                )
        t = fraction[:, None]
        h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        h11 = t**3 - t**2
        values = (
            h00 * y0
            + h10 * cell_width[:, None] * m0
            + h01 * y1
            + h11 * cell_width[:, None] * m1
        )
        failed_value = torch.zeros_like(values)
        failed_value[:, 1] = 1.0
        values = torch.where(lower_sentinel[:, None], failed_value, values)
        return torch.where(upper_sentinel[:, None], torch.zeros_like(values), values)

    def _corner_base(self, states: torch.Tensor) -> torch.Tensor:
        tau, water = states[:, 0:1], states[:, 1:2]
        remaining = torch.clamp(1.0 - tau, min=0.0)
        x = torch.clamp((water - WD) / (UPPER - WD), min=0.0, max=1.0)
        safe_remaining = torch.clamp(remaining, min=1.0e-6)
        denom = SIGMA * torch.sqrt(2.0 * safe_remaining)
        e = torch.special.erfc(x * (UPPER - WD) / denom)
        e_upper = torch.special.erfc((UPPER - WD) / denom)
        base = (e - e_upper) / (1.0 - e_upper)
        terminal = torch.where(x <= 0.0, torch.ones_like(x), torch.zeros_like(x))
        return torch.where(remaining > 0.0, base, terminal)

    def forward_day(
        self, day: int, states: torch.Tensor, budget_index: torch.Tensor
    ) -> torch.Tensor:
        day = int(day)
        if day < 0 or day >= DAYS:
            if day == DAYS:
                risk = torch.where(
                    states[:, 1:2] <= WD,
                    torch.ones_like(states[:, 1:2]),
                    torch.zeros_like(states[:, 1:2]),
                )
                return torch.cat((torch.zeros_like(risk), risk), dim=1)
            raise IndexError(day)
        tau, water = states[:, 0:1], states[:, 1:2]
        x = torch.clamp((water - WD) / (UPPER - WD), min=0.0, max=1.0)
        raw = self.nets[day].raw(states, budget_index)
        if day == DAYS - 1:
            base_cost = torch.zeros_like(x)
            base_risk = self._corner_base(states)
            envelope = x * (1.0 - x) * torch.clamp(1.0 - tau, min=0.0)
        else:
            terminal = self.interpolate_trace(day, water[:, 0], budget_index)
            base_cost = tau * terminal[:, 0:1]
            base_risk = (1.0 - tau) * (1.0 - x) + tau * terminal[:, 1:2]
            envelope = torch.clamp(1.0 - tau, min=0.0) * x * (1.0 - x)
        output = torch.cat((base_cost, base_risk), dim=1) + envelope * raw
        risk_lower = torch.where(water <= WD, torch.ones_like(water), output[:, 1:2])
        risk = torch.where(water >= UPPER, torch.zeros_like(water), risk_lower)
        cost = torch.where(
            (water <= WD) | (water >= UPPER), torch.zeros_like(water), output[:, 0:1]
        )
        return torch.cat((cost, risk), dim=1)


def apply_policy_at_day_start(
    model: FixedPolicyPINN,
    day: int,
    water: torch.Tensor,
    budget_index: torch.Tensor,
) -> torch.Tensor:
    """Apply one frozen daily action map to post-action continuation values."""
    water = water.reshape(-1)
    budget_index = budget_index.to(device=water.device, dtype=torch.long).reshape(-1)
    time_zero = torch.zeros_like(water)
    no_states = torch.stack((time_zero, water), dim=1)
    no_value = model.forward_day(day, no_states, budget_index)
    probability, amount, successor = action_probability(
        model.policy, day, water, budget_index
    )
    yes_states = torch.stack((time_zero, water + amount), dim=1)
    yes_value = model.forward_day(day, yes_states, successor)
    p = probability[:, None]
    q = amount[:, None]
    cost = (1.0 - p) * no_value[:, 0:1] + p * (q + yes_value[:, 0:1])
    risk = (1.0 - p) * no_value[:, 1:2] + p * yes_value[:, 1:2]
    return torch.cat((cost, risk), dim=1)


def drift(water: torch.Tensor) -> torch.Tensor:
    return DRIFT_INTERCEPT - DRIFT_SLOPE * water


def pide_residual(
    model: FixedPolicyPINN,
    day: int,
    states: torch.Tensor,
    budget_index: torch.Tensor,
) -> torch.Tensor:
    states = states.requires_grad_(True)
    values = model.forward_day(day, states, budget_index)
    water = states[:, 1:2]
    jump_values = torch.zeros_like(values)
    for mark, probability in zip(MARKS, MARK_WEIGHTS):
        mapped = water * math.exp(JUMP_SCALE * mark)
        jump_states = torch.cat((states[:, 0:1], mapped), dim=1)
        jump_values = jump_values + probability * model.forward_day(day, jump_states, budget_index)
    nonlocal_term = jump_values - values
    local_residuals = []
    for channel in range(values.shape[1]):
        grad = torch.autograd.grad(
            values[:, channel].sum(), states, create_graph=True, retain_graph=True
        )[0]
        time_grad = grad[:, 0:1]
        water_grad = grad[:, 1:2]
        second = torch.autograd.grad(
            water_grad.sum(), states, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        generator = drift(water) * water_grad + 0.5 * SIGMA**2 * second
        local_residuals.append(time_grad + generator)
    local = torch.cat(local_residuals, dim=1)
    return local + float(model.intensity) * nonlocal_term
