"""Frozen daily policy interface operators for fixed-policy evaluation v1."""
from __future__ import annotations

import torch

BUDGETS = (0.0, 0.06, 0.14, 0.22, 0.30)
ACTION_CAP = 0.08
WD = 0.35
DIFFUSION = 0.06


def action_probability(policy: str, day: int, water: torch.Tensor,
                       budget_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive-action probability, executed amount, and successor budget index."""
    if policy not in {"zero", "schedule", "buffer"}:
        raise ValueError(f"unknown frozen policy: {policy}")
    device, dtype = water.device, water.dtype
    levels = torch.as_tensor(BUDGETS, device=device, dtype=dtype)
    idx = budget_index.to(device=device, dtype=torch.long).reshape(-1)
    beta = levels[idx]
    q = torch.minimum(torch.full_like(beta, ACTION_CAP), beta)
    if policy == "zero":
        prob = torch.zeros_like(beta)
    elif policy == "schedule":
        prob = torch.full_like(beta, 1.0 if day <= 3 else 0.0)
        prob = torch.where(beta > 0.0, prob, torch.zeros_like(prob))
    else:
        lo = WD + DIFFUSION
        hi = WD + 2.0 * DIFFUSION
        prob = torch.clamp((hi - water.reshape(-1)) / (hi - lo), min=0.0, max=1.0)
        prob = torch.where(beta > 0.0, prob, torch.zeros_like(prob))
    alive = (water.reshape(-1) > WD) & (water.reshape(-1) < 1.55)
    prob = torch.where(alive, prob, torch.zeros_like(prob))
    q = torch.where(prob > 0.0, q, torch.zeros_like(q))
    next_idx = torch.where(q > 0.0, torch.clamp(idx - 1, min=0), idx)
    return prob, q, next_idx


def policy_pre_trace(policy: str, day: int, water: torch.Tensor,
                     budget_index: torch.Tensor, value_no_action: torch.Tensor,
                     value_positive_action: torch.Tensor) -> torch.Tensor:
    """Apply a frozen decision rule to (cost, risk) post-action values.

    The cost channel includes the expected immediate water pulse; the risk
    channel is only the mixture of continuation risks.
    """
    prob, q, _ = action_probability(policy, day, water, budget_index)
    p = prob.reshape(-1, 1)
    qcol = q.reshape(-1, 1)
    no = value_no_action
    yes = value_positive_action
    cost = (1.0 - p) * no[:, 0:1] + p * (qcol + yes[:, 0:1])
    risk = (1.0 - p) * no[:, 1:2] + p * yes[:, 1:2]
    return torch.cat((cost, risk), dim=1)
