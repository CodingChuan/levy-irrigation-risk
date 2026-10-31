"""Backward time-slab scalar risk PIDE PINN for v2.6.3."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


LOWER = 0.35
UPPER = 1.55
HORIZON = 20.0
SIGMA = 0.06
DRIFT_INTERCEPT = 0.04
DRIFT_SLOPE = 0.08
MARKS = (-0.6, 0.4)
MARK_WEIGHTS = (0.7, 0.3)
JUMP_SCALE = 0.30
INTENSITIES = (0.0, 0.12)
SEEDS = (20260921, 20260922)
COLLOCATION_SEED = 262063
WIDTH = 64
HIDDEN_LAYERS = 4
STEPS_PER_SLAB = 1500
LEARNING_RATE = 1.0e-3
CHECKPOINT_STEPS = (0, 500, 1000, 1500)
SLAB_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0)
SLAB_ORDER = (3, 2, 1, 0)
SLAB_COUNT = 4
SAFE_REMAINING = 1.0e-6
CORNER_LAYER_NAME = "fixed_drift_matched_corner_layer"
CORNER_FORMULA_ID = "drift_matched_erfc_base_homogeneous_envelope_v1"
SLAB_ARCHITECTURE = "four_backward_time_slabs"
ORDINARY_POOL = 8192
LOWER_POOL = 2048
INTERIOR_BATCH = 1024
ORDINARY_BATCH = 768
LOWER_BATCH = 256
BOUNDARY_POOL = 2048
BOUNDARY_BATCH = 256
TERMINAL_POOL = 2048
TERMINAL_BATCH = 256
LOWER_OFFSETS = (0.0025, 0.005, 0.01, 0.02, 0.04)


def device_mps() -> torch.device:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; CPU fallback is forbidden")
    return torch.device("mps")


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def slab_bounds(slab_index: int) -> tuple[float, float]:
    index = int(slab_index)
    if index < 0 or index >= SLAB_COUNT:
        raise IndexError(index)
    return float(SLAB_EDGES[index]), float(SLAB_EDGES[index + 1])


class RiskPINN(nn.Module):
    """One independent raw scalar correction network for one slab."""

    def __init__(self, slab_index: int = 0, width: int = WIDTH, hidden_layers: int = HIDDEN_LAYERS):
        super().__init__()
        self.slab_index = int(slab_index)
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def features(self, states: torch.Tensor) -> torch.Tensor:
        a, b = slab_bounds(self.slab_index)
        local_time = (states[:, 0:1] - a) / (b - a)
        moisture = 2.0 * (states[:, 1:2] - LOWER) / (UPPER - LOWER) - 1.0
        return torch.cat((local_time, moisture), dim=1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(states))


def _corner_base(states: torch.Tensor) -> torch.Tensor:
    remaining = torch.clamp(HORIZON - states[:, 0:1], min=0.0)
    x = torch.clamp((states[:, 1:2] - LOWER) / (UPPER - LOWER), min=0.0, max=1.0)
    safe_remaining = torch.clamp(remaining, min=SAFE_REMAINING)
    distance = x * (UPPER - LOWER)
    denominator = SIGMA * torch.sqrt(2.0 * safe_remaining)
    mu = DRIFT_INTERCEPT - DRIFT_SLOPE * LOWER
    def hitting(d):
        return 0.5 * torch.special.erfc((d + mu * safe_remaining) / denominator) + 0.5 * torch.exp(-2.0 * mu * d / SIGMA**2) * torch.special.erfc((d - mu * safe_remaining) / denominator)
    upper = hitting(torch.full_like(distance, UPPER - LOWER))
    base_open = (hitting(distance) - upper) / (1.0 - upper)
    terminal_base = torch.where(x <= 0.0, torch.ones_like(x), torch.zeros_like(x))
    return torch.where(remaining > 0.0, base_open, terminal_base)


class PiecewiseRiskPINN(nn.Module):
    """Four independently parameterized backward slabs with frozen interfaces."""

    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList([RiskPINN(index) for index in range(SLAB_COUNT)])

    def raw_network(self, slab_index: int) -> RiskPINN:
        return self.nets[int(slab_index)]

    def forward_slab(self, states: torch.Tensor, slab_index: int) -> torch.Tensor:
        index=int(slab_index)
        a,b=slab_bounds(index)
        raw=self.nets[index](states)
        x=torch.clamp((states[:,1:2]-LOWER)/(UPPER-LOWER),min=0.,max=1.)
        s=torch.clamp((b-states[:,0:1])/(b-a),min=0.)
        if index==SLAB_COUNT-1:base=_corner_base(states)
        else:
            interface_states=torch.cat((torch.full_like(states[:,0:1],b),states[:,1:2]),dim=1)
            base=self.forward_slab(interface_states,index+1)
        mass=raw.square()
        F=s*(1.-x)*mass[:,0:1]
        S=s*torch.clamp(1.-base,min=0.)*mass[:,1:2]
        bounded=(base+F)/(1.+F+S)
        return torch.where(s>0.,bounded,base)

    def dispatch_slab_index(self, times: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(times, dtype=torch.long)
        for index in range(1, SLAB_COUNT):
            result = torch.where(times >= SLAB_EDGES[index], torch.full_like(result, index), result)
        return result

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        times = states[:, 0]
        output = torch.zeros((states.shape[0], 1), dtype=states.dtype, device=states.device)
        for index in range(SLAB_COUNT):
            lower_edge = SLAB_EDGES[index]
            upper_edge = SLAB_EDGES[index + 1]
            mask = (times >= lower_edge) & ((times <= upper_edge) if index == SLAB_COUNT - 1 else (times < upper_edge))
            output = torch.where(mask[:, None], self.forward_slab(states, index), output)
        return output

    def complete_identity(self) -> dict[str, object]:
        return {
            "architecture": SLAB_ARCHITECTURE,
            "slab_edges": list(SLAB_EDGES),
            "slab_order": list(SLAB_ORDER),
            "steps_per_slab": STEPS_PER_SLAB,
            "width": WIDTH,
            "hidden_layers": HIDDEN_LAYERS,
            "raw_output": "two_squared_mass_boundary_adapted_probability",
            "corner_layer_name": CORNER_LAYER_NAME,
            "corner_formula_id": CORNER_FORMULA_ID,
            "safe_remaining": SAFE_REMAINING,
            "collocation_seed": COLLOCATION_SEED,
        }


def drift(moisture: torch.Tensor) -> torch.Tensor:
    return DRIFT_INTERCEPT - DRIFT_SLOPE * moisture


def jump_locations(moisture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return moisture * math.exp(JUMP_SCALE * MARKS[0]), moisture * math.exp(JUMP_SCALE * MARKS[1])


def _evaluate(model: nn.Module, states: torch.Tensor, slab_index: int | None = None) -> torch.Tensor:
    if slab_index is not None and hasattr(model, "forward_slab"):
        return model.forward_slab(states, int(slab_index))
    return model(states)


def value_with_exterior(model: nn.Module, states: torch.Tensor, slab_index: int | None = None) -> torch.Tensor:
    raw = _evaluate(model, states, slab_index)
    moisture = states[:, 1:2]
    return torch.where(moisture < LOWER, torch.ones_like(raw), torch.where(moisture > UPPER, torch.zeros_like(raw), raw))


def _derivatives(model: nn.Module, states: torch.Tensor, slab_index: int | None = None):
    states = states.requires_grad_(True)
    values = _evaluate(model, states, slab_index)
    gradient = torch.autograd.grad(values.sum(), states, create_graph=True)[0]
    second_moisture = torch.autograd.grad(gradient[:, 1:2].sum(), states, create_graph=True)[0][:, 1:2]
    return values, gradient[:, 0:1], gradient[:, 1:2], second_moisture


def pide_residual(model: nn.Module, states: torch.Tensor, intensity: float, slab_index: int | None = None) -> torch.Tensor:
    values, time_grad, moisture_grad, moisture_second = _derivatives(model, states, slab_index)
    moisture = states[:, 1:2]
    lower_jump, upper_jump = jump_locations(moisture)
    jump_states0 = torch.cat((states[:, 0:1], lower_jump), dim=1)
    jump_states1 = torch.cat((states[:, 0:1], upper_jump), dim=1)
    jumps = MARK_WEIGHTS[0] * value_with_exterior(model, jump_states0, slab_index) + MARK_WEIGHTS[1] * value_with_exterior(model, jump_states1, slab_index) - values
    return time_grad + drift(moisture) * moisture_grad + 0.5 * SIGMA**2 * moisture_second + float(intensity) * jumps


def boundary_residual(model: nn.Module, states: torch.Tensor, slab_index: int | None = None) -> torch.Tensor:
    expected = torch.where(states[:, 1:2] <= LOWER, torch.ones_like(states[:, 1:2]), torch.zeros_like(states[:, 1:2]))
    return _evaluate(model, states, slab_index) - expected


def terminal_residual(model: nn.Module, states: torch.Tensor, slab_index: int) -> torch.Tensor:
    return _evaluate(model, states, slab_index)


def interface_residual(model: PiecewiseRiskPINN, states: torch.Tensor, slab_index: int) -> torch.Tensor:
    index = int(slab_index)
    if index >= SLAB_COUNT - 1:
        return terminal_residual(model, states, index)
    _, b = slab_bounds(index)
    interface_states = torch.cat((torch.full_like(states[:, 0:1], b), states[:, 1:2]), dim=1)
    return model.forward_slab(interface_states, index) - model.forward_slab(interface_states, index + 1)


def build_pools(seed: int = COLLOCATION_SEED, slab_index: int | None = None, *, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + (7919 * int(slab_index) if slab_index is not None else 0))
    time_a, time_b = (0.0, HORIZON) if slab_index is None else slab_bounds(slab_index)
    ordinary_t = time_a + torch.rand((ORDINARY_POOL, 1), generator=generator) * (time_b - time_a)
    ordinary_w = LOWER + torch.rand((ORDINARY_POOL, 1), generator=generator) * (UPPER - LOWER)
    ordinary = torch.cat((ordinary_t, ordinary_w), dim=1)
    offsets = torch.tensor(LOWER_OFFSETS, dtype=torch.float64)
    choices = torch.randint(0, len(LOWER_OFFSETS), (LOWER_POOL,), generator=generator)
    lower_t = time_a + torch.rand((LOWER_POOL, 1), generator=generator) * (time_b - time_a)
    lower_w = LOWER + offsets[choices].reshape(-1, 1).float()
    lower = torch.cat((lower_t, lower_w), dim=1)
    boundary_t = time_a + torch.rand((BOUNDARY_POOL // 2, 1), generator=generator) * max(time_b - time_a - 1.0e-3, 1.0e-3)
    boundary = torch.cat((torch.cat((boundary_t, torch.full_like(boundary_t, LOWER)), dim=1), torch.cat((boundary_t, torch.full_like(boundary_t, UPPER)), dim=1)), dim=0)
    terminal_t = torch.full((TERMINAL_POOL, 1), time_b)
    terminal_w = LOWER + 1.0e-3 + torch.rand((TERMINAL_POOL, 1), generator=generator) * (UPPER - LOWER - 2.0e-3)
    terminal = torch.cat((terminal_t, terminal_w), dim=1)
    return {name: value.to(dtype=dtype) for name, value in {"ordinary": ordinary, "lower_layer": lower, "boundary": boundary, "terminal": terminal}.items()}


def fixed_batch_indices(seed: int = COLLOCATION_SEED, step: int = 0, slab_index: int = 0):
    rng = np.random.default_rng(int(seed) + 1000003 * int(step) + 7919 * int(slab_index))
    return (rng.choice(ORDINARY_POOL, size=ORDINARY_BATCH, replace=False), rng.choice(LOWER_POOL, size=LOWER_BATCH, replace=False), rng.choice(BOUNDARY_POOL, size=BOUNDARY_BATCH, replace=False), rng.choice(TERMINAL_POOL, size=TERMINAL_BATCH, replace=False))


def freeze_all_except(model: PiecewiseRiskPINN, active_slab: int) -> None:
    for index, network in enumerate(model.nets):
        for parameter in network.parameters():
            parameter.requires_grad_(index == int(active_slab))


def jsonable_constants() -> dict[str, object]:
    return {"lower": LOWER, "upper": UPPER, "horizon": HORIZON, "sigma": SIGMA, "drift_intercept": DRIFT_INTERCEPT, "drift_slope": DRIFT_SLOPE, "marks": list(MARKS), "mark_weights": list(MARK_WEIGHTS), "jump_scale": JUMP_SCALE, "intensities": list(INTENSITIES), "seeds": list(SEEDS), "collocation_seed": COLLOCATION_SEED, "width": WIDTH, "hidden_layers": HIDDEN_LAYERS, "steps_per_slab": STEPS_PER_SLAB, "checkpoint_steps": list(CHECKPOINT_STEPS), "slab_edges": list(SLAB_EDGES), "slab_order": list(SLAB_ORDER), "ordinary_pool": ORDINARY_POOL, "lower_pool": LOWER_POOL, "interior_batch": INTERIOR_BATCH, "ordinary_batch": ORDINARY_BATCH, "lower_batch": LOWER_BATCH, "boundary_pool": BOUNDARY_POOL, "boundary_batch": BOUNDARY_BATCH, "terminal_pool": TERMINAL_POOL, "terminal_batch": TERMINAL_BATCH, "lower_offsets": list(LOWER_OFFSETS), "architecture": SLAB_ARCHITECTURE, "corner_layer_name": CORNER_LAYER_NAME, "corner_formula_id": CORNER_FORMULA_ID, "safe_remaining": SAFE_REMAINING}


def complete_identity() -> dict[str, object]:
    return PiecewiseRiskPINN().complete_identity()


def parameter_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
