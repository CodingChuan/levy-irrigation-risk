"""Independent CPU-float64 audit for the v2.6.3 common-collocation model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import expm_multiply

import model_reference
from pinn_fixed_policy import (
    CORNER_FORMULA_ID,
    CORNER_LAYER_NAME,
    COLLOCATION_SEED,
    HORIZON,
    INTENSITIES,
    LOWER,
    LOWER_OFFSETS,
    SAFE_REMAINING,
    SEEDS,
    SLAB_EDGES,
    SLAB_COUNT,
    UPPER,
    PiecewiseRiskPINN,
    pide_residual,
    sha256_file,
    write_json,
)

PIDE_MEDIAN_GATE = 0.005
PIDE_P95_GATE = 0.02
BOUNDARY_GATE = 0.01
TERMINAL_GATE = 0.01
FD_MAE_GATE = 0.02
FD_P95_GATE = 0.04
LOWER_LAYER_GATE = 0.05
INITIAL_GATE = 0.02
OUTPUT_MIN = -0.02
OUTPUT_MAX = 1.02
PAIRED_SEED_GATE = 0.025
INTERFACE_GATE = 1.0e-10


def expected_identity() -> dict[str, object]:
    return {
        "architecture": "four_backward_time_slabs",
        "slab_edges": list(SLAB_EDGES),
        "slab_order": [3, 2, 1, 0],
        "steps_per_slab": 1500,
        "width": 64,
        "hidden_layers": 4,
        "raw_output": "two_squared_mass_boundary_adapted_probability",
        "corner_layer_name": CORNER_LAYER_NAME,
        "corner_formula_id": CORNER_FORMULA_ID,
        "safe_remaining": SAFE_REMAINING,
        "collocation_seed": COLLOCATION_SEED,
    }


def load_export(path: Path) -> tuple[PiecewiseRiskPINN, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("identity") != expected_identity():
        raise RuntimeError(f"time-slab identity mismatch: {path}")
    model = PiecewiseRiskPINN().double()
    model.load_state_dict(payload["model"])
    model.eval()
    if any(parameter.dtype != torch.float64 or parameter.device.type != "cpu" for parameter in model.parameters()):
        raise RuntimeError(f"export is not CPU float64: {path}")
    return model, payload


def load_collocation_manifest(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("collocation_seed") != COLLOCATION_SEED:
        raise RuntimeError(f"common collocation seed mismatch: {directory}")
    pool_hashes: dict[str, str] = {}
    batch_hashes: dict[str, str] = {}
    for slab in range(SLAB_COUNT):
        record = manifest.get("slabs", {}).get(str(slab), {})
        pool_path = directory / f"collocation_slab_{slab}.npz"
        batch_path = directory / f"slab_{slab}" / "batch_indices.npz"
        pool_actual = sha256_file(pool_path)
        batch_actual = sha256_file(batch_path)
        if record.get("pool_sha256") != pool_actual or record.get("batch_indices_sha256") != batch_actual:
            raise RuntimeError(f"collocation hash mismatch: {directory}, slab={slab}")
        pool_hashes[str(slab)] = pool_actual
        batch_hashes[str(slab)] = batch_actual
    return {
        "collocation_seed": COLLOCATION_SEED,
        "pool_sha256": pool_hashes,
        "batch_indices_sha256": batch_hashes,
        "initial_parameter_sha256": manifest.get("initial_parameter_sha256"),
    }


def finite_reference(intensity: float, h: float = 0.0025):
    grid, q, _, _ = model_reference.build_generator(UPPER, h, float(intensity))
    terminal = np.zeros(grid.size, dtype=np.float64)
    terminal[0] = 1.0
    return grid, q, terminal


def reference_at(times: np.ndarray, locations: np.ndarray, intensity: float, h: float = 0.0025) -> np.ndarray:
    grid, q, terminal = finite_reference(intensity, h)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    locations = np.asarray(locations, dtype=np.float64).reshape(-1)
    out = np.empty(times.size, dtype=np.float64)
    unique_times, inverse = np.unique(times, return_inverse=True)
    for index, time_value in enumerate(unique_times):
        values = np.asarray(expm_multiply((HORIZON - float(time_value)) * q, terminal), dtype=np.float64)
        mask = inverse == index
        out[mask] = np.interp(locations[mask], grid, values)
        out[mask & (locations <= LOWER)] = 1.0
        out[mask & (locations >= UPPER)] = 0.0
    return out


def chunks(values: np.ndarray, size: int):
    return [values[start:start + size] for start in range(0, len(values), size)]


def model_values(model: PiecewiseRiskPINN, states: np.ndarray, chunk: int = 1024) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.no_grad():
        for part in chunks(np.asarray(states, dtype=np.float64), chunk):
            values.append(model(torch.from_numpy(part)).cpu().numpy().reshape(-1))
    return np.concatenate(values)


def model_pide(model: PiecewiseRiskPINN, states: np.ndarray, intensity: float, chunk: int = 256) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.enable_grad():
        for part in chunks(np.asarray(states, dtype=np.float64), chunk):
            tensor = torch.from_numpy(part).requires_grad_(True)
            values.append(pide_residual(model, tensor, intensity).detach().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def fresh_points(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 7000001)
    ordinary = np.column_stack((rng.uniform(0.0, HORIZON, 4096), rng.uniform(LOWER, UPPER, 4096)))
    offsets = np.asarray(LOWER_OFFSETS, dtype=np.float64)
    lower = np.column_stack((rng.uniform(0.0, HORIZON, 1024), LOWER + offsets[rng.integers(0, len(offsets), 1024)]))
    boundary_times = rng.uniform(0.0, HORIZON - 1.0e-3, 512)
    return {
        "ordinary": ordinary,
        "lower_layer": lower,
        "lower_boundary": np.column_stack((boundary_times, np.full(512, LOWER))),
        "upper_boundary": np.column_stack((boundary_times, np.full(512, UPPER))),
        "terminal": np.column_stack((np.full(512, HORIZON), rng.uniform(LOWER + 1.0e-3, UPPER - 1.0e-3, 512))),
    }


def fd_points():
    grid = np.linspace(LOWER, UPPER, int(round((UPPER - LOWER) / 0.0025)) + 1, dtype=np.float64)[1:-1]
    times = np.asarray((0.0, 5.0, 10.0, 15.0, 19.5))
    return np.repeat(times, grid.size), np.tile(grid, times.size)


def lower_layer_points():
    times = np.asarray((0.0, 5.0, 10.0, 15.0, 19.5))
    offsets = np.asarray(LOWER_OFFSETS)
    return np.repeat(times, offsets.size), np.tile(LOWER + offsets, times.size)


def interface_metric(model: PiecewiseRiskPINN, seed: int) -> dict[str, float | bool]:
    rng = np.random.default_rng(int(seed) + 909090)
    moisture = rng.uniform(LOWER + 1.0e-3, UPPER - 1.0e-3, 512)
    maxima: dict[str, float] = {}
    with torch.no_grad():
        for slab in range(SLAB_COUNT - 1):
            _, edge = float(SLAB_EDGES[slab]), float(SLAB_EDGES[slab + 1])
            states = torch.from_numpy(np.column_stack((np.full(512, edge), moisture)))
            left = model.forward_slab(states, slab)
            right = model.forward_slab(states, slab + 1)
            maxima[str(slab)] = float(torch.max(torch.abs(left - right)))
    maximum = max(maxima.values())
    return {"per_interface_max": maxima, "max_abs_difference": maximum, "pass": maximum <= INTERFACE_GATE}


def branch_audit(model: PiecewiseRiskPINN, intensity: float, seed: int) -> dict[str, object]:
    points = fresh_points(seed)
    residual = model_pide(model, np.vstack((points["ordinary"], points["lower_layer"])), intensity)
    ordinary_residual, lower_residual = residual[:4096], residual[4096:]
    lower_values = model_values(model, points["lower_boundary"])
    upper_values = model_values(model, points["upper_boundary"])
    terminal_values = model_values(model, points["terminal"])
    fd_t, fd_w = fd_points()
    fd_values = model_values(model, np.column_stack((fd_t, fd_w)))
    fd_reference = reference_at(fd_t, fd_w, intensity)
    fd_error = np.abs(fd_values - fd_reference)
    low_t, low_w = lower_layer_points()
    low_values = model_values(model, np.column_stack((low_t, low_w)))
    low_reference = reference_at(low_t, low_w, intensity)
    low_error = np.abs(low_values - low_reference)
    initial = float(model_values(model, np.asarray([[0.0, 0.60]]))[0])
    initial_reference = float(reference_at(np.asarray([0.0]), np.asarray([0.60]), intensity)[0])
    signed_by_time = {}
    for time_value in (0.0, 5.0, 10.0, 15.0, 19.5):
        mask = fd_t == time_value
        signed_by_time[str(time_value)] = {"mean_signed_error": float(np.mean(fd_values[mask] - fd_reference[mask])), "mean_abs_error": float(np.mean(fd_error[mask]))}
    all_values = np.concatenate((model_values(model, np.vstack((points["ordinary"], points["lower_layer"]))), lower_values, upper_values, terminal_values, fd_values, low_values, np.asarray([initial])))
    return {
        "intensity": float(intensity),
        "seed": int(seed),
        "pide": {"ordinary_count": 4096, "lower_layer_count": 1024, "median": float(np.median(np.abs(ordinary_residual))), "p95": float(np.quantile(np.abs(ordinary_residual), 0.95)), "max": float(np.max(np.abs(ordinary_residual))), "lower_layer_max": float(np.max(np.abs(lower_residual))), "lower_layer_p95": float(np.quantile(np.abs(lower_residual), 0.95))},
        "boundary": {"max_abs_error": float(max(np.max(np.abs(lower_values - 1.0)), np.max(np.abs(upper_values))))},
        "terminal": {"max_abs_error": float(np.max(np.abs(terminal_values)))},
        "fd": {"count": int(fd_error.size), "mae": float(np.mean(fd_error)), "p95": float(np.quantile(fd_error, 0.95)), "max": float(np.max(fd_error))},
        "lower_layer_value": {"max_abs_error": float(np.max(low_error)), "mae": float(np.mean(low_error))},
        "initial": {"model": initial, "reference": initial_reference, "abs_error": abs(initial - initial_reference)},
        "signed_error_by_time": signed_by_time,
        "output_range": {"min": float(np.min(all_values)), "max": float(np.max(all_values))},
        "finite": bool(np.isfinite(all_values).all()),
    }


def run_audit(branch_dirs: dict[tuple[float, int], Path], audit_dir: Path) -> dict[str, object]:
    started = time.time()
    loaded = {}
    collocation = {}
    for key, directory in branch_dirs.items():
        loaded[key] = load_export(directory / "model_cpu_float64.pt")
        collocation[key] = load_collocation_manifest(directory)
    branches = []
    for (intensity, seed), (model, _) in loaded.items():
        row = branch_audit(model, intensity, seed)
        row["export_sha256"] = sha256_file(branch_dirs[(intensity, seed)] / "model_cpu_float64.pt")
        row["interface"] = interface_metric(model, seed)
        row["collocation"] = collocation[(intensity, seed)]
        branches.append(row)
    for intensity in INTENSITIES:
        first = loaded[(intensity, SEEDS[0])][0]
        second = loaded[(intensity, SEEDS[1])][0]
        ft, fw = fd_points(); lt, lw = lower_layer_points()
        states = np.vstack((np.column_stack((ft, fw)), np.column_stack((lt, lw))))
        difference = float(np.max(np.abs(model_values(first, states) - model_values(second, states))))
        for row in branches:
            if float(row["intensity"]) == float(intensity): row["paired_prediction_max_difference"] = difference
    gates = []
    for row in branches:
        checks = {
            "finite_cpu_float64": bool(row["finite"]),
            "boundary_max": row["boundary"]["max_abs_error"] <= BOUNDARY_GATE,
            "terminal_max": row["terminal"]["max_abs_error"] <= TERMINAL_GATE,
            "pide_median": row["pide"]["median"] <= PIDE_MEDIAN_GATE,
            "pide_p95": row["pide"]["p95"] <= PIDE_P95_GATE,
            "fd_mae": row["fd"]["mae"] <= FD_MAE_GATE,
            "fd_p95": row["fd"]["p95"] <= FD_P95_GATE,
            "lower_layer_max": row["lower_layer_value"]["max_abs_error"] <= LOWER_LAYER_GATE,
            "initial_risk": row["initial"]["abs_error"] <= INITIAL_GATE,
            "output_range": row["output_range"]["min"] >= OUTPUT_MIN and row["output_range"]["max"] <= OUTPUT_MAX,
            "paired_seed": row["paired_prediction_max_difference"] <= PAIRED_SEED_GATE,
            "interface": row["interface"]["pass"],
        }
        gates.append({"branch": f"lambda_{row['intensity']:g}_seed_{row['seed']}", "checks": checks, "pass": bool(all(checks.values()))})
    result = {"format": "v2.6.2-time-slab-audit", "identity": expected_identity(), "reference": {"generator": "scipy.sparse.linalg.expm_multiply", "h": 0.0025, "upper": UPPER}, "branches": branches, "gates": gates, "overall_pass": bool(len(branches) == 4 and all(row["pass"] for row in gates)), "fallback": None, "stderr": "", "elapsed_seconds": time.time() - started}
    write_json(audit_dir / "audit.json", result)
    lines = ["# v2.6.2 time-slab fixed-policy audit", "", f"overall_pass: {result['overall_pass']}", "", "```json", json.dumps(result, indent=2, sort_keys=True), "```"]
    (audit_dir / "audit.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--branch", action="append", nargs=3, required=True)
    args = parser.parse_args()
    branch_dirs = {(float(i), int(s)): Path(d) for i, s, d in args.branch}
    result = run_audit(branch_dirs, args.audit_dir)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
