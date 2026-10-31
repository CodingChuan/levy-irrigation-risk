"""Sequential backward time-slab training for one fixed-policy branch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from guarded_adam import step as protected_step,flatten

from pinn_fixed_policy import (
    BOUNDARY_BATCH,
    CHECKPOINT_STEPS,
    COLLOCATION_SEED,
    HORIZON,
    INTENSITIES,
    INTERIOR_BATCH,
    LEARNING_RATE,
    LOWER_BATCH,
    ORDINARY_BATCH,
    SAFE_REMAINING,
    SEEDS,
    SLAB_EDGES,
    SLAB_ORDER,
    SLAB_COUNT,
    STEPS_PER_SLAB,
    TERMINAL_BATCH,
    CORNER_FORMULA_ID,
    CORNER_LAYER_NAME,
    PiecewiseRiskPINN,
    boundary_residual,
    build_pools,
    device_mps,
    fixed_batch_indices,
    freeze_all_except,
    interface_residual,
    jsonable_constants,
    pide_residual,
    parameter_sha256,
    seed_everything,
    sha256_file,
    write_json,
)


def identity() -> dict[str, object]:
    return {
        "architecture": "four_backward_time_slabs",
        "slab_edges": list(SLAB_EDGES),
        "slab_order": list(SLAB_ORDER),
        "steps_per_slab": STEPS_PER_SLAB,
        "width": 64,
        "hidden_layers": 4,
        "raw_output": "two_squared_mass_boundary_adapted_probability",
        "corner_layer_name": CORNER_LAYER_NAME,
        "corner_formula_id": CORNER_FORMULA_ID,
        "safe_remaining": SAFE_REMAINING,
        "collocation_seed": COLLOCATION_SEED,
    }


def atomic_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def cpu_state(model: PiecewiseRiskPINN) -> dict[str, torch.Tensor]:
    return {key: value.detach().to(device="cpu").clone() for key, value in model.state_dict().items()}


def save_checkpoint(model: PiecewiseRiskPINN, path: Path, intensity: float, seed: int, slab: int, step: int) -> str:
    atomic_save({
        "format": "v2.6.2-time-slab-checkpoint",
        "intensity": float(intensity),
        "seed": int(seed),
        "slab": int(slab),
        "local_step": int(step),
        "model": cpu_state(model),
        "identity": identity(),
        "dtype": "float32",
    }, path)
    return sha256_file(path)


def export_cpu_float64(model: PiecewiseRiskPINN, output: Path, intensity: float, seed: int, checkpoint_sha256: str) -> str:
    exported = PiecewiseRiskPINN().double()
    exported.load_state_dict(cpu_state(model))
    if any(parameter.dtype != torch.float64 or parameter.device.type != "cpu" for parameter in exported.parameters()):
        raise RuntimeError("complete CPU float64 export failed")
    atomic_save({
        "format": "v2.6.2-time-slab-cpu-float64",
        "intensity": float(intensity),
        "seed": int(seed),
        "model": exported.state_dict(),
        "checkpoint_sha256": checkpoint_sha256,
        "identity": identity(),
        "dtype": "float64",
    }, output)
    return sha256_file(output)


def train_branch(intensity: float, seed: int, run_dir: Path, selected_slab: int) -> dict[str, object]:
    started = time.time()
    device = device_mps()
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    model = PiecewiseRiskPINN().to(device=device, dtype=torch.float32)
    initial_parameter_sha = parameter_sha256(model)
    previous = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else None
    slab_records: dict[str, object] = {} if previous is None else previous["slabs"]
    if selected_slab < 3:
        prior=run_dir/f"complete_after_slab_{selected_slab+1}.pt"
        payload=torch.load(prior,map_location="cpu",weights_only=True)
        if payload["identity"] != identity() or str(selected_slab+1) not in slab_records:raise RuntimeError("invalid same-candidate prefix")
        if sha256_file(prior)!=slab_records[str(selected_slab+1)]["complete_checkpoint_sha256"]:raise RuntimeError("prefix checkpoint hash changed")
        model.load_state_dict(payload["model"])
    elif previous is not None:raise RuntimeError("fresh terminal slab already exists")
    if str(selected_slab) in slab_records:raise RuntimeError("completed slab must not be repeated")
    final_checkpoint_sha = ""

    for slab in [selected_slab]:
        freeze_all_except(model, slab)
        active_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(active_parameters, lr=LEARNING_RATE)
        pools_cpu = build_pools(COLLOCATION_SEED, slab_index=slab, dtype=torch.float32)
        pools = {key: value.to(device) for key, value in pools_cpu.items()}
        pool_path = run_dir / f"collocation_slab_{slab}.npz"
        np.savez_compressed(pool_path, **{key: value.numpy() for key, value in pools_cpu.items()})
        pool_sha = sha256_file(pool_path)
        slab_dir = run_dir / f"slab_{slab}"
        slab_dir.mkdir(parents=True, exist_ok=True)
        histories: list[dict[str, float | int]] = []
        ordinary_trace: list[np.ndarray] = []
        lower_trace: list[np.ndarray] = []
        boundary_trace: list[np.ndarray] = []
        terminal_trace: list[np.ndarray] = []
        checkpoint_hashes: dict[str, str] = {}
        optimizer_hashes = {}
        adam_traces={k:[] for k in ["gradient","delta","moment1","moment2"]}
        def save_optimizer(step):
            path=slab_dir/f"optimizer_{step:04d}.pt"
            payload=optimizer.state_dict()
            def cpu_tree(x):
                if torch.is_tensor(x):return x.detach().cpu().clone()
                if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
                if isinstance(x,list):return [cpu_tree(v) for v in x]
                return x
            atomic_save(cpu_tree(payload),path);optimizer_hashes[str(step)]=sha256_file(path)
        save_optimizer(0)
        checkpoint_hashes["0"] = save_checkpoint(model, slab_dir / "checkpoint_0000.pt", intensity, seed, slab, 0)
        write_json(slab_dir/"slab_progress.json",{"state":"initial","step":0,"checkpoint_sha":checkpoint_hashes["0"],"optimizer_sha":optimizer_hashes["0"]})

        for step in range(1, STEPS_PER_SLAB + 1):
            ordinary_idx, lower_idx, boundary_idx, terminal_idx = fixed_batch_indices(COLLOCATION_SEED, step, slab)
            ordinary_trace.append(ordinary_idx)
            lower_trace.append(lower_idx)
            boundary_trace.append(boundary_idx)
            terminal_trace.append(terminal_idx)
            interior = torch.cat((pools_cpu["ordinary"][ordinary_idx], pools_cpu["lower_layer"][lower_idx]), dim=0).to(device)
            boundary = pools_cpu["boundary"][boundary_idx].to(device)
            terminal = pools_cpu["terminal"][terminal_idx].to(device)
            interior.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            pide = pide_residual(model, interior, intensity, slab)
            boundary_value = boundary_residual(model, boundary, slab)
            interface_value = interface_residual(model, terminal, slab)
            if not (torch.isfinite(pide).all() and torch.isfinite(boundary_value).all() and torch.isfinite(interface_value).all()):
                raise FloatingPointError(f"nonfinite slab residual at slab={slab}, step={step}")
            pide_loss = pide.square().mean()
            boundary_loss = boundary_value.square().mean()
            interface_loss = interface_value.square().mean()
            loss = pide_loss + 10.0 * boundary_loss + 10.0 * interface_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite slab loss at slab={slab}, step={step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(active_parameters, max_norm=float("inf"))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite gradient at slab={slab}, step={step}")
            before_value=float(loss.detach().cpu())
            g=flatten([p.grad for p in active_parameters]);theta=flatten(active_parameters)
            def objective():
                rr=pide_residual(model,interior,intensity,slab)
                bb=boundary_residual(model,boundary,slab)
                ii=interface_residual(model,terminal,slab)
                return rr.square().mean()+10*bb.square().mean()+10*ii.square().mean()
            protection,proposal=protected_step(active_parameters,optimizer,before_value,objective)
            delta=flatten(active_parameters)-theta
            for key,value in zip(adam_traces,[g,delta,flatten([optimizer.state[p]['exp_avg'] for p in active_parameters]),flatten([optimizer.state[p]['exp_avg_sq'] for p in active_parameters])]):adam_traces[key].append(value)
            histories.append({
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "pide_mse": float(pide_loss.detach().cpu()),
                "boundary_mse": float(boundary_loss.detach().cpu()),
                "interface_mse": float(interface_loss.detach().cpu()),
                "gradient_norm": float(gradient_norm.detach().cpu()),
                "protected_loss_after":protection["loss_after"],
                "gradient_dot_actual_delta":float(g.astype(float)@delta.astype(float)),
                "actual_step_norm":float(np.linalg.norm(delta.astype(float))),
                "protection":protection,
            })
            if step in CHECKPOINT_STEPS:
                save_optimizer(step)
                checkpoint_hashes[str(step)] = save_checkpoint(model, slab_dir / f"checkpoint_{step:04d}.pt", intensity, seed, slab, step)
                write_json(slab_dir/"loss_history_partial.json",histories)
                partial=slab_dir/"adam_trace_partial.npz"
                temporary=slab_dir/"adam_trace_partial.tmp"
                with temporary.open("wb") as f:np.savez_compressed(f,**{k:np.stack(v) for k,v in adam_traces.items()})
                temporary.replace(partial)
                write_json(slab_dir/"slab_progress.json",{"step":step,"checkpoint_sha":checkpoint_hashes[str(step)],"optimizer_sha":optimizer_hashes[str(step)],"trace_sha":sha256_file(partial),"history_sha":sha256_file(slab_dir/"loss_history_partial.json")})

        (slab_dir/"adam_trace_partial.npz").replace(slab_dir/"adam_trace.npz")
        write_json(slab_dir/"slab_progress.json",{"state":"complete","step":1500,"trace_sha":sha256_file(slab_dir/"adam_trace.npz")})
        batch_path = slab_dir / "batch_indices.npz"
        np.savez_compressed(batch_path, ordinary=np.asarray(ordinary_trace), lower_layer=np.asarray(lower_trace), boundary=np.asarray(boundary_trace), terminal=np.asarray(terminal_trace))
        batch_sha = sha256_file(batch_path)
        write_json(slab_dir / "loss_history.json", histories)
        write_json(slab_dir / "checkpoint_hashes.json", checkpoint_hashes)
        complete_path = run_dir / f"complete_after_slab_{slab}.pt"
        final_checkpoint_sha = save_checkpoint(model, complete_path, intensity, seed, slab, STEPS_PER_SLAB)
        slab_records[str(slab)] = {
            "slab": slab,
            "order_position": list(SLAB_ORDER).index(slab),
            "steps": STEPS_PER_SLAB,
            "checkpoint_hashes": checkpoint_hashes,
            "optimizer_checkpoint_hashes":optimizer_hashes,
            "adam_trace_sha256":sha256_file(slab_dir/"adam_trace.npz"),
            "adam_proposals":1500,
            "accepted_proposals":sum(x["protection"]["accepted"] for x in histories),
            "blocked_proposals":sum(x["protection"]["blocked"] for x in histories),
            "nonzero_parameter_steps":sum(x["actual_step_norm"]>0 for x in histories),
            "complete_checkpoint": str(complete_path),
            "complete_checkpoint_sha256": final_checkpoint_sha,
            "collocation_seed": COLLOCATION_SEED,
            "pool_sha256": pool_sha,
            "batch_indices_sha256": batch_sha,
            "pool_sizes": {key: int(value.shape[0]) for key, value in pools_cpu.items()},
            "batch_quota": {"ordinary": ORDINARY_BATCH, "lower_layer": LOWER_BATCH, "interior": INTERIOR_BATCH, "boundary": BOUNDARY_BATCH, "terminal": TERMINAL_BATCH},
        }

    export_path = run_dir / "model_cpu_float64.pt"
    export_hash = export_cpu_float64(model, export_path, intensity, seed, final_checkpoint_sha) if selected_slab==0 else None
    manifest = {
        "format": "v2.6.3-common-collocation-time-slab-branch",
        "intensity": float(intensity),
        "seed": int(seed),
        "device": str(device),
        "dtype": "float32",
        "slab_order": list(SLAB_ORDER),
        "slab_edges": list(SLAB_EDGES),
        "steps_per_slab": STEPS_PER_SLAB,
        "identity": identity(),
        "constants": jsonable_constants(),
        "collocation_seed": COLLOCATION_SEED,
        "initial_parameter_sha256": initial_parameter_sha,
        "slabs": slab_records,
        "final_checkpoint_sha256": final_checkpoint_sha,
        "cpu_float64_export": str(export_path),
        "cpu_float64_export_sha256": export_hash,
        "fallback": None,
        "stderr": "",
        "elapsed_seconds": time.time() - started + (0.0 if previous is None else previous["elapsed_seconds"]),
        "completed_slabs": sorted(slab_records),
        "current_slab_elapsed_seconds": time.time()-started,
        "torch_version": str(torch.__version__),
        "optimizer_rule":"direction_protected_armijo_adam_v1",
    }
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slab",type=int,choices=(3,2,1,0),required=True)
    args = parser.parse_args()
    if args.intensity not in INTENSITIES:
        raise SystemExit("invalid intensity")
    print(json.dumps(train_branch(args.intensity, args.seed, args.run_dir, args.slab), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
