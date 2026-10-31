"""New fixed-feature collocation control; no archived model/table is read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq

WD, UPPER, SIGMA = 0.35, 1.55, 0.06
LENGTH = UPPER - WD
MARKS = (-0.6, 0.4)
PROBABILITIES = (0.7, 0.3)
ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def features(seed, count=128):
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.uniform(-3.0, 3.0, (count, 2)), rng.uniform(-3.0, 3.0, count)


def envelope_jet(t, w):
    x = (w - WD) / LENGTH
    g = x * (1 - x)
    return ((1 - t) * g, -g, (1 - t) * (1 - 2*x) / LENGTH,
            -2 * (1 - t) * np.ones_like(x) / LENGTH**2)


def basis_jet(t, w, weights, bias):
    """Value, forward-time derivative, water derivative, water curvature."""
    t, w = np.asarray(t).reshape(-1), np.asarray(w).reshape(-1)
    z = np.column_stack((2*t-1, 2*(w-WD)/LENGTH-1)) @ weights.T + bias
    h = np.tanh(z)
    dz = 1 - h*h
    vt, vw = 2*weights[:, 0], 2*weights[:, 1]/LENGTH
    ones, zeros = np.ones((len(t), 1)), np.zeros((len(t), 1))
    raw = np.column_stack((ones, h))
    raw_t = np.column_stack((zeros, dz*vt))
    raw_w = np.column_stack((zeros, dz*vw))
    raw_ww = np.column_stack((zeros, -2*h*dz*vw**2))
    e, et, ew, eww = [a[:, None] for a in envelope_jet(t, w)]
    return (e*raw, et*raw+e*raw_t, ew*raw+e*raw_w,
            eww*raw+2*ew*raw_w+e*raw_ww)


def basis_values(t, w, weights, bias):
    t, w = np.asarray(t).reshape(-1), np.asarray(w).reshape(-1)
    result = np.zeros((len(t), len(bias)+1))
    alive = (w > WD) & (w < UPPER)
    if alive.any():
        result[alive] = basis_jet(t[alive], w[alive], weights, bias)[0]
    return result


def operator_basis(t, w, weights, bias, intensity):
    value, dt, dw, dww = basis_jet(t, w, weights, bias)
    result = dt + (0.04-0.08*w)[:, None]*dw + 0.5*SIGMA**2*dww
    if intensity != 0:
        jumps = sum(p*basis_values(t, w*np.exp(0.30*m), weights, bias)
                    for m, p in zip(MARKS, PROBABILITIES))
        result = result + intensity*(jumps-value)
    return result


def analytic_jet(t, w):
    """Independent closed-form control, with zero manufactured boundaries."""
    x = (w - WD) / LENGTH
    c = 0.1+0.2*x+0.3*t
    r = np.exp(0.2*t)*np.sin(np.pi*x)
    h = np.column_stack((c, r))
    ht = np.column_stack((np.full_like(t, .3), .2*r))
    hw = np.column_stack((np.full_like(t, .2/LENGTH),
                          np.exp(.2*t)*np.pi*np.cos(np.pi*x)/LENGTH))
    hww = np.column_stack((np.zeros_like(t), -np.pi**2*r/LENGTH**2))
    e, et, ew, eww = [a[:, None] for a in envelope_jet(t, w)]
    return e*h, et*h+e*ht, ew*h+e*hw, eww*h+2*ew*hw+e*hww


def analytic_values(t, w):
    result = np.zeros((len(t), 2))
    alive = (w > WD) & (w < UPPER)
    if alive.any():
        result[alive] = analytic_jet(t[alive], w[alive])[0]
    return result


def forcing(t, w, intensity, coupled=False):
    value, dt, dw, dww = analytic_jet(t, w)
    local = dt + (0.04-.08*w)[:, None]*dw + .5*SIGMA**2*dww
    if coupled:
        local = np.repeat(local.sum(axis=1, keepdims=True), 2, axis=1)
    if intensity != 0:
        jumps = sum(p*analytic_values(t, w*np.exp(.30*m))
                    for m, p in zip(MARKS, PROBABILITIES))
        local += intensity*(jumps-value)
    return local


def points(nt, nw, shift=.5):
    t, x = np.meshgrid((np.arange(nt)+shift)/nt,
                       (np.arange(nw)+shift)/nw, indexing="ij")
    return t.ravel(), WD+LENGTH*x.ravel()


def solve(a, target):
    norms = np.linalg.norm(a, axis=0)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise FloatingPointError("invalid feature column")
    normalized, _, rank, singular = lstsq(a/norms, target,
                                         cond=1e-12, lapack_driver="gelsd")
    return normalized/norms[:, None], int(rank), singular


def run(out, seed, intensity):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    weights, bias = features(seed)
    t, w = points(48, 64)
    a = operator_basis(t, w, weights, bias, intensity)
    rhs = forcing(t, w, intensity)
    coeff, rank, singular = solve(a, rhs)
    tv, wv = points(37, 53, shift=.37)
    jets = basis_jet(tv, wv, weights, bias)
    actual = [j@coeff for j in jets]
    expected = analytic_jet(tv, wv)
    errors = [np.max(np.abs(p-q), axis=0) for p, q in zip(actual, expected)]
    residual = operator_basis(tv, wv, weights, bias, intensity)@coeff - forcing(tv, wv, intensity)
    residual_max = np.max(np.abs(residual), axis=0)
    finite = all(np.isfinite(v).all() for v in [*actual, coeff, residual, singular])
    passed = bool(finite and max(errors[0]) <= 1e-4
                  and max(np.concatenate(errors[1:])) <= 1e-3
                  and max(residual_max) <= 1e-3)
    np.savez(out/"payload.npz", weights=weights, bias=bias, coefficients=coeff,
             singular_values=singular, check_t=tv, check_w=wv,
             check_values=actual[0], check_residual=residual)
    result = dict(seed=seed, intensity=intensity, training_rows=len(t),
                  independent_analytic_rows=len(tv), raw_feature_count=len(bias)+1,
                  effective_rank=rank, normalized_singular_max=float(singular[0]),
                  normalized_singular_min=float(singular[-1]),
                  coeff_norm=np.linalg.norm(coeff, axis=0).tolist(),
                  error_by_channel=dict(value=errors[0].tolist(), time=errors[1].tolist(),
                                        water=errors[2].tolist(), water_curvature=errors[3].tolist(),
                                        operator=residual_max.tolist()),
                  training_residual_rms=np.sqrt(np.mean((a@coeff-rhs)**2, axis=0)).tolist(),
                  finite=bool(finite), control_pass=passed,
                  elapsed_seconds=time.monotonic()-start, device="cpu", dtype="float64",
                  fallback_count=0, physical_candidate_fits=0, evaluator_calls=0, gate_calls=0,
                  payload_sha256=sha(out/"payload.npz"),
                  source_sha256=sha(__file__), protocol_sha256=sha(ROOT/"protocol.md"),
                  scope="manufactured forcing only; not physical risk acceptance")
    write_json(out/"result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--intensity", type=float, required=True)
    args = parser.parse_args()
    if args.seed not in (20260928, 20260929) or args.intensity not in (0., .12):
        raise ValueError("unregistered control assignment")
    run(args.out, args.seed, args.intensity)
