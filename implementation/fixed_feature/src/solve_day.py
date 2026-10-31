"""One atomic day of the new fixed-feature solver; no reference access."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import manufactured_control as m
from resolution_control import nested_features
import heat_lifting as p


class Slab:
    def __init__(self,seed,coefficients,trace=None):
        self.seed=seed;self.weights,self.bias=nested_features(seed,512)
        self.coefficients=np.asarray(coefficients);self.trace=trace
        if self.coefficients.shape!=(5,513,2):raise ValueError('coefficient shape')

    def values(self,t,w,bi):
        t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
        result=p.base_values(t,w,bi,self.trace)
        alive=(t<1)&(w>m.WD)&(w<m.UPPER)
        ids=np.flatnonzero(alive)
        for offset in range(0,len(ids),512):
            ii=ids[offset:offset+512]
            phi=m.basis_jet(t[ii],w[ii],self.weights,self.bias)[0]
            result[ii]+=np.einsum('ij,ijk->ik',phi,self.coefficients[bi[ii]])
        return result

    def jet(self,t,w,bi):
        result=p.base_jet(t,w,bi,self.trace)
        basis=m.basis_jet(t,w,self.weights,self.bias)
        return tuple(x+np.einsum('ij,ijk->ik',phi,self.coefficients[bi]) for x,phi in zip(result,basis))

    def jet_all(self,t,w):
        t,w=np.asarray(t),np.asarray(w);out=[np.zeros((len(t),5,2)) for _ in range(4)]
        lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper);out[0][lower,:,1]=1.
        if self.trace is not None and np.any(terminal):out[0][terminal]=np.stack([self.trace.jet(w[terminal],np.full(terminal.sum(),b))[0] for b in range(5)],axis=1)
        alive=~(lower|upper|terminal)
        if np.any(alive):
            ii=np.flatnonzero(alive)
            for offset in range(0,len(ii),64):
                jj=ii[offset:offset+64];base=p.base_all_jet(t[jj],w[jj],self.trace);basis=m.basis_jet(t[jj],w[jj],self.weights,self.bias)
                for k,(a,b) in enumerate(zip(base,basis)):out[k][jj]=a+np.einsum('nj,bjc->nbc',b,self.coefficients)
        return tuple(out)

    def values_all(self,t,w):
        t,w=np.asarray(t),np.asarray(w)
        result=p.base_all_values(t,w,self.trace)
        ids=np.flatnonzero((t<1)&(w>m.WD)&(w<m.UPPER))
        for offset in range(0,len(ids),512):
            ii=ids[offset:offset+512]
            phi=m.basis_jet(t[ii],w[ii],self.weights,self.bias)[0]
            result[ii]+=np.einsum('nj,bjc->nbc',phi,self.coefficients)
        return result


def base_operator(t,w,bi,trace,intensity):
    v,dt,dw,dww=p.base_jet(t,w,bi,trace)
    value=dt+(.04-.08*w)[:,None]*dw+.5*m.SIGMA**2*dww
    if intensity:
        jumped=sum(prob*p.base_values(t,w*np.exp(.3*mark),bi,trace) for mark,prob in zip(m.MARKS,m.PROBABILITIES))
        value+=intensity*(jumped-v)
    return value


def base_operator_all(t,w,trace,intensity):
    v,dt,dw,dww=p.base_all_jet(t,w,trace)
    value=dt+(.04-.08*w)[:,None,None]*dw+.5*m.SIGMA**2*dww
    if intensity:
        jumped=sum(prob*p.base_all_values(t,w*np.exp(.3*mark),trace) for mark,prob in zip(m.MARKS,m.PROBABILITIES))
        value+=intensity*(jumped-v)
    return value


def save_checkpoint(path,meta,coefficients,trace,pre_trace):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,metadata=np.array(json.dumps(meta,sort_keys=True)),coefficients=coefficients,
                 trace_input=np.empty((0,)) if trace is None else trace.values,
                 pre_trace=np.empty((0,)) if pre_trace is None else pre_trace.values,
                 trace_input_slopes=np.empty((0,)) if trace is None else trace.slopes,
                 pre_trace_slopes=np.empty((0,)) if pre_trace is None else pre_trace.slopes)
    tmp.replace(path)


def load_checkpoint(path,policy,seed,source_sha):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));coefficients=d['coefficients'].copy();tv=d['trace_input'].copy();pv=d['pre_trace'].copy();ts=d['trace_input_slopes'].copy();ps=d['pre_trace_slopes'].copy()
    if (meta['policy'],meta['seed'],meta['source_manifest_sha256'])!=(policy,seed,source_sha):
        raise ValueError('checkpoint identity/source mismatch')
    if not all(np.isfinite(a).all() for a in (coefficients,tv,pv,ts,ps)):raise ValueError('nonfinite checkpoint')
    if ts.shape!=tv.shape or ps.shape!=pv.shape:raise ValueError('analytic slope payload shape mismatch')
    trace=p.OpenTrace(policy,tv) if tv.size else None
    pre=p.OpenTrace(policy,pv) if pv.size else None
    if trace is not None:trace.slopes=ts
    if pre is not None:pre.slopes=ps
    return meta,Slab(seed,coefficients,trace),pre


def run_day(run,day,policy,intensity,seed):
    start=time.monotonic()
    source=m.ROOT/'source_manifest.json';source_sha=m.sha(source)
    mapping=json.loads(source.read_text())
    for path,digest in mapping.items():
        if m.sha(path)!=digest:raise ValueError('source/contract changed before work unit')
    run.mkdir(exist_ok=True,parents=True)
    target=run/f'day_{day:02d}.npz'
    if target.exists():raise FileExistsError('completed day must not be repeated')
    parent_hash=None
    if day==19:trace=None
    else:
        prior=run/f'day_{day+1:02d}.npz'
        meta,_,trace=load_checkpoint(prior,policy,seed,source_sha)
        if meta['day']!=day+1 or meta['intensity']!=intensity or trace is None:raise ValueError('invalid next-day prefix')
        if not meta['trace_pass']:raise ValueError('next day trace not admitted')
        parent_hash=m.sha(prior)
    t,w=m.points(48,64);weights,bias=nested_features(seed,512)
    a=m.operator_basis(t,w,weights,bias,intensity)
    rhs=-base_operator_all(t,w,trace,intensity).reshape(len(t),10)
    coef,rank,singular=m.solve(a,rhs)
    coefficients=coef.reshape(513,5,2).transpose(1,0,2)
    if not np.isfinite(coefficients).all():raise FloatingPointError('nonfinite coefficients')
    slab=Slab(seed,coefficients,trace)
    pre_trace=p.OpenTrace.from_callable(policy,day,slab.values) if day>0 else None
    trace_errors=np.zeros(2)
    if pre_trace is not None:
        rng=np.random.Generator(np.random.PCG64(801247+day*1009+seed))
        ww=rng.uniform(m.WD+1e-8,m.UPPER-1e-8,4096);bi=np.arange(4096)%5
        trace_errors=np.max(np.abs(pre_trace.jet(ww,bi)[0]-p.policy_values(slab.values,policy,day,ww,bi)),axis=0)
    passed=bool(np.isfinite(trace_errors).all() and trace_errors.max()<=1e-4)
    meta=dict(day=day,policy=policy,intensity=intensity,seed=seed,source_manifest_sha256=source_sha,
              parent_checkpoint_sha256=parent_hash,effective_rank=rank,feature_count=513,
              singular_min=float(singular[-1]),singular_max=float(singular[0]),
              train_residual_rms_by_budget_channel=np.sqrt(np.mean((a@coef-rhs)**2,axis=0)).reshape(5,2).tolist(),
              train_residual_max=float(np.max(np.abs(a@coef-rhs))),
              trace_max_abs_error_cost=float(trace_errors[0]),trace_max_abs_error_risk=float(trace_errors[1]),
              trace_pass=passed,finite=True,coefficient_norm_by_budget_channel=np.linalg.norm(coefficients,axis=1).tolist(),
              elapsed_seconds=time.monotonic()-start,device='cpu',dtype='float64',fallback_count=0,
              optimizer_calls=0,readout_solves=1,physical_reference_calls=0,evaluator_calls=0,gate_calls=0,
              completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # A failed partial is retained separately and never admitted as a prefix.
    destination=target if passed else run/f'day_{day:02d}_trace_failed_partial.npz'
    save_checkpoint(destination,meta,coefficients,trace,pre_trace)
    m.write_json(run/f'day_{day:02d}_result.json',{**meta,'checkpoint_sha256':m.sha(destination)})
    if not passed:raise RuntimeError('new physical trace control failed; preserve partial and diagnose')
    m.write_json(run/'progress.json',dict(state='fitting' if day else 'all_days_fitted_independent_audit_pending',
                 completed_days=20-day,total_days=20,next_day=day-1 if day else None,
                 last_checkpoint=str(target),last_checkpoint_sha256=m.sha(target),source_manifest_sha256=source_sha,
                 policy=policy,intensity=intensity,seed=seed,gate_calls=0))
    print(json.dumps(meta,sort_keys=True),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--day',type=int,required=True)
    parser.add_argument('--policy',choices=('zero','schedule','buffer'),required=True)
    parser.add_argument('--intensity',type=float,choices=(0.,.12),required=True)
    parser.add_argument('--seed',type=int,choices=(20260928,20260929),required=True)
    args=parser.parse_args()
    if not 0<=args.day<20:raise ValueError('invalid slab')
    run_day(args.run,args.day,args.policy,args.intensity,args.seed)
