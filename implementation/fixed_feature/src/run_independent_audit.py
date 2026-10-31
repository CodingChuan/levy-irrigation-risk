"""One independent physical audit; atomic caches permit quota suspension."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix
import audit_adapter as a
import reference_model as ref_model

RUN=a.ROOT/'runs/independent_audit_r1'
SUBJECT=None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    for mapping in ('source_manifest.json','parent_binding.json'):
        for path,digest in json.loads((a.ROOT/mapping).read_text()).items():
            if sha(path)!=digest:raise RuntimeError('source/parent changed: '+path)
    for path,digest in json.loads((SUBJECT/'checkpoint_manifest.json').read_text()).items():
        if sha(path)!=digest:raise RuntimeError('candidate payload changed')


def write_npz(path,**payload):
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez(f,**payload)
    tmp.replace(path)


def reference_factory(policy,intensity,upper=1.55,h=.0025):
    identity={'policy':str(policy),'intensity':float(intensity),'upper':float(upper),'h':float(h),
              'reference_eval_sha256':sha(a.ROOT/'audit_src/reference_eval.py'),'reference_model_sha256':sha(a.ROOT/'audit_src/reference_model.py')}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();cache=a.ROOT/'runs/reference_cache';cache.mkdir(exist_ok=True)
    path=cache/(key+'.npz');meta=cache/(key+'.json')
    if path.exists():
        if meta.exists() and sha(path)!=json.loads(meta.read_text())['payload_sha256']:raise RuntimeError('reference cache payload changed')
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['identity']))!=identity:raise RuntimeError('reference cache identity changed')
            obj=object.__new__(a.original.FixedPolicyReference)
            for k in ('policy','intensity','upper','h'):setattr(obj,k,identity[k])
            obj.grid=data['grid'].copy();obj.post=data['post'].copy();obj.pre=data['pre'].copy();obj.day_terminal=data['terminal'].copy()
            obj.row=ref_model.ModelRow(upper=float(upper),h=float(h),intensity=float(intensity),grid=obj.grid,
                                      Q=csr_matrix(data['Q']),P=data['P'].copy(),left_rates=data['left'].copy(),right_rates=data['right'].copy())
            obj._tau_cache={}
            if 'tau_0' in data:
                obj._tau_cache[tuple(float(x) for x in a.original.TAU_AUDIT)]={float(t):data[f'tau_{i}'].copy() for i,t in enumerate(a.original.TAU_AUDIT)}
        if not meta.exists():a.atomic_json(meta,{'identity':identity,'payload_sha256':sha(path),'restored_from_atomic_payload':True})
        return obj
    obj=a.original.FixedPolicyReference(policy,intensity,upper,h)
    payload=dict(identity=np.array(json.dumps(identity,sort_keys=True)),grid=obj.grid,post=obj.post,pre=obj.pre,terminal=obj.day_terminal,
                 Q=obj.row.Q.toarray(),P=obj.row.P,left=obj.row.left_rates,right=obj.row.right_rates)
    if upper==1.55 and h==.0025:
        for i,t in enumerate(a.original.TAU_AUDIT):payload[f'tau_{i}']=obj.values_at_taus()[float(t)]
    write_npz(path,**payload)
    a.atomic_json(meta,{'identity':identity,'payload_sha256':sha(path),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return obj


def main():
    start=time.monotonic();torch.set_num_threads(1);RUN.mkdir(exist_ok=True)
    if (RUN/'gate_result.json').exists():raise RuntimeError('completed scientific gate must not be repeated')
    verify();a.verify_imports()
    metrics=RUN/'metrics.json';residuals=RUN/'residuals.npz';traces=RUN/'traces.npz'
    model=a.AuditModel(SUBJECT,cache=RUN/'prediction_cache')
    a.install(model);a.original.get_reference=reference_factory
    a.atomic_json(RUN/'status.json',{'state':'independent_evaluation','scientific_gate_calls':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if metrics.exists():
        if not residuals.exists() or not traces.exists():raise RuntimeError('incomplete metric checkpoint')
        row=json.loads(metrics.read_text())
    else:
        row,arrays=a.original.branch_audit(SUBJECT/'checkpoint_manifest.json',SUBJECT)
        write_npz(residuals,**arrays['residuals']);write_npz(traces,**arrays['traces'])
        a.atomic_json(metrics,row)
    verify()
    if (RUN/'gate_started.json').exists():raise RuntimeError('gate start exists without completion: inspect before any action')
    a.atomic_json(RUN/'gate_started.json',{'metric_sha256':sha(metrics),'frozen_gate_source_sha256':sha(a.ROOT/'audit_src/reference_eval.py'),'scientific_gate_calls':1})
    gate=a.original._gate_branch(row)
    a.atomic_json(RUN/'gate_result.json',gate)
    verify()
    hashes={str(p):sha(p) for p in RUN.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','status.json')}
    a.atomic_json(RUN/'artifact_manifest.json',hashes)
    result={'state':'complete','scientific_gate_calls':1,'named_checks':len(gate['checks']),'pass':gate['pass'],
            'passed_checks':sum(gate['checks'].values()),'elapsed_seconds':time.monotonic()-start,'source_manifest_sha256':sha(a.ROOT/'source_manifest.json'),
            'candidate_manifest_sha256':sha(SUBJECT/'checkpoint_manifest.json'),
            'metric_sha256':sha(metrics),'gate_result_sha256':sha(RUN/'gate_result.json'),'fallback_count':0,
            'representation':'fixed hidden features and linear readouts, absorbing-heat terminal lifting',
            'pide_derivatives':'analytic neural/heat jets, independently checked against Torch AD; original sampling and thresholds unchanged'}
    a.atomic_json(RUN/'status.json',result);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--candidate-run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();SUBJECT=args.candidate_run.resolve();RUN=args.out.resolve()
    main()
