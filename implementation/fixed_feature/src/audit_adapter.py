"""Read-only new-representation adapter; original sampling/gate unchanged."""
from __future__ import annotations
import hashlib
import importlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT
sys.path.insert(0,str(ROOT/'audit_src'));sys.path.insert(0,str(CANDIDATE/'src'))
import solve_day as s
import manufactured_control as m
import heat_lifting as heat
import reference_eval as original


def verify_imports():
    expected={'solve_day':CANDIDATE/'src/solve_day.py','manufactured_control':CANDIDATE/'src/manufactured_control.py',
              'heat_lifting':CANDIDATE/'src/heat_lifting.py','reference_eval':ROOT/'audit_src/reference_eval.py',
              'model':ROOT/'audit_src/model.py','operators':ROOT/'audit_src/operators.py','reference_model':ROOT/'audit_src/reference_model.py'}
    for name,path in expected.items():
        if Path(importlib.import_module(name).__file__).resolve()!=path.resolve():raise RuntimeError('wrong import topology: '+name)


class AuditModel:
    def __init__(self,run,cache=None):
        run=Path(run).resolve()
        with np.load(run/'day_19.npz',allow_pickle=False) as data:identity=json.loads(str(data['metadata']))
        self.policy=identity['policy'];self.intensity=identity['intensity'];self.seed=identity['seed'];self.slabs={};self.trace_records=[]
        if self.policy not in ('zero','schedule','buffer') or self.intensity not in (0.,.12) or self.seed not in (20260928,20260929):
            raise ValueError('unregistered audit identity')
        self.cache=Path(cache) if cache is not None else None
        if self.cache is not None:self.cache.mkdir(exist_ok=True,parents=True)
        digest=m.sha(run.parents[1]/'source_manifest.json')
        for path,expected in json.loads((Path(run)/'checkpoint_manifest.json').read_text()).items():
            if m.sha(path)!=expected:raise RuntimeError('candidate output changed before audit')
        for day in range(20):
            path=Path(run)/f'day_{day:02d}.npz'
            meta,slab,pre=s.load_checkpoint(path,self.policy,self.seed,digest)
            if meta['day']!=day or meta['intensity']!=self.intensity:raise ValueError('wrong candidate identity')
            self.slabs[day]=slab
            if pre is not None:self.trace_records.append({**meta,'max_abs_error_cost':meta['trace_max_abs_error_cost'],
                                                        'max_abs_error_risk':meta['trace_max_abs_error_risk']})
            if slab.trace is not None:setattr(self,f'terminal_trace_{day}',torch.from_numpy(slab.trace.values.copy()))
        self.trace_grid=torch.from_numpy(self.slabs[0].trace.grid.copy())

    def forward_day(self,day,states,budgets):
        a=states.detach().cpu().numpy();bi=budgets.detach().cpu().numpy().astype(int)
        if day==20:
            values=np.zeros((len(a),2));values[a[:,1]<=m.WD,1]=1.
            return torch.from_numpy(values)
        slab=self.slabs[day]
        if self.cache is not None and len(a)>=5 and np.all(a[:,0]==a[0,0]):
            key=hashlib.sha256(a.astype('<f8').tobytes()+str(day).encode()).hexdigest()
            path=self.cache/(key+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as data:all_values=data['all_values'].copy()
            else:
                all_values=slab.values_all(a[:,0],a[:,1]);tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.savez(f,all_values=all_values,day=day,states=a,source_sha=np.array(m.sha(CANDIDATE/'source_manifest.json')))
                tmp.replace(path)
            values=all_values[np.arange(len(a)),bi]
        else:values=slab.values(a[:,0],a[:,1],bi)
        return torch.from_numpy(values)

    def interpolate_trace(self,day,water,budgets):
        return torch.from_numpy(self.slabs[day].trace.jet(water.detach().cpu().numpy(),budgets.detach().cpu().numpy().astype(int))[0])


def analytic_residual(model,day,states,budgets):
    a=states.detach().cpu().numpy();bi=budgets.detach().cpu().numpy().astype(int);t,w=a[:,0],a[:,1];slab=model.slabs[day]
    value,dt,dw,dww=slab.jet(t,w,bi)
    result=dt+(.04-.08*w)[:,None]*dw+.5*m.SIGMA**2*dww
    jumps=sum(p*slab.values(t,w*np.exp(.30*mark),bi) for mark,p in zip(m.MARKS,m.PROBABILITIES))
    return torch.from_numpy(result+model.intensity*(jumps-value))


def install(model):
    """Only representation loader and derivative implementation are adapted."""
    verify_imports()
    original.load_model=lambda path:(model,{'policy':model.policy,'intensity':model.intensity,'seed':model.seed,'trace_records':model.trace_records})
    original.pide_residual=analytic_residual


def atomic_json(path,payload):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');tmp.replace(path)
