"""Full affected grid with mathematically identical day cases reused."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import log_ndtr
import manufactured_control as m
import heat_lifting as h
from resolution_control import nested_features
from test_physical_lifting import synthetic_post

CASES={'zero':('zero',5),'buffer':('buffer',5),'schedule_active':('schedule',2),'schedule_inactive':('schedule',10),'last':(None,None)}


def case(name):
    start=time.monotonic();policy,day=CASES[name]
    trace=None if policy is None else h.OpenTrace.from_callable(policy,day,synthetic_post)
    grid=m.WD+.0025*np.arange(481);grid[0]=m.WD;grid[-1]=m.UPPER
    t,w,bi=np.meshgrid(np.array([0.,.25,.5,.75]),grid[1:-1],np.arange(5),indexing='ij')
    t,w,bi=t.ravel(),w.ravel(),bi.ravel()
    jet=h.base_jet(t,w,bi,trace)
    lo,hi,origin,coeff=h.trace_polynomials(trace)
    bound=float(np.max(np.sum(np.abs(coeff)*(2*h.GRID_H)**np.arange(4)[None,None,:,None],axis=2))) if trace is not None else 2.
    # Omitted image distance >=2L; erfc bound at maximum delta=1.
    z=2*m.LENGTH/m.SIGMA
    log_tail=np.log(8*max(bound,1.))+np.log(2.)+float(log_ndtr(-z))
    log_tail_dww=log_tail+4*np.log(z+1)-2*np.log(m.SIGMA)
    out=m.ROOT/'runs/heat_support';out.mkdir(exist_ok=True)
    np.savez(out/(name+'.npz'),t=t,w=w,budget=bi,**{f'jet_{k}':a for k,a in enumerate(jet)})
    result={'case':name,'interior_rows':len(t),'finite':bool(all(np.isfinite(j).all() for j in jet)),
            'log_omitted_image_value_bound':float(log_tail),'log_omitted_image_curvature_bound':float(log_tail_dww),
            'pass':bool(all(np.isfinite(j).all() for j in jet) and log_tail_dww<np.log(1e-10)),
            'elapsed_seconds':time.monotonic()-start,'payload_sha256':m.sha(out/(name+'.npz')),
            'evaluator_calls':0,'gate_calls':0,'new_physical_fits':0}
    m.write_json(out/(name+'.json'),result);print(json.dumps(result),flush=True)


def summarize():
    out=m.ROOT/'runs/heat_support';cases={k:json.loads((out/(k+'.json')).read_text()) for k in CASES}
    rows=[]
    for seed in (20260928,20260929):
        weights,bias=nested_features(seed,512);rng=np.random.Generator(np.random.PCG64([seed,20261020]));coefficients=rng.normal(0,.001,(5,513,2))
        cache={}
        with np.load(out/'zero.npz') as data:t,w,bi=[data[k].copy() for k in ('t','w','budget')]
        correction=[np.zeros((len(t),2)) for _ in range(4)]
        for start in range(0,len(t),512):
            sl=slice(start,start+512)
            for k,phi in enumerate(m.basis_jet(t[sl],w[sl],weights,bias)):
                correction[k][sl]=np.einsum('ij,ijk->ik',phi,coefficients[bi[sl]])
        for policy in ('zero','schedule','buffer'):
            finite=True;max_trace=0.;max_interface=0.;count=0
            for day in range(20):
                key='last' if day==19 else ('schedule_active' if day+1<=3 else 'schedule_inactive') if policy=='schedule' else policy
                if key not in cache:
                    with np.load(out/(key+'.npz')) as data:cache[key]=[data[f'jet_{k}'].copy() for k in range(4)]
                finite=finite and all(np.isfinite(a+b).all() for a,b in zip(cache[key],correction))
                if day<19:
                    trace=h.OpenTrace.from_callable(policy,day+1,synthetic_post)
                    rng2=np.random.Generator(np.random.PCG64(801247));ww=rng2.uniform(m.WD+1e-8,m.UPPER-1e-8,4096);bb=np.arange(4096)%5
                    max_trace=max(max_trace,float(np.max(np.abs(trace.jet(ww,bb)[0]-h.policy_values(synthetic_post,policy,day+1,ww,bb)))))
                else:trace=None
                # Terminal and absorbing points are disjoint from kernel calls.
                grid=m.WD+.0025*np.arange(481);grid[0]=m.WD;grid[-1]=m.UPPER
                ww,bb=np.meshgrid(grid,np.arange(5),indexing='ij');ww,bb=ww.ravel(),bb.ravel()
                terminal=h.base_values(np.ones(len(ww)),ww,bb,trace)
                expected=np.zeros_like(terminal) if trace is None else trace.jet(ww,bb)[0];expected[ww<=m.WD]=[0,1]
                max_interface=max(max_interface,float(np.max(np.abs(terminal-expected))))
                for tt in (0.,.25,.5,.75):
                    np.testing.assert_array_equal(h.base_values(np.array([tt,tt]),np.array([m.WD,m.UPPER]),np.array([4,4]),trace),np.array([[0,1],[0,0]]))
                count+=len(t)+800//20+48100//20
            rows.append({'policy':policy,'seed':seed,'unique_support_rows':count,'interior_derivative_rows':len(t)*20,'boundary_rows':800,'terminal_rows':48100,'finite':bool(finite),'trace_max_error':max_trace,'interface_max_error':max_interface,'pass':bool(finite and count==240500 and max_trace<=1e-4 and max_interface<=1e-8)})
    result={'rows':rows,'cases':cases,'pass':all(r['pass'] for r in rows) and all(c['pass'] for c in cases.values()),'cache_scope':'identical synthetic day/trace/base coordinates only, raw seed corrections separately recomputed','physical_fits':0,'gate_calls':0}
    m.write_json(out/'matrix.json',result);print(json.dumps({'pass':result['pass'],'rows':len(rows),'max_trace':max(r['trace_max_error'] for r in rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=CASES);p.add_argument('--summarize',action='store_true');args=p.parse_args()
    summarize() if args.summarize else case(args.case)
