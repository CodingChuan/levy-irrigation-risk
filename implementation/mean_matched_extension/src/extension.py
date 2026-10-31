"""Prospectively specified mean-drift control and saved-policy readout."""
from pathlib import Path
import argparse, hashlib, json, signal, time
import numpy as np
from scipy.linalg import expm
import model, constrained

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'control_reference_v1/runs/20260919T044919602355Z_green_38c0009b82'
KAPPA = .12 * float(np.dot(model.MARK_WEIGHTS, np.expm1(model.JUMP_SCALE*model.MARKS)))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, x):
    p = Path(p); tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n'); tmp.replace(p)

def matched_row(upper, h):
    grid, q, left, right = model.build_generator(upper, h, 0.)
    q = q.tolil(); delta = KAPPA*grid
    left = left.copy(); right = right.copy()
    for i in range(1,len(grid)-1):
        left[i] -= delta[i]/(2*h); right[i] += delta[i]/(2*h)
        q[i,i-1] -= delta[i]/(2*h); q[i,i+1] += delta[i]/(2*h)
    q = q.tocsr(); arr=q.toarray(); off=arr.copy(); np.fill_diagonal(off,0)
    assert off.min() >= -1e-12
    assert np.max(np.abs(arr.sum(axis=1))) < 1e-10
    assert np.all(arr[[0,-1]]==0)
    p=expm(arr)
    assert np.isfinite(p).all() and p.min() >= -1e-12
    assert np.max(np.abs(p.sum(axis=1)-1)) < 1e-10
    return model.ModelRow(upper,h,0.,grid,q,p,left,right)

def safe(s): return s.replace('.','p').replace('-','m').replace(' ','_')
def load_old_model(row):
    f=OLD/f"row_{safe(row['row_key'])}__model.npz"
    return f

def audit_old_file(p, bindings):
    p=Path(p); manifest=json.loads((OLD/'file_hashes.json').read_text())
    assert sha(p)==manifest[p.name],p
    bindings[str(p.resolve())]=sha(p)

def controls():
    t=time.monotonic(); rows=[]; bindings={}
    # RED: the old zero-jump generator has not matched the jump local mean.
    g,q,_,_=model.build_generator(1.55,.005,0.)
    err=float(np.max(np.abs((q@g)[1:-1]-(model.drift(g)+KAPPA*g)[1:-1])))
    assert err>1e-3
    write(ROOT/'runs/red.json',{'expected_failure':'old no-jump first moment differs from target mean drift','max_error':err,'status':'RED reproduced'})
    saved=json.loads((OLD/'matrix_scalar.json').read_text())['rows']
    for upper,h in model.row_specifications():
        m=matched_row(upper,h); g=m.grid
        target=model.drift(g)+KAPPA*g
        e1=float(np.max(np.abs((m.Q@g)[1:-1]-target[1:-1])))
        e2=float(np.max(np.abs((m.Q@(g*g))[1:-1]-(2*g*target+model.DIFFUSION**2)[1:-1])))
        assert e1<1e-10 and e2<1e-10
        jg,jq,_,_=model.build_generator(upper,h,.12)
        inside=(g*np.exp(-.18)>g[0]) & (g*np.exp(.12)<g[-1])
        em=float(np.max(np.abs((jq@g-m.Q@g)[inside])))
        assert em<1e-10
        for lam in (0.,.12):
            oldrow=next(r for r in saved if r['upper']==upper and r['h']==h and r['intensity']==lam)
            f=load_old_model(oldrow);audit_old_file(f,bindings)
            with np.load(f) as z:
                _,oq,_,_=model.build_generator(upper,h,lam)
                assert np.array_equal(oq.toarray(),z['Q'])
        rows.append({'upper':upper,'h':h,'linear_error':e1,'quadratic_error':e2,'jump_inside_mean_error':em,'matched_rowsum_error':float(np.max(abs(m.P.sum(1)-1)))})
    # Check the new drift increment in isolation on all frozen strata.
    assert KAPPA<0 and model.DRIFT_SLOPE==.08 and constrained.BISECTION_STEPS==24 and model.EPSILON==.2
    res={'status':'PASS','kappa':KAPPA,'rows':rows,'passing_original_generator_controls':10,'elapsed_seconds':time.monotonic()-t,'bindings':bindings,'scientific_optimizer_calls':0}
    write(ROOT/'runs/controls.json',res);print(json.dumps(res),flush=True)

def run_matrix():
    assert json.loads((ROOT/'runs/controls.json').read_text())['status']=='PASS'
    t=time.monotonic(); rows=[]
    for idx,(upper,h) in enumerate(model.row_specifications()):
        signal.alarm(720)
        out=ROOT/'runs'/f'matched_{idx}';out.mkdir(exist_ok=False)
        row=matched_row(upper,h)
        np.savez_compressed(out/'model.npz',grid=row.grid,Q=row.Q.toarray(),P=row.P)
        def callback(rec):
            s=rec.solution; f=out/(safe(rec.label)+'.npz')
            np.savez_compressed(f,actions=s.actions,C=s.C,R=s.R,V=s.V,successor_budget=s.successor_budget)
            write(out/(safe(rec.label)+'.json'),{'summary':rec.summary,'arrays_file':f.name,'arrays_sha256':sha(f)})
            write(ROOT/'runs/progress.json',{'row':idx,'upper':upper,'h':h,'completed_trials':rec.label,'elapsed_seconds':time.monotonic()-t})
        result=constrained.solve_row(row,on_trial=callback)
        summary=constrained.row_summary(result)
        write(out/'result.json',summary)
        if result.status=='FEASIBLE': assert summary['gate_summary']['pass'],summary
        else: assert result.status=='INFEASIBLE',summary
        rows.append(summary);write(ROOT/'runs/matrix.json',{'rows':rows,'elapsed_seconds':time.monotonic()-t,'complete':len(rows)==5,'kappa':KAPPA})
        print(json.dumps({'row':idx,'status':result.status,'min_risk':result.initial_risk_min,'cost':summary['mixture_cost'],'elapsed_seconds':time.monotonic()-t}),flush=True)
        del result
        signal.alarm(0)
    return rows

def forward(actions,p,grid):
    mass=np.zeros((5,len(grid)));pos=(model.W0-grid[0])/(grid[1]-grid[0]);i=int(round(pos));assert abs(pos-i)<1e-8;mass[-1,i]=1
    costs=[];fail=[]
    for day in range(20):
        post=np.zeros_like(mass);post[:,0]=mass[:,0];post[:,-1]=mass[:,-1];cost=0.
        for b in range(5):
            for j in range(1,len(grid)-1):
                q=float(actions[day,b,j]);bs=model.successor_budget_index(b,q)
                cost+=mass[b,j]*q
                loc=(grid[j]+q-grid[0])/(grid[1]-grid[0]);dest=min(int(round(loc)),len(grid)-1)
                assert abs(loc-round(loc))<1e-8
                post[bs,dest]+=mass[b,j]
        mass=post@p
        assert mass.min()>-1e-12 and abs(mass.sum()-1)<1e-10
        costs.append(cost);fail.append(float(mass[:,0].sum()))
    return {'daily_water':costs,'cumulative_failure':fail,'total_water':float(sum(costs)),'final_risk':float(mass[:,0].sum())}

def read_policies():
    bindings={};summaries=[]
    for row in json.loads((OLD/'matrix_scalar.json').read_text())['rows']:
        if row['h']!=.0025 or row['upper']!=1.55:continue
        mf=load_old_model(row);audit_old_file(mf,bindings)
        with np.load(mf) as z:g=z['grid'];p=z['P']
        result={'intensity':row['intensity'],'alpha':row['mixture_alpha'],'endpoints':{}}
        for label in ['mu_0',row['risky_label'],row['safe_label']]:
            meta=OLD/f"row_{safe(row['row_key'])}__trial_{safe(label)}.json";audit_old_file(meta,bindings)
            d=json.loads(meta.read_text());f=OLD/d['arrays_file'];audit_old_file(f,bindings);assert sha(f)==d['arrays_sha256']
            with np.load(f) as z:actions=z['actions']; c=z['C'];risk=z['R']
            stats=forward(actions,p,g)
            assert abs(stats['total_water']-d['summary']['initial_C'])<1e-10
            assert abs(stats['final_risk']-d['summary']['initial_R'])<1e-10
            dest=ROOT/'runs'/f"policy_{row['intensity']}_{'zero' if label=='mu_0' else 'risky' if label==row['risky_label'] else 'safe'}.npz"
            np.savez_compressed(dest,grid=g,actions=actions)
            result['endpoints'][label]={'source':str(f.resolve()),'daily':stats,'actions_file':dest.name,'initial':d['summary']}
        ar=row['mixture_alpha'];r=result['endpoints'][row['risky_label']]['daily'];s=result['endpoints'][row['safe_label']]['daily']
        mix=ar*np.asarray(r['daily_water'])+(1-ar)*np.asarray(s['daily_water'])
        assert abs(mix.sum()-row['mixture_cost'])<1e-10
        result.update({'mixture_daily_water':mix.tolist(),'mixture_cost':row['mixture_cost'],'mixture_risk':row['mixture_risk'],'risky_label':row['risky_label'],'safe_label':row['safe_label']})
        summaries.append(result)
    write(ROOT/'runs/policy_readout.json',{'status':'PASS','rows':summaries,'bindings':bindings,'new_optimization_calls':0})
    print(json.dumps({'policy_readout':'PASS','rows':len(summaries)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['controls','matrix','policies']);args=parser.parse_args()
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('frozen execution cap')))
    signal.alarm(120 if args.mode!='matrix' else 3600)
    try:
        {'controls':controls,'matrix':run_matrix,'policies':read_policies}[args.mode]()
    except BaseException:
        import traceback
        (ROOT/'runs'/f'{args.mode}_failure.txt').write_text(traceback.format_exc())
        raise
