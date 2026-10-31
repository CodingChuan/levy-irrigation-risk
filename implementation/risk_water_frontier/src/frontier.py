"""Risk-target scan with immutable baselines and unchanged Bellman evaluation."""
from pathlib import Path
import json,hashlib,time,signal
import numpy as np
import model,bellman

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT.parent/'mechanism_decision_extension_20261024_r1'
OLD=ROOT.parent/'control_reference_v1/runs/20260919T044919602355Z_green_38c0009b82'
TARGETS=[0.,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    q=Path(str(p)+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def safe(s):return s.replace('.','p').replace('-','m').replace(' ','_')
def inputs():
    bindings={};answer=[]
    oldmanifest=json.loads((OLD/'file_hashes.json').read_text())
    extmanifest=json.loads((EXT/'artifact_manifest.json').read_text())
    for mech in ['Brownian','Mean-matched diffusion','Jump diffusion']:
        if mech=='Mean-matched diffusion':
            row=json.loads((EXT/'runs/matrix.json').read_text())['rows'][2];mf=EXT/'runs/matched_2/model.npz'
            risk=json.loads((EXT/'runs/matched_2/risk_only.json').read_text())['summary'];zero=json.loads((EXT/'runs/matched_2/mu_0.json').read_text())['summary']
            for p in [mf,EXT/'runs/matched_2/risk_only.json',EXT/'runs/matched_2/mu_0.json',EXT/'runs/matrix.json']:
                assert sha(p)==extmanifest[str(p.relative_to(EXT))];bindings[str(p.resolve())]=sha(p)
        else:
            lam=0. if mech=='Brownian' else .12
            row=next(x for x in json.loads((OLD/'matrix_scalar.json').read_text())['rows'] if x['h']==.0025 and x['upper']==1.55 and x['intensity']==lam)
            mf=OLD/f"row_{safe(row['row_key'])}__model.npz";risk=next(x for x in row['trials'] if x['label']=='risk_only');zero=next(x for x in row['trials'] if x['label']=='mu_0')
            for p in [mf,OLD/'matrix_scalar.json']:
                assert sha(p)==oldmanifest[p.name];bindings[str(p.resolve())]=sha(p)
        with np.load(mf) as z:grid=z['grid'];p=z['P']
        assert grid.shape==(481,) and np.max(abs(p.sum(1)-1))<1e-10
        answer.append((mech,row,risk,zero,grid,p))
    write(ROOT/'runs/input_bindings.json',bindings);return answer

def solve_target(eps,zero,grid,p,out):
    trials=[dict(zero)];started=time.monotonic()
    def run(mu,label):
        sol=bellman.solve(mu,p,grid);ev=bellman.evaluate_policy(sol.actions,p,grid);s=bellman.solution_summary(sol,ev)
        idx=int(round((model.W0-grid[0])/(grid[1]-grid[0])))
        s.update(mu=mu,label=label,initial_R=float(sol.R[0,-1,idx]),initial_C=float(sol.C[0,-1,idx]),initial_V=float(sol.V[0,-1,idx]))
        assert s['finite'] and s['bellman_gap']<=1e-9 and s['identity_gap']<=1e-8 and s['policy_eval_R_gap']<=1e-10 and s['policy_eval_C_gap']<=1e-10 and s['policy_budget_feasible']
        assert s['risk_min']>=-1e-10 and s['risk_max']<=1+1e-10 and s['cost_min']>=-1e-10 and s['cost_max']<=.30+1e-10
        trials.append(s);write(out/'trials.json',trials)
        return s,sol
    low=dict(zero);high=None
    for mu in [1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,1024.]:
        s,sol=run(mu,f'bracket_{mu:g}')
        if s['initial_R']<=eps+1e-10:high=s;break
        low=s
    assert high is not None,'frozen bracket exhausted'
    for n in range(24):
        mu=.5*(low['mu']+high['mu']);s,sol=run(mu,f'bisection_{n+1}')
        if s['initial_R']<=eps+1e-10:high=s
        else:low=s
    den=low['initial_R']-high['initial_R']
    alpha=(eps-high['initial_R'])/den if den>1e-14 else 0.
    assert 0<=alpha<=1
    c=alpha*low['initial_C']+(1-alpha)*high['initial_C'];risk=alpha*low['initial_R']+(1-alpha)*high['initial_R']
    dual=max(x['initial_V']-x['mu']*eps for x in trials);gap=c-dual
    assert risk<=eps+1e-10 and -1e-8<=gap<=1e-5
    result={'risk_limit':eps,'status':'FEASIBLE','expected_water':c,'risk':risk,'alpha':alpha,'risky':low,'safe':high,'gap':gap,'trial_count':len(trials)-1,'elapsed_seconds':time.monotonic()-started,'source':'new_target'}
    write(out/'result.json',result);return result

def main():
    start=time.monotonic();ans=[];new=0
    sources=inputs();write(ROOT/'admission.json',{'status':'PASS','risk_limits':TARGETS,'source_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'input_binding_sha':sha(ROOT/'runs/input_bindings.json')})
    for k,(name,row,rmin,zero,g,p) in enumerate(sources):
        for eps in TARGETS:
            assert time.monotonic()-start<1800,'cumulative cap'
            if eps==.2:s={'risk_limit':eps,'status':'FEASIBLE','expected_water':row['mixture_cost'],'risk':row['mixture_risk'],'gap':row['duality_gap'],'source':'reused_original_0.20'}
            elif eps<row['initial_risk_min']-1e-10:s={'risk_limit':eps,'status':'INFEASIBLE','expected_water':None,'risk':None,'source':'reused_minimum_risk'}
            elif zero['initial_R']<=eps+1e-10:s={'risk_limit':eps,'status':'FEASIBLE','expected_water':0.,'risk':zero['initial_R'],'gap':0.,'source':'reused_zero_water'}
            else:
                out=ROOT/'runs'/f'{k}_epsilon_{eps:.2f}';out.mkdir();signal.alarm(120)
                s=solve_target(eps,zero,g,p,out);signal.alarm(0);new+=1
            ans.append({'mechanism':name,**s});write(ROOT/'runs/frontier.json',{'rows':ans,'complete':len(ans)==42,'new_targets':new,'elapsed_seconds':time.monotonic()-start})
            print(json.dumps({'mechanism':name,'epsilon':eps,'status':s['status'],'water':s['expected_water'],'source':s['source']}),flush=True)
    for name,*_ in sources:
        v=[x['expected_water'] for x in ans if x['mechanism']==name and x['status']=='FEASIBLE'];assert np.all(np.diff(v)<=1e-8)
    write(ROOT/'runs/completion.json',{'status':'PASS','cells':42,'new_targets':new,'original_0.20_reruns':0,'monotonicity':'PASS','elapsed_seconds':time.monotonic()-start})
if __name__=='__main__':
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('frozen target cap')))
    try:main()
    except BaseException:
        import traceback
        (ROOT/'runs/failure.txt').write_text(traceback.format_exc());raise
