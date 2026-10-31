from pathlib import Path
import json,hashlib,time,signal,sys
import numpy as np
from scipy.linalg import expm
import model,constrained
R=Path(__file__).resolve().parents[1];O=R.parents[3]/'mechanism_decision_extension_20261024_r1';OLD=O.parent/'control_reference_v1/runs/20260919T044919602355Z_green_38c0009b82'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def guard():
 for f in ['parent_binding.json','source_binding.json']:
  for p,h in read(R/f).items():assert sha(p)==h,p
def generator(h,lam,typ):
 g,Q,l,r=model.build_generator(1.55,h,lam if typ=='jump' else 0.)
 if typ=='mean_matched_diffusion':
  Q=Q.tolil();delta=lam*float(model.MARK_WEIGHTS@np.expm1(model.JUMP_SCALE*model.MARKS))*g
  for i in range(1,len(g)-1):
   l[i]-=delta[i]/(2*h);r[i]+=delta[i]/(2*h);Q[i,i-1]-=delta[i]/(2*h);Q[i,i+1]+=delta[i]/(2*h)
  Q=Q.tocsr()
 a=Q.toarray();off=a.copy();np.fill_diagonal(off,0);assert off.min()>=-1e-12 and abs(a.sum(1)).max()<1e-10 and not a[[0,-1]].any();return g,Q,l,r

def controls():
 guard();signal.alarm(30);out=[];red=[]
 for h in [.005,.0025]:
  for lam in [0.,.04,.08,.12]:
   for typ in ['jump','mean_matched_diffusion']:
    g,Q,l,r=generator(h,lam,typ);target=model.drift(g)+lam*float(model.MARK_WEIGHTS@np.expm1(model.JUMP_SCALE*model.MARKS))*g
    inside=(g*np.exp(-.18)>g[0])&(g*np.exp(.12)<g[-1]);e=float(abs((Q@g-target)[inside]).max());assert e<1e-10
    if typ=='mean_matched_diffusion':
     eq=float(abs((Q@(g*g)-(2*g*target+model.DIFFUSION**2))[1:-1]).max());assert eq<1e-10
     if lam>0:
      _,oldq,_,_=model.build_generator(1.55,h,0.);bad=float(abs((oldq@g-target)[inside]).max());assert bad>1e-3;red.append({'h':h,'lambda':lam,'unmatched_error':bad})
    out.append({'h':h,'lambda':lam,'mechanism':typ,'linear_error':e})
 # Existing endpoints: exact generator equality; no solve or gate repeated.
 old=read(OLD/'matrix_scalar.json')['rows'];matched=read(O/'runs/matrix.json')['rows'];reuse=[]
 for h in [.005,.0025]:
  for typ in ['jump','mean_matched_diffusion']:
   for lam in [0.,.12]:
    if typ=='mean_matched_diffusion' and lam==.12:
     i=next(i for i,z in enumerate(matched) if z['upper']==1.55 and z['h']==h);z=matched[i];path=O/'runs'/f'matched_{i}/model.npz'
    else:
     z=next(z for z in old if z['upper']==1.55 and z['h']==h and z['intensity']==lam);key=z['row_key'].replace('.','p').replace('-','m').replace(' ','_');path=OLD/f'row_{key}__model.npz'
    with np.load(path) as q:assert np.array_equal(generator(h,lam,typ)[1].toarray(),q['Q'])
    assert z['gate_summary']['pass'];reuse.append({'h':h,'lambda':lam,'mechanism':typ,'summary':z,'model_path':str(path),'sha256':sha(path)})
 write(R/'runs/controls.json',{'passed':True,'matrix':out,'RED':red,'reused_endpoints':reuse,'optimizer_calls':0});signal.alarm(0);print('controls16 passed;6 RED;8 endpoint cells reused')
def solve(index):
 guard();assert read(R/'runs/controls.json')['passed'];assign=[(h,lam,typ) for h in [.005,.0025] for lam in [.04,.08] for typ in ['jump','mean_matched_diffusion']];h,lam,typ=assign[index];out=R/'runs'/f'new_{index}';out.mkdir();signal.alarm(60);t=time.monotonic();g,Q,l,r=generator(h,lam,typ);P=expm(Q.toarray());assert P.min()>=-1e-12 and abs(P.sum(1)-1).max()<1e-10;row=model.ModelRow(1.55,h,lam if typ=='jump' else 0.,g,Q,P,l,r);np.savez_compressed(out/'model.npz',grid=g,Q=Q.toarray(),P=P)
 def callback(rec):
  s=rec.solution;path=out/f'trial_{len(list(out.glob("trial_*.npz"))):02d}.npz';np.savez_compressed(path,C=s.C,R=s.R,V=s.V,actions=s.actions,successor_budget=s.successor_budget);write(path.with_suffix('.json'),{'summary':rec.summary,'file_sha':sha(path)})
 result=constrained.solve_row(row,on_trial=callback);summary=constrained.row_summary(result)
 if result.status=='FEASIBLE':assert summary['gate_summary']['pass'],summary
 else:assert result.status=='INFEASIBLE' and result.initial_risk_min>.2+1e-10
 summary.update({'target_intensity':lam,'mechanism':typ,'elapsed_seconds':time.monotonic()-t,'new_index':index});write(out/'result.json',summary);guard();signal.alarm(0);print(json.dumps({k:summary[k] for k in ['new_index','mechanism','target_intensity','h','initial_risk_min','mixture_cost','elapsed_seconds']}))
def finalize():
 guard();c=read(R/'runs/controls.json');rows=[]
 for z in c['reused_endpoints']:
  row=dict(z['summary']);row.update(target_intensity=z['lambda'],mechanism=z['mechanism'],reused=True);rows.append(row)
 for i in range(8):
  z=read(R/'runs'/f'new_{i}/result.json');z['reused']=False;rows.append(z)
 assert len(rows)==16 and len({(z['h'],z['target_intensity'],z['mechanism']) for z in rows})==16
 grid=[]
 for lam in [0.,.04,.08,.12]:
  for typ in ['jump','mean_matched_diffusion']:
   a,b=[next(z for z in rows if z['h']==h and z['target_intensity']==lam and z['mechanism']==typ) for h in [.005,.0025]];dr=abs(a['initial_risk_min']-b['initial_risk_min']);dc=None if a['mixture_cost'] is None or b['mixture_cost'] is None else abs(a['mixture_cost']-b['mixture_cost']);grid.append({'lambda':lam,'mechanism':typ,'Rmin_change':dr,'cost_change':dc,'within_frozen_grid_tolerance':dr<=.001 and (dc is None or dc<=.001)})
 elapsed=sum(z['elapsed_seconds'] for z in rows if not z['reused']);assert elapsed<480
 write(R/'runs/result.json',{'completed':True,'reported_cells':16,'new_solves':8,'reused_cells':8,'rows':rows,'grid_checks':grid,'matrix_compute_seconds':elapsed,'no_neural_training':True,'old_gate_calls_repeated':0});manifest={str(p):sha(p) for p in R.rglob('*') if p.is_file() and p.name not in ['artifact_manifest.json'] and '__pycache__' not in p.parts};write(R/'artifact_manifest.json',manifest);print(json.dumps({'completed':True,'new_solves':8,'grid_pass':all(g['within_frozen_grid_tolerance'] for g in grid),'seconds':elapsed}))
if __name__=='__main__':
 if sys.argv[1]=='controls':controls()
 elif sys.argv[1]=='finalize':finalize()
 else:solve(int(sys.argv[1]))
