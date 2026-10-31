from pathlib import Path
import sys,time,json,tempfile,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[1]
import heat_lifting as h
import solve_day as s
import manufactured_control as m

def exact(t,w,bi):
 x=w-.35;a=.012/.06**2;k=np.pi/1.2;E=np.exp(-2*a*1.2);amp=(bi+1)/5
 ev=np.exp(-a*x-(.5*.06**2*k*k+.012**2/(2*.06**2))*(1-t))*np.sin(k*x)
 H=(np.exp(-2*a*x)-E)/(1-E)
 v=np.stack((.02*amp*ev,H+.1*amp*ev),axis=1)
 dw=np.stack((.02*amp*np.exp(-a*x-(.5*.06**2*k*k+.012**2/(2*.06**2))*(1-t))*(k*np.cos(k*x)-a*np.sin(k*x)),-2*a*np.exp(-2*a*x)/(1-E)+.1*amp*np.exp(-a*x-(.5*.06**2*k*k+.012**2/(2*.06**2))*(1-t))*(k*np.cos(k*x)-a*np.sin(k*x))),axis=1)
 outside=(w<=.35)|(w>=1.55);v[w<=.35]=[0,1];v[w>=1.55]=0.;dw[outside]=0
 return v,dw
class Toy:
 def jet_all(self,t,w):
  v=[];dw=[]
  for b in range(5):
   a,c=exact(t,w,np.full(len(w),b));v.append(a);dw.append(c)
  v=np.stack(v,1);dw=np.stack(dw,1);return v,np.zeros_like(v),dw,np.zeros_like(v)
 def values(self,t,w,bi):return exact(t,w,bi)[0]

def main():
 start=time.monotonic();torch.set_num_threads(1);toy=Toy();records=[]
 for policy in ['zero','schedule','buffer']:
  for day in range(20):
   tr=h.OpenTrace.from_callable(policy,day,toy.values);grid=tr.grid
   for bi in range(5):
    budgets=np.full(len(grid),bi);v=h.policy_values(toy.values,policy,day,grid,budgets)
    value_error=float(np.max(np.abs(tr.values[bi]-v)));assert value_error<1e-13
    mask=np.ones(len(grid),dtype=bool)
    for kink in [.41,.47,1.55-.06,1.55-.08]:mask&=np.abs(grid-kink)>1e-6
    step=1e-6;fd=(h.policy_values(toy.values,policy,day,grid[mask]+step,budgets[mask])-h.policy_values(toy.values,policy,day,grid[mask]-step,budgets[mask]))/(2*step)
    deriv_error=float(np.max(np.abs(tr.slopes[bi,mask]-fd)));assert deriv_error<1e-7,deriv_error
    records.append({'policy':policy,'day':day,'budget':bi,'value_error':value_error,'slope_error':deriv_error})
 # Production Slab jet_all, scalar jets, values and checkpoint round trip.
 trace=h.OpenTrace.from_callable('zero',1,toy.values);coef=np.zeros((5,513,2));slab=s.Slab(20260928,coef,trace);t=np.array([0.,.5,.95,.999]);w=np.array([.3525,.6,1.,1.5475]);alljet=slab.jet_all(t,w)
 for b in range(5):
  jj=slab.jet(t,w,np.full(len(t),b));assert max(float(np.max(np.abs(a[:,b]-c))) for a,c in zip(alljet,jj))<1e-10
 assert float(np.max(np.abs(slab.values_all(t,w)-alljet[0])))<1e-10
 boundary=slab.values(np.array([.2,.4]),np.array([.35,1.55]),np.array([2,3]));assert np.array_equal(boundary,np.array([[0.,1.],[0.,0.]]))
 terminal=slab.values(np.ones(4),w,np.arange(4));assert np.array_equal(terminal,trace.jet(w,np.arange(4))[0])
 out=R/'controls/checkpoint_roundtrip.npz';meta={'policy':'zero','seed':20260928,'source_manifest_sha256':'manufactured_control','day':1,'intensity':0.}
 s.save_checkpoint(out,meta,coef,trace,trace);_,restored,pre=s.load_checkpoint(out,'zero',20260928,'manufactured_control');assert np.array_equal(trace.slopes,pre.slopes);assert np.array_equal(trace.slopes,restored.trace.slopes)
 assert np.array_equal(slab.values_all(t,w),restored.values_all(t,w))
 # Analytic manufactured trace slopes remove the previously frozen interpolation defect.
 grid=trace.grid;values=[];slopes=[]
 for b in range(5):
  v,dw=exact(np.ones(len(grid)),grid,np.full(len(grid),b));values.append(v);slopes.append(dw)
 tr=h.OpenTrace('zero',np.stack(values));tr.slopes=np.stack(slopes)
 inp=np.array([[1-delta,.35+dist,b] for delta in [1.,.5,.05,.01,.001] for dist in [.0025,.005,.02,.25,.6,1.1975] for b in range(5)])
 packed=h.base_all_jet(inp[:,0],inp[:,1],tr)
 scalar=h.base_jet(inp[:,0],inp[:,1],inp[:,2].astype(int),tr)
 packed_err=max(float(np.max(np.abs(a[np.arange(len(inp)),inp[:,2].astype(int)]-c))) for a,c in zip(packed,scalar));assert packed_err<1e-8,packed_err
 # Independent torch derivatives for exact eigenfunction.
 tt=torch.tensor(inp[:,:2],dtype=torch.float64,requires_grad=True);bb=torch.from_numpy(inp[:,2]);x=tt[:,1]-.35;a=.012/.06**2;k=np.pi/1.2;E=np.exp(-2*a*1.2);amp=(bb+1)/5;ev=torch.exp(-a*x-(.5*.06**2*k*k+.012**2/(2*.06**2))*(1-tt[:,0]))*torch.sin(k*x);H=(torch.exp(-2*a*x)-E)/(1-E);y=torch.stack((.02*amp*ev,H+.1*amp*ev),1);grad=[];ww=[]
 for c in range(2):
  g=torch.autograd.grad(y[:,c].sum(),tt,create_graph=True,retain_graph=True)[0];grad.append(g);ww.append(torch.autograd.grad(g[:,1].sum(),tt,retain_graph=True)[0][:,1])
 refs=[y.detach().numpy(),torch.stack([g[:,0] for g in grad],1).detach().numpy(),torch.stack([g[:,1] for g in grad],1).detach().numpy(),torch.stack(ww,1).detach().numpy()];errors=[float(np.max(np.abs(a-b))) for a,b in zip(scalar,refs)];assert max(errors)<1e-7,errors
 # RED: omitted drift with the same exact terminal trace, zero-drift physics control retained.
 old=h.heat.base_jet(inp[:,0],inp[:,1],inp[:,2].astype(int),tr);old_full=old[1]+.012*old[2]+.5*.06**2*old[3];old_zero=old[1]+.5*.06**2*old[3];assert np.max(np.abs(old_full))>.01 and np.max(np.abs(old_zero))<1e-10
 (R/'controls/trace_matrix.json').write_text(json.dumps(records,indent=2)+'\n')
 result={'state':'preflight_controls_pass','trace_cases':len(records),'grid_values_and_slopes_tested':len(records)*len(grid)*2,'production_roundtrip':True,'production_terminal_boundary':True,'scalar_packed_max_difference':packed_err,'manufactured_errors':errors,'old_drift_omission_RED':float(np.max(np.abs(old_full))),'old_zero_drift_control':float(np.max(np.abs(old_zero))),'elapsed_seconds':time.monotonic()-start,'fitting_calls':0,'optimizer_calls':0,'gate_calls':0}
 (R/'controls/preflight_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
