"""One fixed direction/Armijo rule shared by all repair entry points."""
import numpy as np
import torch

def flatten(items):return torch.cat([x.detach().reshape(-1).cpu() for x in items]).numpy().copy()
def assign(params,theta):
 pos=0
 with torch.no_grad():
  for p in params:n=p.numel();p.copy_(torch.tensor(theta[pos:pos+n],device=p.device,dtype=p.dtype).reshape(p.shape));pos+=n

def protect_proposal(params,theta,gradient,proposal,before_loss,closure):
 g=gradient.astype(float);d=proposal.astype(float);original_dot=float(g@d);norm=float(np.linalg.norm(d));replaced=False
 if original_dot>=0 and np.linalg.norm(g)>0 and norm>0:
  d=-g/(np.abs(g)+1e-8);d*=norm/np.linalg.norm(d);replaced=True
 dot=float(g@d);trials=[];accepted=False;after_loss=before_loss;alpha_used=0.
 for k in range(8):
  alpha=4.*.5**k;assign(params,(theta.astype(float)+alpha*d).astype(theta.dtype));value=closure();value=float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
  if not np.isfinite(value):raise FloatingPointError('nonfinite protected objective')
  target=before_loss+1e-4*alpha*dot;trials.append({'alpha':alpha,'loss':value,'armijo_target':target})
  if value<=target:accepted=True;after_loss=value;alpha_used=alpha;break
 if not accepted:assign(params,theta)
 return {'direction_replaced':replaced,'original_gradient_dot_proposal':original_dot,'guarded_gradient_dot_proposal':dot,'accepted':accepted,'blocked':not accepted,'alpha':alpha_used,'loss_before':before_loss,'loss_after':after_loss,'objective_evaluations':len(trials),'trials':trials}

def step(params,optimizer,before_loss,closure):
 theta=flatten(params);g=flatten([p.grad for p in params]);optimizer.step();proposal=flatten(params)-theta;result=protect_proposal(params,theta,g,proposal,before_loss,closure);return result,proposal
