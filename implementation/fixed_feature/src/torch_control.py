"""Independent Torch AD of Gaussian heat value, used only on synthetic rows."""
import math
import torch
import numpy as np
from audit_adapter import heat,m


def heat_value(t,w,bi,trace):
    lo,hi,origin,c=heat.trace_polynomials(trace)
    lo,hi,origin,c=[torch.as_tensor(a,dtype=torch.float64) for a in (lo,hi,origin,c)]
    scale=m.SIGMA*torch.sqrt(1-t);value=torch.zeros((len(t),2),dtype=t.dtype)
    for n in (-1,0,1):
        for center,sign in ((w+2*n*m.LENGTH,1.),(2*m.WD-w-2*n*m.LENGTH,-1.)):
            a=(lo[None,:]-center[:,None])/scale[:,None];b=(hi[None,:]-center[:,None])/scale[:,None]
            j0=.5*(torch.erf(b/math.sqrt(2))-torch.erf(a/math.sqrt(2)))
            j0=torch.where(a>0,.5*(torch.erfc(a/math.sqrt(2))-torch.erfc(b/math.sqrt(2))),j0)
            j0=torch.where(b<0,.5*(torch.erfc(-b/math.sqrt(2))-torch.erfc(-a/math.sqrt(2))),j0)
            pa=torch.exp(-.5*a*a)/math.sqrt(2*math.pi);pb=torch.exp(-.5*b*b)/math.sqrt(2*math.pi)
            moments=[j0,pa-pb]
            for k in (2,3):moments.append(a**(k-1)*pa-b**(k-1)*pb+(k-1)*moments[k-2])
            offset=center[:,None]-origin[None,:]
            for j in range(4):
                integral=sum(math.comb(j,k)*scale[:,None]**k*offset**(j-k)*moments[k] for k in range(j+1))
                value=value+sign*torch.einsum('ns,nsj->nj',integral,c[bi,:,j,:])
    linear=torch.stack((torch.zeros_like(w),1-(w-m.WD)/m.LENGTH),dim=1)
    return value+linear


class IndependentTorchSlab:
    def __init__(self,slab,intensity):self.slab=slab;self.intensity=intensity

    def forward_day(self,day,states,budgets):
        t,w=states[:,0],states[:,1];bi=budgets.to(torch.long)
        alive=(w>m.WD)&(w<m.UPPER)&(t<1)
        values=states[:,0:1].repeat(1,2)*0
        failed=(w<=m.WD)
        values=values+torch.stack((torch.zeros_like(w),failed.to(w.dtype)),dim=1)
        if bool(alive.any()):
            tt,ww,bb=t[alive],w[alive],bi[alive]
            base=heat_value(tt,ww,bb,self.slab.trace)
            weights=torch.from_numpy(self.slab.weights);bias=torch.from_numpy(self.slab.bias)
            features=torch.tanh(torch.stack((2*tt-1,2*(ww-m.WD)/m.LENGTH-1),dim=1)@weights.T+bias)
            raw=torch.cat((torch.ones((len(tt),1),dtype=tt.dtype),features),dim=1)
            x=(ww-m.WD)/m.LENGTH;phi=((1-tt)*x*(1-x))[:,None]*raw
            coeff=torch.from_numpy(self.slab.coefficients)[bb]
            output=base+torch.einsum('ij,ijk->ik',phi,coeff)
            values=values.index_copy(0,torch.nonzero(alive).flatten(),output)
        return values
