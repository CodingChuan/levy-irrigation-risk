"""Isolated constant-drift Dirichlet lifting prototype, no training access."""
from pathlib import Path
import sys
import numpy as np
import diffusion_heat as heat
import manufactured_control as m
MU=.04-.08*m.WD

def _weighted(center,scale,lo,hi,origin,c,bi,rate):
 shifted=center+rate*scale**2
 v,d,dd=heat.gaussian_polynomial_jet(shifted,scale,lo,hi,origin,c,bi)
 fac=np.exp(rate*(center-m.WD)+.5*rate**2*scale**2)[:,None]
 return fac*v,fac*(d+rate*v),fac*(dd+2*rate*d+rate**2*v)

def base_jet(t,w,bi,trace=None,mu=MU):
 t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
 if mu==0:return heat.base_jet(t,w,bi,trace)
 if np.any((t<0)|(t>=1)|(w<=m.WD)|(w>=m.UPPER)):raise ValueError('open interior only')
 alpha=mu/m.SIGMA**2;beta=mu**2/(2*m.SIGMA**2);E=np.exp(-2*alpha*m.LENGTH);den=1-E
 lo,hi,origin,c=trace._prepared if hasattr(trace,'_prepared') else heat.trace_polynomials(trace);c=c.copy()
 c[:,:,0,1::2]+=(1-(origin-m.WD)/m.LENGTH)[None,:,None];c[:,:,1,1::2]-=1/m.LENGTH
 channels=c.shape[-1];const=np.zeros((c.shape[0],1,4,channels));const[:,:,0,1::2]=1.
 scale=m.SIGMA*np.sqrt(1-t);S=np.zeros((len(t),channels));Sw=S.copy();Sww=S.copy()
 for n in [-1,0,1]:
  for reflect in [False,True]:
   center=(2*m.WD-w if reflect else w)+2*n*m.LENGTH
   sign=-1. if reflect else 1.;chain=-1. if reflect else 1.
   jets=_weighted(center,scale,lo,hi,origin,c,bi,alpha)
   minus=_weighted(center,scale,np.array([m.WD]),np.array([m.UPPER]),np.array([m.WD]),const,bi,-alpha)
   plus=_weighted(center,scale,np.array([m.WD]),np.array([m.UPPER]),np.array([m.WD]),const,bi,alpha)
   J=[a-b/den+E*cc/den for a,b,cc in zip(jets,minus,plus)]
   S+=sign*J[0];Sw+=sign*chain*J[1];Sww+=sign*J[2]
 x=w-m.WD;ex=np.exp(-2*alpha*x);H=np.zeros((len(t),channels));Hw=H.copy();Hww=H.copy();H[:,1::2]=((ex-E)/den)[:,None];Hw[:,1::2]=(-2*alpha*ex/den)[:,None];Hww[:,1::2]=(4*alpha**2*ex/den)[:,None]
 pref=np.exp(-alpha*x-beta*(1-t))[:,None]
 v=H+pref*S;dw=Hw+pref*(Sw-alpha*S);dww=Hww+pref*(Sww-2*alpha*Sw+alpha**2*S)
 dt=-mu*dw-.5*m.SIGMA**2*dww
 return v,dt,dw,dww


BUDGETS=heat.BUDGETS
action=heat.action
policy_values=heat.policy_values
GRID_H=heat.GRID_H


def base_all_jet(t,w,trace=None):
    from types import SimpleNamespace
    if trace is not None and hasattr(trace,'_drift_packed'):packed=trace._drift_packed
    else:
        lo,hi,origin,c=heat.trace_polynomials(trace)
        packed=(lo,hi,origin,c.transpose(1,2,0,3).reshape(len(lo),4,10)[None,...])
        if trace is not None:trace._drift_packed=packed
    proxy=SimpleNamespace(_prepared=packed)
    return tuple(a.reshape(len(t),5,2) for a in base_jet(np.asarray(t),np.asarray(w),np.zeros(len(t),dtype=int),proxy))


def base_all_values(t,w,trace=None):
    t,w=np.asarray(t),np.asarray(w);out=np.zeros((len(t),5,2));lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper)
    out[lower,:,1]=1.
    if trace is not None and np.any(terminal):out[terminal]=np.stack([trace.jet(w[terminal],np.full(terminal.sum(),b))[0] for b in range(5)],axis=1)
    interior=~(lower|upper|terminal)
    if np.any(interior):out[interior]=base_all_jet(t[interior],w[interior],trace)[0]
    return out


def base_values(t,w,bi,trace=None):
    t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int);out=np.zeros((len(t),2));lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper)
    out[lower,1]=1.
    if trace is not None and np.any(terminal):out[terminal]=trace.jet(w[terminal],bi[terminal])[0]
    interior=~(lower|upper|terminal)
    if np.any(interior):out[interior]=base_jet(t[interior],w[interior],bi[interior],trace)[0]
    return out


class OpenTrace(heat.OpenTrace):
    @classmethod
    def from_callable(cls,policy,day,evaluate):
        owner=getattr(evaluate,'__self__',None)
        if owner is None or not hasattr(owner,'jet_all'):raise TypeError('analytic trace requires production jet_all')
        grid=np.arange(m.WD+GRID_H,m.UPPER-GRID_H/2,GRID_H);t=np.zeros_like(grid)
        no=owner.jet_all(t,grid);shifted={q:owner.jet_all(t,grid+q) for q in [.06,.08]};values=[];slopes=[]
        for bi in range(5):
            prob,q,nxt=action(policy,day,grid,np.full(len(grid),bi));idx=np.arange(len(grid))
            yes=np.where((q>.07)[:,None],shifted[.08][0][idx,nxt],shifted[.06][0][idx,nxt])
            dy=np.where((q>.07)[:,None],shifted[.08][2][idx,nxt],shifted[.06][2][idx,nxt])
            dp=np.where((bi>0)&(policy=='buffer')&(grid>m.WD+.06)&(grid<m.WD+.12),-1/.06,0.)
            value=(1-prob[:,None])*no[0][:,bi]+prob[:,None]*yes;value[:,0]+=prob*q
            slope=(1-prob[:,None])*no[2][:,bi]+prob[:,None]*dy+dp[:,None]*(yes-no[0][:,bi]);slope[:,0]+=dp*q
            values.append(value);slopes.append(slope)
        result=cls(policy,np.stack(values));result.slopes=np.stack(slopes);return result
