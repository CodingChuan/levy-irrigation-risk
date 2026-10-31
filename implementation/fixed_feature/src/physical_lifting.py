"""Continuous physical boundary lifting for a fixed-feature PIDE solver."""
from __future__ import annotations
import numpy as np
from scipy.special import erf, erfc
import manufactured_control as m

BUDGETS=np.array([0.,.06,.14,.22,.30])
GRID_H=.000625
SQRT_PI=np.sqrt(np.pi)


def physical_qh_jet(t,w):
    t,w=np.asarray(t),np.asarray(w)
    if np.any((t<0)|(t>=1)) or np.any((w<=m.WD)|(w>=m.UPPER)):
        raise ValueError('interior heat lifting requires 0<=t<1 and WD<w<UPPER')
    delta=1-t;denom=m.SIGMA*np.sqrt(2*delta)
    a=(w-m.WD)/denom;b=(m.UPPER-w)/denom;u=m.LENGTH/denom
    ea,eb,eu=np.exp(-a*a),np.exp(-b*b),np.exp(-u*u)
    qa,qu=erfc(a),erfc(u);den=1-qu
    qat=-a*ea/(SQRT_PI*delta);qut=-u*eu/(SQRT_PI*delta)
    q=(qa-qu)/den
    qt=(qat-qut)/den+(qa-qu)*qut/den**2
    qw=-2*ea/(SQRT_PI*denom*den)
    qww=4*a*ea/(SQRT_PI*denom**2*den)
    fa,fb=erf(a),erf(b)
    fat=a*ea/(SQRT_PI*delta);fbt=b*eb/(SQRT_PI*delta)
    faw=2*ea/(SQRT_PI*denom);fbw=-2*eb/(SQRT_PI*denom)
    faww=-4*a*ea/(SQRT_PI*denom**2);fbww=-4*b*eb/(SQRT_PI*denom**2)
    h=fa*fb;ht=fat*fb+fa*fbt;hw=faw*fb+fa*fbw
    hww=faww*fb+2*faw*fbw+fa*fbww
    return (q,qt,qw,qww),(h,ht,hw,hww)


def action(policy,day,w,bi):
    w=np.asarray(w);bi=np.asarray(bi,dtype=int)
    beta=BUDGETS[bi];amount=np.minimum(.08,beta)
    if policy=='zero':p=np.zeros_like(w)
    elif policy=='schedule':p=np.full_like(w,float(day<=3))
    elif policy=='buffer':p=np.clip((m.WD+.12-w)/.06,0.,1.)
    else:raise ValueError(policy)
    p=np.where((beta>0)&(w>m.WD)&(w<m.UPPER),p,0.)
    amount=np.where(p>0,amount,0.)
    nxt=np.where(amount>0,bi-1,bi)
    return p,amount,nxt


def policy_values(evaluate,policy,day,w,bi):
    p,q,nxt=action(policy,day,w,bi)
    no=evaluate(np.zeros_like(w),w,bi)
    yes=evaluate(np.zeros_like(w),w+q,nxt)
    val=(1-p[:,None])*no+p[:,None]*yes
    val[:,0]+=p*q
    return val


class OpenTrace:
    def __init__(self,policy,values):
        self.policy=policy
        self.grid=np.arange(m.WD+GRID_H,m.UPPER-GRID_H/2,GRID_H)
        self.values=np.asarray(values,dtype=float)
        if self.values.shape!=(5,len(self.grid),2) or not np.isfinite(self.values).all():
            raise ValueError('trace shape or finite contract')
        self.slopes=np.empty_like(self.values)
        self.slopes[:,0]=(-3*self.values[:,0]+4*self.values[:,1]-self.values[:,2])/(2*GRID_H)
        self.slopes[:,-1]=(3*self.values[:,-1]-4*self.values[:,-2]+self.values[:,-3])/(2*GRID_H)
        self.slopes[:,1:-1]=(self.values[:,2:]-self.values[:,:-2])/(2*GRID_H)

    @classmethod
    def from_callable(cls,policy,day,evaluate):
        grid=np.arange(m.WD+GRID_H,m.UPPER-GRID_H/2,GRID_H)
        values=np.stack([policy_values(evaluate,policy,day,grid,np.full(len(grid),bi)) for bi in range(5)])
        return cls(policy,values)

    def jet(self,w,bi):
        w=np.asarray(w);bi=np.asarray(bi,dtype=int)
        idx=np.floor((w-self.grid[0])/GRID_H).astype(int)
        idx=np.clip(idx,0,len(self.grid)-2)
        x=(w-self.grid[idx])/GRID_H
        y0=self.values[bi,idx];y1=self.values[bi,idx+1]
        s0=self.slopes[bi,idx].copy();s1=self.slopes[bi,idx+1].copy()
        if self.policy in ('buffer','schedule'):
            kinks=(m.WD+.06,m.WD+.12) if self.policy=='buffer' else (m.UPPER-.06,m.UPPER-.08)
            for kink in kinks:
                k=int(round((kink-self.grid[0])/GRID_H))
                left=(self.values[:,k]-self.values[:,k-1])/GRID_H
                right=(self.values[:,k+1]-self.values[:,k])/GRID_H
                s0=np.where(((bi>0)&(idx==k))[:,None],right[bi],s0)
                s1=np.where(((bi>0)&(idx+1==k))[:,None],left[bi],s1)
        x=x[:,None];d=GRID_H
        v=(2*x**3-3*x*x+1)*y0+(x**3-2*x*x+x)*d*s0+(-2*x**3+3*x*x)*y1+(x**3-x*x)*d*s1
        dw=((6*x*x-6*x)*y0+(3*x*x-4*x+1)*d*s0+(-6*x*x+6*x)*y1+(3*x*x-2*x)*d*s1)/d
        dww=((12*x-6)*y0+(6*x-4)*d*s0+(-12*x+6)*y1+(6*x-2)*d*s1)/d**2
        lower=w<=m.WD;upper=w>=m.UPPER
        v[lower]=[0.,1.];v[upper]=0.
        dw[lower|upper]=0.;dww[lower|upper]=0.
        return v,dw,dww


def base_jet(t,w,bi,trace=None):
    q,h=physical_qh_jet(t,w)
    if trace is None:
        tv=np.zeros((len(t),2));tw=tv.copy();tww=tv.copy()
    else:tv,tw,tww=trace.jet(w,bi)
    value=h[0][:,None]*tv
    dt=h[1][:,None]*tv
    dw=h[2][:,None]*tv+h[0][:,None]*tw
    dww=h[3][:,None]*tv+2*h[2][:,None]*tw+h[0][:,None]*tww
    for a,b in zip((value,dt,dw,dww),q):a[:,1]+=b
    return value,dt,dw,dww


def base_values(t,w,bi,trace=None):
    t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
    values=np.zeros((len(t),2))
    lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper)
    values[lower,1]=1.
    if trace is not None and np.any(terminal):values[terminal]=trace.jet(w[terminal],bi[terminal])[0]
    interior=~(lower|upper|terminal)
    if np.any(interior):values[interior]=base_jet(t[interior],w[interior],bi[interior],trace)[0]
    return values
