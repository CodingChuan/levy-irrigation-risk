"""Absorbing interval heat lifting by exact Gaussian/cubic moments."""
from __future__ import annotations
import math
from types import SimpleNamespace
import numpy as np
from scipy.special import erf,erfc
import manufactured_control as m
import physical_lifting as old

BUDGETS=old.BUDGETS
action=old.action
policy_values=old.policy_values
GRID_H=old.GRID_H


class OpenTrace(old.OpenTrace):
    @classmethod
    def from_callable(cls,policy,day,evaluate):
        owner=getattr(evaluate,'__self__',None)
        if owner is None or not hasattr(owner,'values_all'):
            return super().from_callable(policy,day,evaluate)
        grid=np.arange(m.WD+GRID_H,m.UPPER-GRID_H/2,GRID_H)
        no=owner.values_all(np.zeros_like(grid),grid)
        shifted={.06:owner.values_all(np.zeros_like(grid),grid+.06),.08:owner.values_all(np.zeros_like(grid),grid+.08)}
        rows=[]
        for bi in range(5):
            prob,q,nxt=action(policy,day,grid,np.full(len(grid),bi))
            yes=np.where((q>.07)[:,None],shifted[.08][np.arange(len(grid)),nxt],shifted[.06][np.arange(len(grid)),nxt])
            val=(1-prob[:,None])*no[:,bi]+prob[:,None]*yes
            val[:,0]+=prob*q;rows.append(val)
        return cls(policy,np.stack(rows))


def trace_polynomials(trace):
    if trace is None:
        lo=np.array([m.WD]);hi=np.array([m.UPPER]);origin=lo.copy()
        c=np.zeros((5,1,4,2));c[:,:,0,1]=-1.;c[:,:,1,1]=1/m.LENGTH
        return lo,hi,origin,c
    values=trace.values;grid=trace.grid;h=GRID_H
    s0=trace.slopes[:,:-1].copy();s1=trace.slopes[:,1:].copy()
    kinks=(m.WD+.06,m.WD+.12) if trace.policy=='buffer' else ((m.UPPER-.06,m.UPPER-.08) if trace.policy=='schedule' else ())
    for kink in kinks:
        k=int(round((kink-grid[0])/h))
        s1[1:,k-1]=(values[1:,k]-values[1:,k-1])/h
        s0[1:,k]=(values[1:,k+1]-values[1:,k])/h
    y0=values[:,:-1];y1=values[:,1:]
    coeff=np.stack((y0,s0,(3*(y1-y0)/h-2*s0-s1)/h,(2*(y0-y1)/h+s0+s1)/h**2),axis=2)
    coeff=np.concatenate((coeff[:,:1],coeff,coeff[:,-1:]),axis=1)
    lo=np.concatenate(([m.WD],grid[:-1],grid[-1:]));hi=np.concatenate((grid[:1],grid[1:],[m.UPPER]))
    origin=np.concatenate((grid[:1],grid[:-1],grid[-2:-1]))
    coeff[:,:,0,1]-=1-(origin-m.WD)/m.LENGTH
    coeff[:,:,1,1]+=1/m.LENGTH
    return lo,hi,origin,coeff


def gaussian_moments(a,b):
    cdf=.5*(erf(b/np.sqrt(2))-erf(a/np.sqrt(2)))
    pos=a>0;neg=b<0
    cdf[pos]=.5*(erfc(a[pos]/np.sqrt(2))-erfc(b[pos]/np.sqrt(2)))
    cdf[neg]=.5*(erfc(-b[neg]/np.sqrt(2))-erfc(-a[neg]/np.sqrt(2)))
    pa=np.exp(-.5*a*a)/np.sqrt(2*np.pi);pb=np.exp(-.5*b*b)/np.sqrt(2*np.pi)
    moments=[cdf,pa-pb]
    for k in range(2,6):moments.append(a**(k-1)*pa-b**(k-1)*pb+(k-1)*moments[k-2])
    return moments


def gaussian_polynomial_jet(center,scale,lo,hi,origin,coeff,bi):
    a=(lo[None,:]-center[:,None])/scale[:,None]
    b=(hi[None,:]-center[:,None])/scale[:,None]
    moments=gaussian_moments(a,b);offset=center[:,None]-origin[None,:]
    v=np.zeros((len(center),coeff.shape[-1]));d=v.copy();dd=v.copy()
    for j in range(4):
        z=np.zeros_like(a);dz=z.copy();ddz=z.copy()
        for k in range(j+1):
            factor=math.comb(j,k)*scale[:,None]**k*offset**(j-k)
            z+=factor*moments[k]
            dz+=factor*moments[k+1]/scale[:,None]
            ddz+=factor*(moments[k+2]-moments[k])/scale[:,None]**2
        c=coeff[bi,:,j,:]
        v+=np.einsum('ns,nsj->nj',z,c)
        d+=np.einsum('ns,nsj->nj',dz,c)
        dd+=np.einsum('ns,nsj->nj',ddz,c)
    return v,d,dd


def _base_jet(t,w,bi,trace=None):
    t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
    if np.any((t<0)|(t>=1)) or np.any((w<=m.WD)|(w>=m.UPPER)):
        raise ValueError('heat interior requires 0<=t<1 and WD<w<UPPER')
    if trace is not None and hasattr(trace,'_heat_polynomials'):polys=trace._heat_polynomials
    else:
        polys=trace_polynomials(trace)
        if trace is not None:trace._heat_polynomials=polys
    lo,hi,origin,c=polys
    value=np.zeros((len(t),c.shape[-1]));dw=value.copy();dww=value.copy()
    for start in range(0,len(t),64):
        ids=slice(start,start+64);tt=t[ids];ww=w[ids];bb=bi[ids]
        scale=m.SIGMA*np.sqrt(1-tt)
        a,da,dda=gaussian_polynomial_jet(ww,scale,lo,hi,origin,c,bb)
        lower,dl,ddl=gaussian_polynomial_jet(2*m.WD-ww,scale,lo,hi,origin,c,bb)
        upper,du,ddu=gaussian_polynomial_jet(2*m.UPPER-ww,scale,lo,hi,origin,c,bb)
        value[ids]=a-lower-upper;dw[ids]=da+dl+du;dww[ids]=dda-ddl-ddu
        # Complete n=-1,0,1 image sum, including remote images.
        for n in (-1,1):
            extra,de,dde=gaussian_polynomial_jet(ww+2*n*m.LENGTH,scale,lo,hi,origin,c,bb)
            value[ids]+=extra;dw[ids]+=de;dww[ids]+=dde
        extra,de,dde=gaussian_polynomial_jet(2*m.WD-ww-2*m.LENGTH,scale,lo,hi,origin,c,bb)
        value[ids]-=extra;dw[ids]+=de;dww[ids]-=dde
    dt=-.5*m.SIGMA**2*dww
    value[:,1::2]+=(1-(w-m.WD)/m.LENGTH)[:,None];dw[:,1::2]-=1/m.LENGTH
    return value,dt,dw,dww


def base_all_jet(t,w,trace=None):
    if trace is not None and hasattr(trace,'_packed_heat_polynomials'):packed=trace._packed_heat_polynomials
    else:
        lo,hi,origin,c=trace_polynomials(trace)
        packed=(lo,hi,origin,c.transpose(1,2,0,3).reshape(len(lo),4,10)[None,...])
        if trace is not None:trace._packed_heat_polynomials=packed
    proxy=SimpleNamespace(_heat_polynomials=packed)
    return tuple(a.reshape(len(t),5,2) for a in _base_jet(np.asarray(t),np.asarray(w),np.zeros(len(t),dtype=int),proxy))


def base_jet(t,w,bi,trace=None):
    t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
    coordinates,inverse=np.unique(np.column_stack((t,w)),axis=0,return_inverse=True)
    if len(coordinates)*2<=len(t):
        all_jets=base_all_jet(coordinates[:,0],coordinates[:,1],trace)
        return tuple(a[inverse,bi] for a in all_jets)
    return _base_jet(t,w,bi,trace)


def base_all_values(t,w,trace=None):
    t,w=np.asarray(t),np.asarray(w)
    out=np.zeros((len(t),5,2));lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper)
    out[lower,:,1]=1.
    if trace is not None and np.any(terminal):
        out[terminal]=np.stack([trace.jet(w[terminal],np.full(terminal.sum(),b))[0] for b in range(5)],axis=1)
    interior=~(lower|upper|terminal)
    if np.any(interior):out[interior]=base_all_jet(t[interior],w[interior],trace)[0]
    return out


def base_values(t,w,bi,trace=None):
    t,w,bi=np.asarray(t),np.asarray(w),np.asarray(bi,dtype=int)
    out=np.zeros((len(t),2));lower=w<=m.WD;upper=w>=m.UPPER;terminal=(t==1)&~(lower|upper)
    out[lower,1]=1.
    if trace is not None and np.any(terminal):out[terminal]=trace.jet(w[terminal],bi[terminal])[0]
    interior=~(lower|upper|terminal)
    if np.any(interior):out[interior]=base_jet(t[interior],w[interior],bi[interior],trace)[0]
    return out
