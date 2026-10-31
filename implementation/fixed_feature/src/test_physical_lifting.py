import sys
import unittest
import numpy as np
import manufactured_control as m
import physical_lifting as p


def synthetic_post(t,w,bi):
    x=(w-m.WD)/m.LENGTH
    value=np.column_stack((.03*x*(1-x)*(np.asarray(bi)>0),(1-x)**2))
    value[w<=m.WD]=[0,1];value[w>=m.UPPER]=0
    return value


class PhysicalControls(unittest.TestCase):
    def test_interior_api_rejects_terminal_boundary(self):
        for t,w in [(1.,.8),(.5,m.WD),(.5,m.UPPER)]:
            with self.assertRaises(ValueError):p.physical_qh_jet(np.array([t]),np.array([w]))

    def test_heat_and_survival_lifting_jets(self):
        t=np.array([.1,.31,.7,.95]);w=np.array([.36,.41,1.48,1.52]);h=1e-6
        q,s=p.physical_qh_jet(t,w)
        for index,jets in enumerate((q,s)):
            f=lambda tt,ww:p.physical_qh_jet(tt,ww)[index][0]
            np.testing.assert_allclose(jets[1],(f(t+h,w)-f(t-h,w))/(2*h),atol=1e-7,rtol=1e-6)
            np.testing.assert_allclose(jets[2],(f(t,w+h)-f(t,w-h))/(2*h),atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(jets[3],(f(t,w+h)-2*f(t,w)+f(t,w-h))/h**2,atol=.002,rtol=1e-4)

    def test_trace_and_base_jets(self):
        trace=p.OpenTrace.from_callable('buffer',2,synthetic_post)
        t=np.array([.11,.31,.71,.95]);w=np.array([.3821,.4421,.8021,1.5021]);bi=np.array([0,1,3,4]);h=2e-6
        value,dt,dw,dww=p.base_jet(t,w,bi,trace)
        f=lambda tt,ww:p.base_values(tt,ww,bi,trace)
        np.testing.assert_allclose(dt,(f(t+h,w)-f(t-h,w))/(2*h),atol=2e-7,rtol=1e-6)
        np.testing.assert_allclose(dw,(f(t,w+h)-f(t,w-h))/(2*h),atol=2e-6,rtol=1e-6)
        np.testing.assert_allclose(dww,(f(t,w+h)-2*f(t,w)+f(t,w-h))/h**2,atol=1e-3,rtol=1e-4)

    def test_terminal_and_boundaries_are_exact(self):
        trace=p.OpenTrace.from_callable('schedule',2,synthetic_post)
        w=np.array([m.WD,.4,.8,m.UPPER]);bi=np.arange(4)
        np.testing.assert_array_equal(p.base_values(np.ones(4),w,bi,trace),trace.jet(w,bi)[0])
        np.testing.assert_array_equal(p.base_values(np.ones(4),w,bi,None),np.array([[0,1],[0,0],[0,0],[0,0]]))
        w=np.array([m.WD,m.UPPER]);bi=np.array([4,4])
        for t in (0.,.5,.9999,1.):np.testing.assert_array_equal(p.base_values(np.full(2,t),w,bi,trace),np.array([[0,1],[0,0]]))

    def test_policy_budget_and_channel_mixture_independent(self):
        for policy in ('zero','schedule','buffer'):
            for day in range(20):
                for b in range(5):
                    w=np.array([m.WD,.38,.41,.44,.47,.8,1.53,m.UPPER]);bi=np.full(len(w),b)
                    prob,q,nxt=p.action(policy,day,w,bi)
                    np.testing.assert_allclose(p.BUDGETS[bi]-q,p.BUDGETS[nxt],atol=1e-15)
                    if policy=='zero':expected=np.zeros(len(w))
                    elif policy=='schedule':expected=np.full(len(w),float(day<=3 and b>0))
                    else:expected=np.maximum(0,np.minimum(1,(.47-w)/.06)) if b>0 else np.zeros(len(w))
                    expected[(w<=m.WD)|(w>=m.UPPER)]=0
                    np.testing.assert_allclose(prob,expected,atol=1e-14)
                    no=synthetic_post(w*0,w,bi);yes=synthetic_post(w*0,w+q,nxt)
                    ref=(1-expected[:,None])*no+expected[:,None]*yes;ref[:,0]+=expected*q
                    np.testing.assert_allclose(p.policy_values(synthetic_post,policy,day,w,bi),ref,atol=1e-14)


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PhysicalControls))
    sys.exit(0 if result.wasSuccessful() else 1)
