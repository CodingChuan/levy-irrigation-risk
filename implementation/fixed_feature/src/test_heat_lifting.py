import sys
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import erf
import manufactured_control as m
import heat_lifting as h
from test_physical_lifting import synthetic_post


class HeatControls(unittest.TestCase):
    def test_moments_independent_quadrature(self):
        a=np.array([[-3.,.1,2.]]);b=np.array([[-.2,1.4,5.]])
        moments=h.gaussian_moments(a,b)
        for k in range(6):
            exact=[quad(lambda z:z**k*np.exp(-z*z/2)/np.sqrt(2*np.pi),aa,bb,epsabs=1e-13)[0] for aa,bb in zip(a.ravel(),b.ravel())]
            np.testing.assert_allclose(moments[k].ravel(),exact,atol=1e-13)

    def test_polynomial_gaussian_integral_and_jets(self):
        origin=np.array([.35]);lo=np.array([.35]);hi=np.array([1.55]);c=np.zeros((5,1,4,2));c[:,:,0]=[.13,.7];c[:,:,1]=[.24,-.4];c[:,:,2]=[.03,.2];c[:,:,3]=[-.02,.1]
        center=np.array([.38,.8,1.5]);scale=np.array([.04,.03,.05]);bi=np.array([0,2,4]);v,d,dd=h.gaussian_polynomial_jet(center,scale,lo,hi,origin,c,bi)
        for i,(mu,s) in enumerate(zip(center,scale)):
            for j in range(2):
                poly=lambda y:sum(c[bi[i],0,k,j]*(y-.35)**k for k in range(4))
                exact=quad(lambda y:poly(y)*np.exp(-.5*((y-mu)/s)**2)/(s*np.sqrt(2*np.pi)),.35,1.55,epsabs=1e-12)[0]
                self.assertAlmostEqual(v[i,j],exact,places=11)
        step=2e-6;f=lambda mu:h.gaussian_polynomial_jet(mu,scale,lo,hi,origin,c,bi)[0]
        np.testing.assert_allclose(d,(f(center+step)-f(center-step))/(2*step),atol=2e-7)
        np.testing.assert_allclose(dd,(f(center+step)-2*v+f(center-step))/step**2,atol=1e-3)

    def test_heat_lifting_time_and_water_derivatives(self):
        trace=h.OpenTrace.from_callable('buffer',2,synthetic_post)
        t=np.array([.11,.71]);w=np.array([.4031,.8121]);bi=np.array([4,2]);step=3e-6
        v,dt,dw,dww=h.base_jet(t,w,bi,trace)
        f=lambda tt,ww:h.base_values(tt,ww,bi,trace)
        np.testing.assert_allclose(dt,(f(t+step,w)-f(t-step,w))/(2*step),atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(dw,(f(t,w+step)-f(t,w-step))/(2*step),atol=2e-6,rtol=1e-5)
        np.testing.assert_allclose(dww,(f(t,w+step)-2*v+f(t,w-step))/step**2,atol=.003,rtol=1e-4)

    def test_terminal_boundary_and_interior_kink_smoothing(self):
        trace=h.OpenTrace.from_callable('buffer',2,synthetic_post)
        for w in (.41,.47):
            x=np.array([w-1e-7,w+1e-7]);t=np.array([.5,.5]);bi=np.array([4,4]);der=h.base_jet(t,x,bi,trace)[2]
            self.assertLess(float(np.max(np.abs(der[1]-der[0]))),1e-4)
        w=np.array([m.WD,.41,.8,m.UPPER]);bi=np.array([0,1,3,4])
        np.testing.assert_array_equal(h.base_values(np.ones(4),w,bi,trace),trace.jet(w,bi)[0])
        with self.assertRaises(ValueError):h.base_jet(np.array([1.]),np.array([.8]),np.array([0]),trace)


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HeatControls))
    sys.exit(0 if result.wasSuccessful() else 1)
