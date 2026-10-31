import tempfile
from pathlib import Path
import unittest
import sys
import numpy as np
import manufactured_control as m
import physical_lifting as p
import solve_day as s
from test_physical_lifting import synthetic_post


class ProductionControls(unittest.TestCase):
    def setUp(self):
        self.trace=p.OpenTrace.from_callable('buffer',3,synthetic_post)
        self.coef=np.random.Generator(np.random.PCG64(37)).normal(0,.001,(5,513,2))

    def test_real_slab_jets_against_values(self):
        t=np.array([.11,.31,.71,.95]);w=np.array([.3821,.4421,.8021,1.5021]);bi=np.array([0,1,3,4]);h=2e-6
        for trace in (None,self.trace):
            slab=s.Slab(20260928,self.coef,trace);v,dt,dw,dww=slab.jet(t,w,bi)
            np.testing.assert_allclose(v,slab.values(t,w,bi),atol=1e-13)
            f=slab.values
            np.testing.assert_allclose(dt,(f(t+h,w,bi)-f(t-h,w,bi))/(2*h),atol=2e-7,rtol=1e-6)
            np.testing.assert_allclose(dw,(f(t,w+h,bi)-f(t,w-h,bi))/(2*h),atol=2e-6,rtol=1e-6)
            np.testing.assert_allclose(dww,(f(t,w+h,bi)-2*f(t,w,bi)+f(t,w-h,bi))/h**2,atol=1e-3,rtol=1e-4)

    def test_atomic_checkpoint_roundtrip_and_identity_rejection(self):
        meta={'policy':'buffer','seed':20260928,'source_manifest_sha256':'test','day':18}
        with tempfile.TemporaryDirectory(dir=m.ROOT/'runs') as td:
            f=Path(td)/'model.npz';s.save_checkpoint(f,meta,self.coef,self.trace,self.trace)
            got,model,pre=s.load_checkpoint(f,'buffer',20260928,'test')
            np.testing.assert_array_equal(model.coefficients,self.coef);np.testing.assert_array_equal(pre.values,self.trace.values)
            with self.assertRaises(ValueError):s.load_checkpoint(f,'buffer',20260929,'test')
            with self.assertRaises(ValueError):s.load_checkpoint(f,'buffer',20260928,'changed')
            self.assertFalse(f.with_suffix('.tmp').exists())

    def test_channel_separation_and_affine_operator(self):
        t=np.array([.2,.6]);w=np.array([.45,.9]);bi=np.array([1,4]);coef=self.coef.copy()
        first=s.Slab(20260928,coef,self.trace).values(t,w,bi)
        coef[:,:,1]+=1.
        other=s.Slab(20260928,coef,self.trace).values(t,w,bi)
        np.testing.assert_array_equal(first[:,0],other[:,0])
        slab=s.Slab(20260928,self.coef,self.trace);v,dt,dw,dww=slab.jet(t,w,bi)
        direct=dt+(.04-.08*w)[:,None]*dw+.5*m.SIGMA**2*dww
        direct+=.12*(sum(prob*slab.values(t,w*np.exp(.3*mark),bi) for mark,prob in zip(m.MARKS,m.PROBABILITIES))-v)
        matrix=m.operator_basis(t,w,slab.weights,slab.bias,.12)
        assembled=s.base_operator(t,w,bi,self.trace,.12)+np.einsum('ij,ijk->ik',matrix,self.coef[bi])
        np.testing.assert_allclose(direct,assembled,atol=1e-12)


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProductionControls))
    sys.exit(0 if result.wasSuccessful() else 1)
