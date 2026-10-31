import sys
import unittest
import numpy as np
import manufactured_control as m
import heat_lifting as h
import solve_day as s
from test_physical_lifting import synthetic_post


class BatchingControls(unittest.TestCase):
    def test_packed_budget_jets_and_operator_equal_scalar(self):
        trace=h.OpenTrace.from_callable('buffer',2,synthetic_post)
        t=np.array([.11,.41,.71]);w=np.array([.381,.731,1.501])
        all_jets=h.base_all_jet(t,w,trace)
        for bi in range(5):
            one=h._base_jet(t,w,np.full(len(t),bi),trace)
            for a,b in zip(all_jets,one):np.testing.assert_allclose(a[:,bi],b,atol=1e-11)
        operator=s.base_operator_all(t,w,trace,.12)
        for bi in range(5):np.testing.assert_allclose(operator[:,bi],s.base_operator(t,w,np.full(len(t),bi),trace,.12),atol=1e-11)

    def test_production_values_and_trace_builder_equal_scalar(self):
        trace=h.OpenTrace.from_callable('buffer',2,synthetic_post)
        coefs=np.random.Generator(np.random.PCG64(37)).normal(0,.001,(5,513,2));slab=s.Slab(20260928,coefs,trace)
        t=np.array([0.,.31,1.,.9]);w=np.array([.381,.831,1.11,1.61]);all_values=slab.values_all(t,w)
        for bi in range(5):np.testing.assert_allclose(all_values[:,bi],slab.values(t,w,np.full(len(t),bi)),atol=1e-11)
        # Tiny query verifies action-map mixture independently of batching.
        w=np.array([.36,.42,.46,.8,1.53]);t=np.zeros(len(w));v=slab.values_all(t,w)
        for policy in ('zero','schedule','buffer'):
            for bi in range(5):
                prob,q,nxt=h.action(policy,2,w,np.full(len(w),bi));yes=slab.values(t,w+q,nxt)
                expected=(1-prob[:,None])*v[:,bi]+prob[:,None]*yes;expected[:,0]+=prob*q
                np.testing.assert_allclose(h.policy_values(slab.values,policy,2,w,np.full(len(w),bi)),expected,atol=1e-11)


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BatchingControls))
    sys.exit(0 if result.wasSuccessful() else 1)
