import argparse
import json
import time
import numpy as np
import manufactured_control as m
import heat_lifting as h
from test_physical_lifting import synthetic_post
from heat_support_control import CASES


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,required=True);args=parser.parse_args();start=time.monotonic()
    parent=m.ROOT.parent/'supplement_redesign_20261020_r6/runs/heat_support';path=parent/(args.case+'.npz')
    with np.load(path) as d:
        t,w,bi=[d[k].copy() for k in ('t','w','budget')];expected=[d[f'jet_{k}'].copy() for k in range(4)]
    policy,day=CASES[args.case];trace=None if policy is None else h.OpenTrace.from_callable(policy,day,synthetic_post)
    actual=h.base_jet(t,w,bi,trace);errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,expected)]
    passed=all(e<=bound for e,bound in zip(errors,(1e-10,1e-10,1e-10,1e-8)))
    result={'case':args.case,'rows':len(t),'max_error_value_time_water_curvature':errors,'pass':bool(passed),
            'elapsed_seconds':time.monotonic()-start,'parent_payload_sha256':m.sha(path),'source_sha256':m.sha(__file__),
            'fallback_count':0,'physical_fits':0,'gate_calls':0}
    out=m.ROOT/'runs/batching_support';out.mkdir(exist_ok=True)
    m.write_json(out/(args.case+'.json'),result);print(json.dumps(result),flush=True)
