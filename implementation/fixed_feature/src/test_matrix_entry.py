import ast
import json
from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np
import audit_adapter as a


class MatrixEntryControls(unittest.TestCase):
    def test_production_functions_identical_to_passed_representative(self):
        parent=a.ROOT.parent/'supplement_redesign_20261020_r7/src/solve_day.py'
        left=ast.parse(parent.read_text());right=ast.parse((a.ROOT/'src/solve_day.py').read_text())
        for name in ('Slab','run_day','base_operator','base_operator_all','save_checkpoint','load_checkpoint'):
            x=next(n for n in left.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            y=next(n for n in right.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name)
            self.assertEqual(ast.dump(x,include_attributes=False),ast.dump(y,include_attributes=False))

    def test_all_12_real_metadata_load_paths(self):
        a.verify_imports();digest=a.m.sha(a.ROOT/'source_manifest.json');coeff=np.zeros((5,513,2))
        grid=np.arange(a.m.WD+a.heat.GRID_H,a.m.UPPER-a.heat.GRID_H/2,a.heat.GRID_H)
        v=np.zeros((5,len(grid),2));v[:,:,1]=(1-(grid-a.m.WD)/a.m.LENGTH)[None,:]
        for policy in ('zero','schedule','buffer'):
            trace=a.heat.OpenTrace(policy,v)
            for intensity in (0.,.12):
                for seed in (20260928,20260929):
                    with tempfile.TemporaryDirectory(dir=a.ROOT/'runs') as td:
                        run=Path(td);files={}
                        for day in range(20):
                            meta={'policy':policy,'intensity':intensity,'seed':seed,'day':day,'source_manifest_sha256':digest,'trace_max_abs_error_cost':0.,'trace_max_abs_error_risk':0.}
                            p=run/f'day_{day:02d}.npz';a.s.save_checkpoint(p,meta,coeff,None if day==19 else trace,None if day==0 else trace);files[str(p)]=a.m.sha(p)
                        (run/'checkpoint_manifest.json').write_text(json.dumps(files))
                        model=a.AuditModel(run)
                        self.assertEqual((model.policy,model.intensity,model.seed),(policy,intensity,seed));self.assertEqual(len(model.trace_records),19)
                        self.assertEqual(max(float(r['max_abs_error_risk']) for r in model.trace_records),0.)


if __name__=='__main__':
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MatrixEntryControls))
    sys.exit(0 if result.wasSuccessful() else 1)
