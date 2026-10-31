"""Call the unchanged original audit once and retain its residual arrays."""
from pathlib import Path
import sys,numpy as np
import evaluate_fixed_policy as audit
out=Path(sys.argv[sys.argv.index('--audit-dir')+1]);out.mkdir(parents=True,exist_ok=True)
original_load=audit.load_export;original_pide=audit.model_pide;identities={}
def observed_load(path):
 model,payload=original_load(path);identities[id(model)]=(float(payload['intensity']),int(payload['seed']));return model,payload
def observed_pide(model,states,intensity,chunk=256):
 result=original_pide(model,states,intensity,chunk);lam,seed=identities[id(model)];np.savez_compressed(out/f'raw_pide_lambda_{lam:g}_seed_{seed}.npz',states=states,residual=result);return result
audit.load_export=observed_load;audit.model_pide=observed_pide
if __name__=='__main__':raise SystemExit(audit.main())
