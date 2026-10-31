"""One prospectively frozen nested resolution ladder, manufactured only."""
import argparse
import json
from pathlib import Path
import numpy as np
import manufactured_control as m

PARENT_FEATURES = m.features


def nested_features(seed, count):
    if count not in (128,256,512):
        raise ValueError('unregistered resolution')
    w,b=PARENT_FEATURES(seed)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,20261020])))
    extra=rng.uniform(-3,3,(384,3))
    return np.vstack((w,extra[:count-128,:2])),np.concatenate((b,extra[:count-128,2]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--intensity',type=float,required=True);p.add_argument('--width',type=int,required=True)
    args=p.parse_args()
    if args.width not in (256,512) or args.seed not in (20260928,20260929) or args.intensity not in (0.,.12):
        raise ValueError('unregistered new control row')
    m.features=lambda seed:nested_features(seed,args.width)
    m.run(args.out,args.seed,args.intensity)
    result=json.loads((args.out/'result.json').read_text())
    result.update(entry_source_sha256=m.sha(__file__),hidden_width=args.width)
    m.write_json(args.out/'result.json',result)
