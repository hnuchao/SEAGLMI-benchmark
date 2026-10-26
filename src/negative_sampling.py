"""Negative sampling used by the benchmark.

``sample`` draws negatives for a list of positive pairs.
* scheme='uniform_train' (standard protocol): uniform lncRNA-miRNA pairs from the given
  lncRNA and miRNA universes.
* any other scheme (strict protocol): for each positive, keep one endpoint and redraw the
  other from the positive endpoint list, so that endpoint frequencies follow the positives;
  copies alternate which endpoint is kept.
Known, blocked (validation/test/excluded) and duplicate pairs are rejected; up to ``retries``
proposals are tried per negative. ``seed`` derives reproducible per-unit random seeds.
"""
import csv, hashlib, io
from pathlib import Path
import numpy as np

COLS = ['mirna_node', 'ensg', 'label', 'arm_inferred']
def pair(r):return r['mirna_node'],r['ensg']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,rows,cols=COLS):
    s=io.StringIO();w=csv.DictWriter(s,fieldnames=cols,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows);b=s.getvalue().encode()
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==b,('Immutable input changed',str(p))
    else:p.write_bytes(b)
    return sha(p)
def seed(master,unit,role):return int.from_bytes(hashlib.sha256(f'{master}/{unit}/{role}'.encode()).digest()[:8],'big')
def positives(rows):return [dict(r,label=1) for r in rows if int(float(r['label']))==1]
def sample(pos,ratio,known,blocked,scheme,rng,lnc=None,mirna=None,retries=200):
    n=len(pos);M=[r['mirna_node'] for r in pos];L=[r['ensg'] for r in pos]
    forbidden=set(known)|set(blocked)|{pair(r) for r in pos};out=[];seen=set()
    if scheme=='uniform_train':
        lm=list(lnc) if lnc is not None else sorted(set(L));mm=list(mirna) if mirna is not None else sorted(set(M))
        for _ in range(retries*ratio*n):
            if len(out)>=ratio*n:break
            k=(mm[rng.integers(len(mm))],lm[rng.integers(len(lm))])
            if k not in forbidden and k not in seen:seen.add(k);out.append(k)
    else:
        for c in range(ratio):
            for j,i in enumerate(rng.permutation(n)):
                keep_lnc=(j+c)%2==0
                for _ in range(retries):
                    k=(M[rng.integers(n)],L[i]) if keep_lnc else (M[i],L[rng.integers(n)])
                    if k not in forbidden and k not in seen:seen.add(k);out.append(k);break
    assert out and not seen&forbidden
    return [dict(mirna_node=m,ensg=l,label=0,arm_inferred=False) for m,l in out]
