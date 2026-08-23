# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""최종 bound: max(표본 추정 p99, 계통 이동 후 비율) <= 한도.
두 리스크를 더하면 이중 계산이고(관측 이동이 이미 표본 분포의 90분위),
무시하면 fast처럼 꼬리가 얇은 등급이 노출된다."""
import json
from pathlib import Path
import numpy as np
from ossp_router import gbm_router as gbm
from ossp_router.gbm_features import extract_matrix, detect_family
from ossp_router.heuristic import episode_text
from ossp_router.protocol import MODEL_IDS, load_bundled_policy
from train_gbm import _load_split

policy = load_bundled_policy()
art = json.load(open('src/ossp_router/resources/gbm-artifact.v1.json'))
MULT = {'fast':1.25,'balanced':2.0,'premium':4.0}
W = {'fast':0.4,'balanced':0.3,'premium':0.3}
SHIFT = 2*0.054
vin, vs, vc = _load_split(Path('data/materialized/dev/inputs.json'), Path('data/dev/outcomes.json'), policy)
n = len(vin.episodes)
fam = [detect_family(episode_text(e)) for e in vin.episodes]
rows = gbm.augment_rows(art, extract_matrix(vin.episodes, art['hash_bins']))
ps, pc = gbm.predict_heads(art, rows, fam)
lp = np.array([pc[i][MODEL_IDS[0]] for i in range(n)])
SPLITS = [np.random.default_rng(s).permutation(n) for s in range(250)]
picks = {}; tot = 0.0
for tier, mult in MULT.items():
    rc = gbm.selection_costs(art, rows, tier, pc)
    print(f"\n[{tier}] 한도 {mult}")
    for s in [round(0.86+0.02*i, 3) for i in range(16)]:
        sel_f = gbm.select_models(ps, rc, lp.sum()*mult*s)
        jf = [MODEL_IDS.index(m) for m in sel_f]
        full = vc[np.arange(n), jf].sum()/vc[:,0].sum()
        q = vs[np.arange(n), jf].mean()
        half = []
        for perm in SPLITS:
            for k in (0,1):
                idx = perm[k*440:(k+1)*440]
                b = lp[idx].sum()*mult*s
                sl = gbm.select_models([ps[i] for i in idx], [rc[i] for i in idx], b)
                jj = [MODEL_IDS.index(m) for m in sl]
                half.append(vc[idx,jj].sum()/vc[idx,0].sum())
        h = np.array(half)
        p99 = full + 0.5*(np.percentile(h,99) - np.median(h))
        shifted = full*(1+SHIFT)
        bound = max(p99, shifted)
        ok = bound <= mult
        print(f"  s={s}: 품질 {q:.4f} 비율 {full:.3f} | 표본p99 {p99:.3f} 계통 {shifted:.3f} → bound {bound:.3f} {'OK' if ok else 'X'}", flush=True)
        if ok and (tier not in picks or q > picks[tier][1]):
            picks[tier] = (s, q, full, bound)
    if tier in picks:
        s,q,f,b = picks[tier]; tot += W[tier]*q
        print(f"  -> 최적 s={s} 품질 {q:.4f} 비율 {f:.3f} bound {b:.3f}")
print(f"\n=== 최적: " + ", ".join(f"{t} s={picks[t][0]}" for t in MULT if t in picks))
print(f"가중 품질 = {tot:.4f}  (v10 배포본 0.6840)")
json.dump({t: picks[t][0] for t in picks}, open('build/bound_max_safety.json','w'))
