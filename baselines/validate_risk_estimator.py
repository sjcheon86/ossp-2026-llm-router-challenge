# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""리스크 추정기 검증: 실패가 관측된 hash-regex baseline에 적용해
관측값(premium 약 4.2)이 우리 예측 분포의 어느 분위인지 확인."""
import sys; sys.path.insert(0, 'baselines')
import json
from pathlib import Path
import numpy as np
import hash_regex as hr
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input
from train_gbm import _load_split

policy = load_bundled_policy()
art = hr.load_artifact(Path('baselines/hash-regex-public.v1.json'))
inputs = load_input(Path('data/materialized/dev/inputs.json'))
_, vs, vc = _load_split(Path('data/materialized/dev/inputs.json'), Path('data/dev/outcomes.json'), policy)
n = len(inputs.episodes)
preds = [hr.predict_episode(e, art) for e in inputs.episodes]
scores = [p[0] for p in preds]; costs = [p[1] for p in preds]
MULT = {'fast':1.25,'balanced':2.0,'premium':4.0}
OBSERVED = {'premium': 4.2}   # 공개 문서에 기록된 채점셋 실측치

def run(idx, tier):
    mult = float(policy.tiers[tier].budget_multiplier)
    sel, _ = hr.select_models([scores[i] for i in idx], [costs[i] for i in idx],
                              budget_multiplier=mult,
                              safety_ratio=art.tier_safety_ratios[tier])
    if tier == 'premium':
        sel, _ = hr.fill_ax31_upgrades(sel, [scores[i] for i in idx], [costs[i] for i in idx],
                                       budget_multiplier=mult,
                                       safety_ratio=hr.PREMIUM_AX31_FILL_SAFETY_RATIO)
    j = [MODEL_IDS.index(m) for m in sel]
    return vc[idx, j].sum()/vc[idx, 0].sum()

print("### hash-regex baseline에 우리 추정기 적용 (dev 880문항)\n")
rng = np.random.default_rng(4242)
for tier in ('fast','balanced','premium'):
    mult = MULT[tier]
    full = run(np.arange(n), tier)
    half = []
    for _ in range(400):
        perm = rng.permutation(n)
        for k in (0,1):
            half.append(run(perm[k*440:(k+1)*440], tier))
    h = np.array(half); med = np.median(h)
    exc99 = np.percentile(h, 99) - med
    # n=880 환산 (초과폭 x0.5), 정규 근사로 분위 계산용 스케일
    scale880 = 0.5 * (np.percentile(h, 84) - med)   # 약 1시그마 환산
    est_p99 = full + 0.5*exc99
    line = f"{tier}: dev 전체 {full:.3f} / 한도 {mult} | n=880 추정 p99 {est_p99:.3f}"
    if tier in OBSERVED:
        obs = OBSERVED[tier]
        z = (obs - full) / max(scale880, 1e-9)
        # 추정 분포에서 관측값의 분위 (독립분할 표본을 n=880로 축소 환산)
        shrunk = med + 0.5*(h - med)
        pct = (shrunk < obs).mean()*100
        line += f" | **관측 {obs} → 추정분포의 {pct:.1f} 분위 (z≈{z:+.1f})**"
    print(line, flush=True)

print("\n### 다른 baseline들의 예측 리스크 (dev 제출물 고정, 부분집합 재추출)\n")
ids = {e.episode_id: i for i, e in enumerate(inputs.episodes)}
import glob, os
for name, d in (('prompt-heuristic','build/ph-dev'), ('hash-regex','build/hash-regex/dev'),
                ('우리 v10','build/gbm/dev-v10')):
    if not os.path.isdir(d): continue
    out = []
    for tier in ('fast','balanced','premium'):
        f = f'{d}/{tier}.json'
        if not os.path.exists(f): continue
        sub = json.load(open(f))
        sel = [None]*n
        for dec in sub['decisions']: sel[ids[dec['episode_id']]] = dec['model_id']
        j = np.array([MODEL_IDS.index(m) for m in sel])
        full = vc[np.arange(n), j].sum()/vc[:,0].sum()
        hh = []
        for _ in range(400):
            perm = rng.permutation(n)
            for k in (0,1):
                idx = perm[k*440:(k+1)*440]
                hh.append(vc[idx, j[idx]].sum()/vc[idx,0].sum())
        hh = np.array(hh); med = np.median(hh)
        est = full + 0.5*(np.percentile(hh,99)-med)
        out.append(f"{tier} {full:.3f}→p99 {est:.3f}/{MULT[tier]} ({'안전' if est<=MULT[tier] else '위험'})")
    print(f"  {name:<14} " + " | ".join(out), flush=True)
