# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Train the GBM router: LightGBM heads + tier safety calibration.

학습에만 LightGBM/NumPy를 사용하고, 결과는 순수 Python 라우터(gbm.py)가
읽는 JSON artifact로 내보냅니다. 안전계수는 Train OOF 예측으로 보정합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router import gbm_router as gbm
from ossp_router.gbm_features import (
    DEFAULT_HASH_BINS,
    FEATURE_VERSION,
    detect_family,
    extract_matrix,
)
from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Outcome,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
)

FOLDS = 5
SAFETY_GRID = [round(0.30 + 0.01 * i, 4) for i in range(76)]  # 0.30 .. 1.05
PESSIMISM_GRID = (0.0, 0.3, 0.6, 1.0, 1.4)  # 선택 제약용 비관 계수 k (cost×e^{kσ})
# 안전계수는 "한 표본에서 예산을 지키는가"가 아니라 "평가셋을 다시 추출해도
# 지키는가"로 고릅니다. 실현 비용비의 부트스트랩 상위 분위가 한도 안에 들어야
# 합니다. 한 표본에 맞춰 한도에 붙이면 다른 표본에서 0점을 맞습니다.
RATIO_QUANTILE = 99.0
BUDGET_MARGIN = 0.99
BOOTSTRAP_REPS = 2000
FAMILY_MIN_ITEMS = 80  # 이보다 작은 family는 전역 보정을 사용
FAMILY_PRIOR_WEIGHT = 120.0  # family별 보정을 전역 쪽으로 축소하는 강도
# 기대점수의 argmax는 부트스트랩 추출에 따라 흔들립니다(곡선이 정점 근처에서
# 평평하기 때문). 대신 "초과확률이 목표 이하인 가장 큰 안전계수"를 고르고,
# 인접한 더 낮은 값도 같은 조건을 만족할 것을 요구해 우연한 점을 배제합니다.
OVERRUN_TARGET = 0.015


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = float(policy.token_unit)
    return (
        float(rates.fixed_cost)
        + outcome.input_tokens * float(rates.input_token_rate) / unit
        + outcome.output_tokens * float(rates.output_token_rate) / unit
    )


def _load_split(input_path: Path, outcomes_path: Path, policy: RoutingPolicy):
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    outcome_map: Dict[str, Dict[str, Outcome]] = {}
    for outcome in outcomes.outcomes:
        outcome_map.setdefault(outcome.episode_id, {})[outcome.model_id] = outcome
    scores = []
    costs = []
    for episode in inputs.episodes:
        by_model = outcome_map[episode.episode_id]
        scores.append([float(by_model[m].score) for m in MODEL_IDS])
        costs.append([_outcome_cost(by_model[m], policy) for m in MODEL_IDS])
    return inputs, np.asarray(scores), np.asarray(costs)


def _cluster_augment(
    X: np.ndarray,
    actual_scores: np.ndarray,
    actual_log_costs: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """k-means 클러스터별 타깃 인코딩 특징을 만듭니다.

    반환: (OOF 인코딩 행렬, full-train 인코딩 행렬, artifact용 dict).
    """

    from sklearn.cluster import KMeans

    n = len(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=7)
    labels = km.fit_predict(Z)
    fold_ids = np.arange(n) % FOLDS
    targets6 = np.hstack([actual_scores, actual_log_costs])
    global_mean = targets6.mean(axis=0)

    stats_oof = np.zeros((n, 6))
    for fold in range(FOLDS):
        valid = fold_ids == fold
        for cluster in range(n_clusters):
            members = (~valid) & (labels == cluster)
            value = targets6[members].mean(axis=0) if members.sum() >= 5 else global_mean
            rows = valid & (labels == cluster)
            stats_oof[rows] = value

    stats_full = np.zeros((n_clusters, 6))
    for cluster in range(n_clusters):
        members = labels == cluster
        stats_full[cluster] = (
            targets6[members].mean(axis=0) if members.sum() >= 5 else global_mean
        )

    X_cv = np.hstack([X, stats_oof])
    X_final = np.hstack([X, stats_full[labels]])
    augment = {
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "centroids": [[float(v) for v in c] for c in km.cluster_centers_],
        "stats": [[float(v) for v in row] for row in stats_full],
    }
    return X_cv, X_final, augment


RIDGE_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


HEAD_SEEDS = (7, 17, 27)


def _train_heads(
    X_cv: np.ndarray,
    X_final: np.ndarray,
    targets: np.ndarray,
    params: Mapping[str, Any],
    multi_seed_columns: frozenset = frozenset(),
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]]]:
    """Train GBM + ridge per target, blend by OOF RMSE.

    X_cv는 OOF 타깃 인코딩이 적용된 fold용 행렬, X_final은 추론 시점과
    같은 full-train 인코딩 행렬입니다.
    Returns per-head export dicts, the blended OOF matrix, and summaries.
    """

    import lightgbm as lgb
    from sklearn.linear_model import Ridge

    n, n_targets = targets.shape
    fold_ids = np.arange(n) % FOLDS
    mu_cv = X_cv.mean(axis=0)
    sd_cv = X_cv.std(axis=0)
    sd_cv[sd_cv == 0] = 1.0
    Xs_cv = (X_cv - mu_cv) / sd_cv
    mu = X_final.mean(axis=0)
    sd = X_final.std(axis=0)
    sd[sd == 0] = 1.0
    Xs_final = (X_final - mu) / sd

    oof = np.zeros_like(targets)
    heads: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for col in range(n_targets):
        y = targets[:, col]
        seeds = HEAD_SEEDS if col in multi_seed_columns else (params["random_state"],)
        oof_tree = np.zeros(n)
        col_best = []
        for fold in range(FOLDS):
            valid = fold_ids == fold
            fold_preds = []
            for seed in seeds:
                seed_params = dict(params)
                seed_params["random_state"] = seed
                model = lgb.LGBMRegressor(**seed_params)
                model.fit(
                    X_cv[~valid],
                    y[~valid],
                    eval_set=[(X_cv[valid], y[valid])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                fold_preds.append(model.predict(X_cv[valid]))
                col_best.append(model.best_iteration_ or params["n_estimators"])
            oof_tree[valid] = np.mean(fold_preds, axis=0)
        final_iters = max(30, int(np.median(col_best)))
        finals = []
        for seed in seeds:
            final_params = dict(params)
            final_params["n_estimators"] = final_iters
            final_params["random_state"] = seed
            model = lgb.LGBMRegressor(**final_params)
            model.fit(X_final, y)
            finals.append(model)
        final = finals[0]

        best_alpha = None
        best_alpha_rmse = None
        best_oof_ridge = None
        for alpha in RIDGE_ALPHAS:
            oof_ridge = np.zeros(n)
            for fold in range(FOLDS):
                valid = fold_ids == fold
                ridge = Ridge(alpha=alpha)
                ridge.fit(Xs_cv[~valid], y[~valid])
                oof_ridge[valid] = ridge.predict(Xs_cv[valid])
            rmse = float(np.sqrt(np.mean((oof_ridge - y) ** 2)))
            if best_alpha_rmse is None or rmse < best_alpha_rmse:
                best_alpha, best_alpha_rmse, best_oof_ridge = alpha, rmse, oof_ridge
        final_ridge = Ridge(alpha=best_alpha)
        final_ridge.fit(Xs_final, y)

        best_w = None
        best_w_rmse = None
        for w in BLEND_WEIGHTS:
            blend = w * oof_tree + (1 - w) * best_oof_ridge
            rmse = float(np.sqrt(np.mean((blend - y) ** 2)))
            if best_w_rmse is None or rmse < best_w_rmse:
                best_w, best_w_rmse = w, rmse
        oof[:, col] = best_w * oof_tree + (1 - best_w) * best_oof_ridge

        # 선형 성분은 원 특징 공간 계수로 변환해 export합니다.
        coef_raw = final_ridge.coef_ / sd
        intercept_raw = float(
            final_ridge.intercept_ - np.dot(final_ridge.coef_, mu / sd)
        )
        sample = X_final[: min(64, n)]
        check = intercept_raw + sample @ coef_raw
        reference = final_ridge.predict(Xs_final[: min(64, n)])
        if float(np.max(np.abs(check - reference))) > 1e-6:
            raise RuntimeError("ridge export 검증 실패")

        head = _export_head([m.booster_ for m in finals], X_final)
        head["linear"] = {
            "intercept": intercept_raw,
            "coef": [float(v) for v in coef_raw],
        }
        head["weight_tree"] = best_w
        heads.append(head)
        summaries.append(
            {
                "iters": final_iters,
                "ridge_alpha": best_alpha,
                "weight_tree": best_w,
                "oof_rmse": best_w_rmse,
            }
        )
    return heads, oof, summaries


def _train_quantile_heads(
    X_cv: np.ndarray,
    X_final: np.ndarray,
    log_costs: np.ndarray,
    params: Mapping[str, Any],
    alpha: float,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """선택 제약용 log-cost 상위 quantile head를 학습합니다 (ridge 없음)."""

    import lightgbm as lgb

    n, n_targets = log_costs.shape
    fold_ids = np.arange(n) % FOLDS
    q_params = dict(params)
    q_params["objective"] = "quantile"
    q_params["alpha"] = alpha
    oof = np.zeros_like(log_costs)
    heads: List[Dict[str, Any]] = []
    for col in range(n_targets):
        y = log_costs[:, col]
        col_best = []
        for fold in range(FOLDS):
            valid = fold_ids == fold
            model = lgb.LGBMRegressor(**q_params)
            model.fit(
                X_cv[~valid],
                y[~valid],
                eval_set=[(X_cv[valid], y[valid])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof[valid, col] = model.predict(X_cv[valid])
            col_best.append(model.best_iteration_ or q_params["n_estimators"])
        final_iters = max(30, int(np.median(col_best)))
        final_params = dict(q_params)
        final_params["n_estimators"] = final_iters
        final = lgb.LGBMRegressor(**final_params)
        final.fit(X_final, y)
        heads.append(_export_head(final.booster_, X_final))
    return heads, oof


def _flatten_tree(structure: Mapping[str, Any]) -> Dict[str, List[float]]:
    features: List[float] = []
    thresholds: List[float] = []
    lefts: List[float] = []
    rights: List[float] = []
    values: List[float] = []

    def add(node: Mapping[str, Any]) -> int:
        index = len(features)
        features.append(0.0)
        thresholds.append(0.0)
        lefts.append(0.0)
        rights.append(0.0)
        values.append(0.0)
        if "leaf_value" in node and "split_feature" not in node:
            features[index] = -1.0
            values[index] = float(node["leaf_value"])
            return index
        features[index] = float(node["split_feature"])
        thresholds[index] = float(node["threshold"])
        lefts[index] = float(add(node["left_child"]))
        rights[index] = float(add(node["right_child"]))
        return index

    add(structure)
    return {"f": features, "t": thresholds, "l": lefts, "r": rights, "v": values}


def _export_head(boosters: Any, X: np.ndarray) -> Dict[str, Any]:
    """하나 또는 여러 booster를 평균 앙상블 head로 export합니다."""

    if not isinstance(boosters, (list, tuple)):
        boosters = [boosters]
    scale = 1.0 / len(boosters)
    trees: List[Dict[str, List[float]]] = []
    for booster in boosters:
        dump = booster.dump_model()
        for info in dump["tree_info"]:
            tree = _flatten_tree(info["tree_structure"])
            tree["v"] = [value * scale for value in tree["v"]]
            trees.append(tree)
    head = {"base": 0.0, "trees": trees}
    sample = X[: min(64, len(X))]
    reference = np.mean([b.predict(sample) for b in boosters], axis=0)
    mine = np.array([gbm.eval_ensemble(head, row) for row in sample.tolist()])
    offset = float(np.mean(reference - mine))
    head["base"] = offset
    check = mine + offset
    max_diff = float(np.max(np.abs(check - reference)))
    if max_diff > 1e-6:
        raise RuntimeError(f"트리 export 검증 실패: max diff {max_diff}")
    return head


def _realized(
    selection: Sequence[str],
    actual_scores: np.ndarray,
    actual_costs: np.ndarray,
) -> Tuple[float, float]:
    indices = [MODEL_IDS.index(model) for model in selection]
    rows = np.arange(len(selection))
    quality = float(actual_scores[rows, indices].mean())
    cost = float(actual_costs[rows, indices].sum())
    ratio = cost / float(actual_costs[:, 0].sum())
    return quality, ratio


def _bootstrap_ratio(
    selection: Sequence[str],
    actual_scores: np.ndarray,
    actual_costs: np.ndarray,
    rng: np.random.Generator,
    eval_size: Optional[int] = None,
    multiplier: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """고정된 선택에 대해 평가셋 재추출 시 품질과 실현 비용비 분포를 냅니다.

    반환: (품질, 비용비 중앙값, 상위 RATIO_QUANTILE 분위, 예산 초과 확률).
    """

    indices = np.fromiter(
        (MODEL_IDS.index(model) for model in selection), dtype=np.int64
    )
    rows = np.arange(len(selection))
    picked_cost = actual_costs[rows, indices]
    light_cost = actual_costs[:, 0]
    picked_score = actual_scores[rows, indices]
    size = min(eval_size or len(selection), len(selection))
    draws = rng.integers(0, len(selection), (BOOTSTRAP_REPS, size))
    ratios = picked_cost[draws].sum(axis=1) / light_cost[draws].sum(axis=1)
    overrun = float((ratios > multiplier).mean()) if multiplier else 0.0
    return (
        float(picked_score.mean()),
        float(np.median(ratios)),
        float(np.percentile(ratios, RATIO_QUANTILE)),
        overrun,
    )


def _train_spread_heads(
    X_cv: np.ndarray,
    X_final: np.ndarray,
    residuals: np.ndarray,
    params: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """log-cost 잔차 크기(E|r|)를 예측하는 head를 학습합니다.

    문항별 비용 예측 불확실성을 추정해 선택 단계의 비관 보정을 문항 단위로
    적용하기 위한 것입니다. 타깃은 log(|잔차| + eps)입니다.
    """

    import lightgbm as lgb

    n, n_targets = residuals.shape
    fold_ids = np.arange(n) % FOLDS
    targets = np.log(np.abs(residuals) + 1e-3)
    heads: List[Dict[str, Any]] = []
    oof = np.zeros_like(targets)
    for col in range(n_targets):
        y = targets[:, col]
        col_best = []
        for fold in range(FOLDS):
            valid = fold_ids == fold
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_cv[~valid],
                y[~valid],
                eval_set=[(X_cv[valid], y[valid])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            oof[valid, col] = model.predict(X_cv[valid])
            col_best.append(model.best_iteration_ or params["n_estimators"])
        final_params = dict(params)
        final_params["n_estimators"] = max(30, int(np.median(col_best)))
        final = lgb.LGBMRegressor(**final_params)
        final.fit(X_final, y)
        heads.append(_export_head(final.booster_, X_final))
    return heads, oof


def _expected_score(
    pred_scores: Sequence[Mapping[str, float]],
    sel_costs: Sequence[Mapping[str, float]],
    budget_costs: Sequence[Mapping[str, float]],
    actual_scores: np.ndarray,
    actual_costs: np.ndarray,
    policy: RoutingPolicy,
    safety: Mapping[str, float],
    pessimism: Mapping[str, float],
    sigma: Mapping[str, float],
    eval_size: Optional[int] = None,
    reps: int = 400,
    mean_log_cost: Optional[np.ndarray] = None,
    item_sigma: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """등급별 기대점수(품질 x 예산준수확률)와 가중 합을 계산합니다.

    평가 배치를 eval_size 크기로 다시 추출하며, 라우터가 매 배치에서 예산을
    다시 계산하는 동작까지 재현합니다.
    """

    n = len(pred_scores)
    size = min(eval_size or n, n)
    rng = np.random.default_rng(101)
    tiers: Dict[str, Any] = {}
    final = 0.0
    for tier in TIERS:
        multiplier = float(policy.tiers[tier].budget_multiplier)
        if item_sigma is not None:
            robust = [
                {
                    m: math.exp(
                        mean_log_cost[i][j] + pessimism[tier] * item_sigma[i][j]
                    )
                    for j, m in enumerate(MODEL_IDS)
                }
                for i in range(n)
            ]
        else:
            inflation = {
                m: math.exp(pessimism[tier] * sigma.get(m, 0.0)) for m in MODEL_IDS
            }
            robust = [
                {m: sel_costs[i][m] * inflation[m] for m in MODEL_IDS}
                for i in range(n)
            ]
        ratios, quals = [], []
        for _ in range(reps):
            idx = rng.integers(0, n, size)
            light_pred = sum(budget_costs[i][MODEL_IDS[0]] for i in idx)
            selection = gbm.select_models(
                [pred_scores[i] for i in idx],
                [robust[i] for i in idx],
                light_pred * multiplier * safety[tier],
            )
            columns = [MODEL_IDS.index(model) for model in selection]
            ratios.append(actual_costs[idx, columns].sum() / actual_costs[idx, 0].sum())
            quals.append(actual_scores[idx, columns].mean())
        ratios = np.asarray(ratios)
        quality = float(np.mean(quals))
        overrun = float((ratios > multiplier).mean())
        expected = quality * (1.0 - overrun)
        tiers[tier] = {
            "expected": expected,
            "quality": quality,
            "overrun": overrun,
            "median_ratio": float(np.median(ratios)),
        }
        final += float(policy.tiers[tier].weight) * expected
    return {"tiers": tiers, "final": final, "eval_size": size}


def _calibrate_safety(
    pred_scores: Sequence[Mapping[str, float]],
    sel_costs: Sequence[Mapping[str, float]],
    budget_costs: Sequence[Mapping[str, float]],
    actual_scores: np.ndarray,
    actual_costs: np.ndarray,
    policy: RoutingPolicy,
    sigma: Mapping[str, float],
    eval_size: Optional[int] = None,
    mean_log_cost: Optional[np.ndarray] = None,
    item_sigma: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Fold-robust safety: 각 fold를 독립 배치로 보고 전 fold에서 예산을
    지키는 (비관 계수, 안전계수) 조합 중 전체 품질이 가장 높은 값을 고릅니다."""

    n = len(pred_scores)
    rng = np.random.default_rng(11)
    # 문항별 sigma가 있으면 exp(평균 + k*sigma_i)로 비관 보정합니다.
    if item_sigma is not None:
        robust_variants = {
            k: [
                {
                    m: math.exp(mean_log_cost[i][j] + k * item_sigma[i][j])
                    for j, m in enumerate(MODEL_IDS)
                }
                for i in range(n)
            ]
            for k in PESSIMISM_GRID
        }
    else:
        robust_variants = {
            k: [
                {
                    m: sel_costs[i][m] * math.exp(k * sigma.get(m, 0.0))
                    for m in MODEL_IDS
                }
                for i in range(n)
            ]
            for k in PESSIMISM_GRID
        }
    light_pred_total = sum(budget_costs[i][MODEL_IDS[0]] for i in range(n))
    ratios: Dict[str, float] = {}
    pessimism: Dict[str, float] = {}
    detail: Dict[str, Any] = {}
    for tier in TIERS:
        multiplier = float(policy.tiers[tier].budget_multiplier)
        limit = multiplier * BUDGET_MARGIN
        best = None
        for k in PESSIMISM_GRID:
            robust_costs = robust_variants[k]
            for safety in SAFETY_GRID:
                budget = light_pred_total * multiplier * safety
                selection = gbm.select_models(pred_scores, robust_costs, budget)
                quality, median_ratio, tail_ratio, overrun = _bootstrap_ratio(
                    selection, actual_scores, actual_costs, rng, eval_size,
                    multiplier,
                )
                # 목표는 명목 품질이 아니라 기대점수입니다. 예산을 초과한
                # 등급은 0점이므로 품질 x 예산준수확률을 최대화해야 하고,
                # 꼬리 제약만 두면 여전히 한도 경계에 붙습니다.
                expected = quality * (1.0 - overrun)
                if overrun > OVERRUN_TARGET:
                    continue
                # 같은 비관계수에서 더 큰 안전계수를 선호하되, 기대점수가
                # 확실히 나은 쪽을 택합니다(0.002 이내는 동률로 보고 보수적
                # 선택을 유지).
                key = (round(expected / 0.002), safety)
                if best is None or key > best[0]:
                    best = (key, safety, k, quality, median_ratio, tail_ratio, expected, overrun)
        if best is None:
            # 어떤 조합도 상위 분위 기준을 못 지키면 all-light로 후퇴합니다.
            # 비용비가 정확히 1.0이므로 모든 등급에서 예산 안입니다.
            best = ((0.0, 0.0), SAFETY_GRID[0], PESSIMISM_GRID[-1], 0.0, 1.0, 1.0, 0.0, 0.0)
        ratios[tier] = best[1]
        pessimism[tier] = best[2]
        detail[tier] = {
            "safety_ratio": best[1],
            "pessimism_k": best[2],
            "oof_quality": best[3],
            "oof_median_ratio": best[4],
            f"oof_p{int(RATIO_QUANTILE)}_ratio": best[5],
            "oof_expected": best[6],
            "oof_overrun": best[7],
        }
    return ratios, pessimism, detail


def _rows_from_matrix(
    scores: np.ndarray, log_costs: np.ndarray, sigma: Mapping[str, float]
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    inflation = [
        math.exp(0.5 * float(sigma.get(m, 0.0)) ** 2) for m in MODEL_IDS
    ]
    pred_scores = []
    pred_costs = []
    for i in range(len(scores)):
        score_row = {MODEL_IDS[0]: 0.0}
        for j, m in enumerate(MODEL_IDS[1:]):
            score_row[m] = float(min(1.0, max(-1.0, scores[i, j])))
        cost_row = {
            m: float(math.exp(log_costs[i, j])) * inflation[j]
            for j, m in enumerate(MODEL_IDS)
        }
        light = cost_row[MODEL_IDS[0]]
        cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], light * (1 + 1e-12))
        cost_row[MODEL_IDS[2]] = max(
            cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1 + 1e-12)
        )
        pred_scores.append(score_row)
        pred_costs.append(cost_row)
    return pred_scores, pred_costs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="GBM 라우터 학습")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--outcomes", type=Path, action="append", required=True)
    parser.add_argument(
        "--risk-eval-size",
        type=int,
        default=880,
        help="안전계수 보정 시 가정하는 평가 배치 크기. 작을수록 보수적입니다.",
    )
    parser.add_argument("--validation-input", type=Path)
    parser.add_argument("--validation-outcomes", type=Path)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--hash-bins", type=int, default=DEFAULT_HASH_BINS)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument(
        "--family-calibration",
        action="store_true",
        help="family별 uplift 보정 사용. Dev 프론티어에서 -0.0065로 측정되어"
        " 기본 비활성입니다(train OOF 노이즈에 과적합).",
    )
    parser.add_argument("--cost-quantile", type=float, default=0.0)
    parser.add_argument("--num-leaves", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--feature-fraction", type=float, default=0.7)
    args = parser.parse_args(argv)

    policy = load_policy(args.policy) if args.policy else load_bundled_policy()
    if len(args.input) != len(args.outcomes):
        raise SystemExit("--input과 --outcomes 개수가 같아야 합니다.")
    episodes = []
    score_parts, cost_parts = [], []
    for input_path, outcomes_path in zip(args.input, args.outcomes):
        split_inputs, split_scores, split_costs = _load_split(
            input_path, outcomes_path, policy
        )
        episodes.extend(split_inputs.episodes)
        score_parts.append(split_scores)
        cost_parts.append(split_costs)
        print(f"  {input_path}: {len(split_inputs.episodes)}문항", flush=True)
    actual_scores = np.vstack(score_parts)
    actual_costs = np.vstack(cost_parts)
    print(f"학습 합계 {len(episodes)}문항, 특징 추출 중...", flush=True)
    X = np.asarray(extract_matrix(episodes, args.hash_bins))
    # score는 light 대비 uplift(Δ)를 직접 회귀합니다. 라우팅 선택은
    # 에피소드 내 모델 간 차이만 사용하므로 절대 score가 필요 없고,
    # 공통 난이도 노이즈가 상쇄되어 Δ 추정이 더 안정적입니다.
    uplift = actual_scores[:, 1:] - actual_scores[:, :1]
    targets = np.hstack([uplift, np.log(actual_costs)])

    params = dict(
        objective="regression",
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        colsample_bytree=args.feature_fraction,
        subsample=0.9,
        subsample_freq=1,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        random_state=7,
    )
    print("k-means 클러스터 타깃 인코딩 생성...", flush=True)
    X_cv, X_final, cluster_augment = _cluster_augment(
        X, actual_scores, np.log(actual_costs), args.clusters
    )

    print("GBM+ridge 6개 head 학습(5-fold OOF, blend)...", flush=True)
    heads, oof, summaries = _train_heads(X_cv, X_final, targets, params)
    n_uplift = 2
    score_rmse = float(np.sqrt(np.mean((oof[:, :n_uplift] - targets[:, :n_uplift]) ** 2)))
    cost_rmse = float(np.sqrt(np.mean((oof[:, n_uplift:] - targets[:, n_uplift:]) ** 2)))
    print(f"OOF uplift RMSE={score_rmse:.4f}  log-cost RMSE={cost_rmse:.4f}")
    labels = [f"uplift:{m}" for m in MODEL_IDS[1:]] + [f"cost:{m}" for m in MODEL_IDS]
    for label, summary in zip(labels, summaries):
        print(f"    {label}: {summary}")

    log_cost_sigma = {
        m: float(np.std(oof[:, n_uplift + j] - targets[:, n_uplift + j]))
        for j, m in enumerate(MODEL_IDS)
    }
    print(f"log-cost 잔차 σ: { {m: round(v, 3) for m, v in log_cost_sigma.items()} }")

    # 이분산 비용 모델: 평균 log-cost head의 OOF 잔차로 문항별 오차 크기를
    # 학습합니다. 선택 단계에서 exp(평균 + k*sigma_i)를 비용으로 사용해
    # 예측이 불안정한 문항만 보수적으로 처리합니다.
    print("문항별 비용 분산(이분산) head 학습...", flush=True)
    cost_residuals = oof[:, n_uplift:] - targets[:, n_uplift:]
    spread_heads, spread_oof = _train_spread_heads(
        X_cv, X_final, cost_residuals, params
    )
    oof_sigma = np.minimum(3.0, np.exp(spread_oof) / 0.7978845608028654)
    for j, model in enumerate(MODEL_IDS):
        print(
            f"    sigma[{model}] 중앙 {np.median(oof_sigma[:, j]):.3f}"
            f" p90 {np.percentile(oof_sigma[:, j], 90):.3f}"
            f" (전역 상수 {log_cost_sigma[model]:.3f})"
        )

    quantile_heads = None
    if args.cost_quantile > 0:
        print(f"quantile(alpha={args.cost_quantile}) 비용 head 학습...", flush=True)
        quantile_heads, q_oof = _train_quantile_heads(
            X_cv, X_final, np.log(actual_costs), params, args.cost_quantile
        )

    # OOF에서 uplift shrinkage 보정 (slope, intercept)을 학습해 예측
    # 과신(특히 ax31 uplift)을 줄입니다. 추론과 보정 모두에 적용합니다.
    uplift_calibration = {}
    oof_cal = oof.copy()
    for j, model in enumerate(MODEL_IDS[1:]):
        design = np.vstack([oof[:, j], np.ones(len(X))]).T
        (slope, intercept), *_ = np.linalg.lstsq(
            design, targets[:, j], rcond=None
        )
        uplift_calibration[model] = [float(slope), float(intercept)]
        oof_cal[:, j] = slope * oof[:, j] + intercept
        print(f"  uplift 보정 {model}: slope={slope:.3f} intercept={intercept:+.3f}")

    # family별 uplift 보정: 어휘 특징이 통하는 문항군(other, math_synth)에서는
    # slope가 1에 가깝고, 통하지 않거나 역상관인 군(mcq, rules)에서는 0으로
    # 수렴해 그 family 평균으로 후퇴합니다. family 표본이 작으면 전역 보정
    # 쪽으로 축소합니다(경험적 베이즈, tau=FAMILY_PRIOR_WEIGHT).
    families = np.array([detect_family(episode_text(e)) for e in episodes])
    by_family: Dict[str, Dict[str, List[float]]] = {}
    for family in sorted(set(families)) if args.family_calibration else ():
        mask = families == family
        if mask.sum() < FAMILY_MIN_ITEMS:
            continue
        entry: Dict[str, List[float]] = {}
        for j, model in enumerate(MODEL_IDS[1:]):
            design = np.vstack([oof[mask, j], np.ones(int(mask.sum()))]).T
            (slope_raw, intercept_raw), *_ = np.linalg.lstsq(
                design, targets[mask, j], rcond=None
            )
            weight = mask.sum() / (mask.sum() + FAMILY_PRIOR_WEIGHT)
            g_slope, g_intercept = uplift_calibration[model]
            slope = weight * slope_raw + (1 - weight) * g_slope
            intercept = weight * intercept_raw + (1 - weight) * g_intercept
            slope = float(np.clip(slope, 0.0, 1.2))
            entry[model] = [slope, float(intercept)]
        by_family[family] = entry
        summary = " ".join(
            f"{m.split('-')[0]} slope={entry[m][0]:.2f}" for m in MODEL_IDS[1:]
        )
        print(f"  family 보정 {family:<13} n={int(mask.sum()):>4}  {summary}")

    # family 보정을 OOF 예측에 적용해 보정/평가에 반영
    for j, model in enumerate(MODEL_IDS[1:]):
        for family, entry in by_family.items():
            mask = families == family
            slope, intercept = entry[model]
            oof_cal[mask, j] = slope * oof[mask, j] + intercept

    pred_scores, pred_costs = _rows_from_matrix(
        oof_cal[:, :n_uplift], oof_cal[:, n_uplift:], log_cost_sigma
    )
    # 평균 비용 예측을 문항별 sigma로 보정합니다 (라우터와 같은 규칙).
    for i in range(len(X)):
        for j, model in enumerate(MODEL_IDS):
            base = math.exp(oof[i, n_uplift + j] + 0.5 * oof_sigma[i, j] ** 2)
            pred_costs[i][model] = base
        light = pred_costs[i][MODEL_IDS[0]]
        pred_costs[i][MODEL_IDS[1]] = max(
            pred_costs[i][MODEL_IDS[1]], light * (1 + 1e-12)
        )
        pred_costs[i][MODEL_IDS[2]] = max(
            pred_costs[i][MODEL_IDS[2]], pred_costs[i][MODEL_IDS[1]] * (1 + 1e-12)
        )

    # 선택 제약용 OOF 비용 (gbm.selection_costs와 같은 규칙).
    if quantile_heads is not None:
        sel_costs = []
        for i in range(len(X)):
            row = {m: float(math.exp(q_oof[i, j])) for j, m in enumerate(MODEL_IDS)}
            light = max(row[MODEL_IDS[0]], pred_costs[i][MODEL_IDS[0]])
            row[MODEL_IDS[0]] = light
            row[MODEL_IDS[1]] = max(row[MODEL_IDS[1]], light * (1 + 1e-12))
            row[MODEL_IDS[2]] = max(row[MODEL_IDS[2]], row[MODEL_IDS[1]] * (1 + 1e-12))
            sel_costs.append(row)
    else:
        sel_costs = pred_costs
    safety, pessimism, detail = _calibrate_safety(
        pred_scores,
        sel_costs,
        pred_costs,
        actual_scores,
        actual_costs,
        policy,
        log_cost_sigma,
        args.risk_eval_size,
        oof[:, n_uplift:],
        oof_sigma,
    )
    for tier in TIERS:
        d = detail[tier]
        tail_key = f"oof_p{int(RATIO_QUANTILE)}_ratio"
        print(
            f"  {tier}: safety={safety[tier]} k={pessimism[tier]}"
            f"  OOF quality={d['oof_quality']:.4f}"
            f"  비용비 중앙={d['oof_median_ratio']:.3f}"
            f" p{int(RATIO_QUANTILE)}={d[tail_key]:.3f}"
        )
    # OOF 기대점수: 폴드 밖 예측으로 계산한 품질에 예산 초과 확률을 반영한
    # 값입니다. 학습 데이터를 어디까지 쓰든 편향되지 않는 유일한 지표이므로
    # 설정 비교는 이 값으로 합니다.
    oof_expected = _expected_score(
        pred_scores,
        sel_costs,
        pred_costs,
        actual_scores,
        actual_costs,
        policy,
        safety,
        pessimism,
        log_cost_sigma,
        args.risk_eval_size,
        400,
        oof[:, n_uplift:],
        oof_sigma,
    )
    for tier in TIERS:
        row = oof_expected["tiers"][tier]
        print(
            f"  [OOF 기대] {tier}: 기대 {row['expected']:.4f}"
            f" = 품질 {row['quality']:.4f} x 준수 {1 - row['overrun']:.3f}"
        )
    print(f"  [OOF 기대] 가중 최종 = {oof_expected['final']:.4f}")

    print("트리 export 및 검증...", flush=True)
    artifact = {
        "artifact_type": gbm.ARTIFACT_TYPE,
        "feature_version": FEATURE_VERSION,
        "hash_bins": args.hash_bins,
        "log_cost_sigma": log_cost_sigma,
        "cluster_augment": cluster_augment,
        "tier_safety_ratios": safety,
        "selection_cost_inflation": {
            tier: {
                m: math.exp(pessimism[tier] * log_cost_sigma[m])
                for m in MODEL_IDS
            }
            for tier in TIERS
        },
        "score_heads": {},
        "log_cost_heads": {},
        "training_summary": {
            "oof_score_rmse": score_rmse,
            "oof_log_cost_rmse": cost_rmse,
            "head_summaries": summaries,
            "params": {k: v for k, v in params.items() if k != "n_jobs"},
            "safety_detail": detail,
        },
    }
    artifact["uplift_calibration"] = uplift_calibration
    artifact["uplift_calibration_by_family"] = by_family
    artifact["log_cost_spread_heads"] = {
        m: spread_heads[j] for j, m in enumerate(MODEL_IDS)
    }
    artifact["selection_pessimism"] = {t: pessimism[t] for t in TIERS}
    if quantile_heads is not None:
        artifact["selection_cost_heads"] = {
            m: quantile_heads[j] for j, m in enumerate(MODEL_IDS)
        }
    artifact["score_heads"][MODEL_IDS[0]] = {"base": 0.0, "trees": []}
    artifact["score_heads"][MODEL_IDS[1]] = heads[0]
    artifact["score_heads"][MODEL_IDS[2]] = heads[1]
    for j, model in enumerate(MODEL_IDS):
        artifact["log_cost_heads"][model] = heads[n_uplift + j]

    report: Dict[str, Any] = {"train_oof": detail, "oof_expected": oof_expected}
    if args.validation_input and args.validation_outcomes:
        vin, v_scores, v_costs = _load_split(
            args.validation_input, args.validation_outcomes, policy
        )
        rows = extract_matrix(vin.episodes, args.hash_bins)
        rows = gbm.augment_rows(artifact, rows)
        p_scores, p_costs = gbm.predict_heads(artifact, rows)
        light_total_pred = sum(r[MODEL_IDS[0]] for r in p_costs)
        report["validation"] = {}
        weighted = 0.0
        for tier in TIERS:
            multiplier = float(policy.tiers[tier].budget_multiplier)
            robust_costs = gbm.selection_costs(artifact, rows, tier, p_costs)
            # Dev는 한 방향 안전 장치로만 씁니다. Train OOF에서 고른
            # 안전계수를 Dev 부트스트랩으로 다시 확인하고, 상위 분위가
            # 한도를 넘으면 낮추기만 합니다. Dev 품질을 최대화하려고 계수를
            # 올리면 그 한 표본에 과적합되어 다른 표본에서 0점을 맞습니다.
            dev_rng = np.random.default_rng(23)
            limit = multiplier * BUDGET_MARGIN
            adjusted = None
            best_dev = None
            candidate = safety[tier]
            while candidate >= SAFETY_GRID[0]:
                budget = light_total_pred * multiplier * candidate
                selection_c = gbm.select_models(p_scores, robust_costs, budget)
                q_c, med_c, tail_c, over_c = _bootstrap_ratio(
                    selection_c, v_scores, v_costs, dev_rng,
                    args.risk_eval_size, multiplier,
                )
                if over_c > OVERRUN_TARGET:
                    candidate = round(candidate - 0.01, 4)
                    continue
                key = (round(q_c * (1.0 - over_c) / 0.002), candidate)
                if best_dev is None or key > best_dev:
                    best_dev = key
                    adjusted = candidate
                    selection, quality = selection_c, q_c
                    median_ratio, tail_ratio, overrun = med_c, tail_c, over_c
                candidate = round(candidate - 0.01, 4)
            if adjusted != safety[tier]:
                print(
                    f"  [dev] {tier}: safety {safety[tier]} -> {adjusted}"
                    f" (dev 기대점수 최대화로 하향)"
                )
                safety[tier] = adjusted
                artifact["tier_safety_ratios"][tier] = adjusted
            if adjusted is None:
                # 어떤 후보도 초과 목표를 만족하지 못하면 all-light로 후퇴
                adjusted = SAFETY_GRID[0]
                selection = [MODEL_IDS[0]] * len(p_scores)
                quality, median_ratio, tail_ratio, overrun = _bootstrap_ratio(
                    selection, v_scores, v_costs, dev_rng,
                    args.risk_eval_size, multiplier,
                )
            ok = overrun <= OVERRUN_TARGET
            tier_score = quality if ok else 0.0
            weight = float(policy.tiers[tier].weight)
            weighted += tier_score * weight
            counts = {m: selection.count(m) for m in MODEL_IDS}
            report["validation"][tier] = {
                "quality": quality,
                "cost_ratio_median": median_ratio,
                f"cost_ratio_p{int(RATIO_QUANTILE)}": tail_ratio,
                "within_budget": ok,
                "safety_ratio": adjusted,
                "model_counts": counts,
            }
            print(
                f"  [dev] {tier}: quality={quality:.4f} 비용비 중앙={median_ratio:.3f}"
                f" p{int(RATIO_QUANTILE)}={tail_ratio:.3f} (한도 {multiplier})"
                f" {'OK' if ok else 'OVER!'} {counts}"
            )
        report["validation"]["final_score"] = weighted
        print(f"  [dev] final(가중) = {weighted:.6f}")

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact), encoding="utf-8")
    size_mb = args.artifact.stat().st_size / 1e6
    print(f"OK: artifact 저장 ({size_mb:.1f} MB): {args.artifact}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
