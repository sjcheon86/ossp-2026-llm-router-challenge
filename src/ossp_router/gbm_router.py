# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""GBM router: tree-ensemble prediction + Lagrangian budget selection.

추론은 표준 라이브러리만 사용합니다. 학습된 트리는 JSON artifact로 전달되며
프롬프트 본문 특징만으로 모델별 예상 score와 log-cost를 예측한 뒤, 등급
예산 안에서 λ 이진탐색과 greedy fill로 선택을 확정합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .gbm_features import FEATURE_VERSION, detect_family, extract_vector
from .heuristic import episode_text, write_submission_atomic
from .protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_policy,
    parse_submission,
    submission_to_dict,
)

ARTIFACT_TYPE = "ossp-gbm-router-v1"

# 트리 노드 인코딩: 각 트리는 {"f": [...], "t": [...], "l": [...], "r": [...],
# "v": [...]} 병렬 배열입니다. f[i] < 0 이면 리프이고 값은 v[i]입니다.
# 자식 인덱스는 같은 배열 안의 위치입니다.


def eval_tree(tree: Mapping[str, Sequence[float]], row: Sequence[float]) -> float:
    features = tree["f"]
    thresholds = tree["t"]
    lefts = tree["l"]
    rights = tree["r"]
    values = tree["v"]
    index = 0
    while True:
        feature = features[index]
        if feature < 0:
            return values[index]
        if row[int(feature)] <= thresholds[index]:
            index = int(lefts[index])
        else:
            index = int(rights[index])


def eval_ensemble(
    head: Mapping[str, Any], row: Sequence[float]
) -> float:
    """트리 합과 선형 성분을 blend 가중치로 결합한 예측을 반환합니다."""

    tree_total = head.get("base", 0.0)
    for tree in head["trees"]:
        tree_total += eval_tree(tree, row)
    linear = head.get("linear")
    if not linear:
        return tree_total
    linear_total = linear["intercept"]
    for coefficient, value in zip(linear["coef"], row):
        linear_total += coefficient * value
    weight = head.get("weight_tree", 1.0)
    return weight * tree_total + (1.0 - weight) * linear_total


def augment_rows(
    artifact: Mapping[str, Any], rows: Sequence[Sequence[float]]
) -> List[List[float]]:
    """클러스터 타깃 인코딩 특징을 기본 벡터 뒤에 붙입니다.

    artifact의 centroids는 표준화 공간에 있으며, 각 행을 가장 가까운
    centroid의 클러스터 통계(모델별 평균 score·log-cost)로 확장합니다.
    """

    augment = artifact.get("cluster_augment")
    if not augment:
        return [list(row) for row in rows]
    mu = augment["mu"]
    sd = augment["sd"]
    centroids = augment["centroids"]
    stats = augment["stats"]
    result = []
    for row in rows:
        z = [(value - m) / s for value, m, s in zip(row, mu, sd)]
        best_distance = None
        best_index = 0
        for index, centroid in enumerate(centroids):
            distance = sum((a - b) * (a - b) for a, b in zip(z, centroid))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        result.append(list(row) + list(stats[best_index]))
    return result


def predict_heads(
    artifact: Mapping[str, Any],
    rows: Sequence[Sequence[float]],
    families: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Return per-episode {model: score} and {model: cost} predictions."""

    scores: List[Dict[str, float]] = []
    costs: List[Dict[str, float]] = []
    score_heads = artifact["score_heads"]
    cost_heads = artifact["log_cost_heads"]
    # log-cost의 잔차 분산 보정: L2로 학습한 log 예측을 exp만 하면 기대
    # 비용을 계통적으로 과소평가하므로 exp(σ²/2)를 곱합니다. 문항별 분산
    # head가 있으면 문항마다 σ를 추정해 적용하고, 없으면 모델별 상수를
    # 사용합니다.
    sigma = artifact.get("log_cost_sigma", {})
    spread_heads = artifact.get("log_cost_spread_heads")
    inflation = {
        model: math.exp(0.5 * float(sigma.get(model, 0.0)) ** 2)
        for model in MODEL_IDS
    }
    # OOF에서 학습한 uplift shrinkage (slope, intercept) 보정.
    # family별 보정이 있으면 그것을 우선 적용합니다. 어휘 특징이 무의미한
    # 문항군에서는 slope가 0에 수렴해 그 family의 평균 uplift로 후퇴하므로,
    # 잘못된 순위를 신뢰해 예산을 낭비하는 일을 막습니다.
    calibration = artifact.get("uplift_calibration", {})
    by_family = artifact.get("uplift_calibration_by_family") or {}
    for index, row in enumerate(rows):
        family = families[index] if families is not None else None
        family_calibration = by_family.get(family) if family else None
        # score head는 light 대비 uplift(Δscore)를 예측할 수 있으므로
        # [-1, 1]로 클립합니다. λ-선택은 에피소드 내 차이만 사용합니다.
        score_row = {}
        for model in MODEL_IDS:
            value = eval_ensemble(score_heads[model], row)
            slope_intercept = None
            if family_calibration is not None:
                slope_intercept = family_calibration.get(model)
            if slope_intercept is None:
                slope_intercept = calibration.get(model)
            if slope_intercept:
                value = slope_intercept[0] * value + slope_intercept[1]
            score_row[model] = min(1.0, max(-1.0, value))
        if spread_heads is None:
            cost_row = {
                model: math.exp(eval_ensemble(cost_heads[model], row))
                * inflation[model]
                for model in MODEL_IDS
            }
        else:
            cost_row = {}
            for model in MODEL_IDS:
                deviation = item_sigma(spread_heads[model], row)
                cost_row[model] = math.exp(
                    eval_ensemble(cost_heads[model], row) + 0.5 * deviation**2
                )
        light = cost_row[MODEL_IDS[0]]
        cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], light * (1.0 + 1e-12))
        cost_row[MODEL_IDS[2]] = max(
            cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1.0 + 1e-12)
        )
        scores.append(score_row)
        costs.append(cost_row)
    return scores, costs


# 잔차 크기 head는 E|r|을 예측합니다. 정규분포에서 E|r| = sigma*sqrt(2/pi)
# 이므로 표준편차로 되돌릴 때 이 상수로 나눕니다.
_MEAN_ABS_TO_SIGMA = 0.7978845608028654
_MAX_SIGMA = 3.0


def item_sigma(head: Mapping[str, Any], row: Sequence[float]) -> float:
    """문항별 log-cost 예측 오차의 표준편차 추정값을 반환합니다."""

    mean_abs = math.exp(eval_ensemble(head, row))
    return min(_MAX_SIGMA, mean_abs / _MEAN_ABS_TO_SIGMA)


def selection_costs(
    artifact: Mapping[str, Any],
    rows: Sequence[Sequence[float]],
    tier: str,
    mean_costs: Sequence[Mapping[str, float]],
) -> List[Dict[str, float]]:
    """선택 제약에 사용할 비용을 계산합니다.

    quantile 비용 head가 있으면 per-item 상위 quantile 예측을 쓰고, 없으면
    mean 예측을 사용합니다. 그 위에 등급별 보정 계수(비관 계수)를 곱합니다.
    """

    spread_heads = artifact.get("log_cost_spread_heads")
    pessimism = (artifact.get("selection_pessimism") or {}).get(tier)
    if spread_heads is not None and pessimism is not None:
        # 문항별 예측 불확실성에 비례해 비용을 올려 잡습니다. 비용이 잘
        # 예측되는 문항에는 공격적으로, 튀는 문항에는 보수적으로 예산을
        # 씁니다. 모델별 상수 계수보다 예산을 효율적인 쪽으로 보냅니다.
        mean_heads = artifact["log_cost_heads"]
        result: List[Dict[str, float]] = []
        for row in rows:
            cost_row = {}
            for model in MODEL_IDS:
                deviation = item_sigma(spread_heads[model], row)
                cost_row[model] = math.exp(
                    eval_ensemble(mean_heads[model], row)
                    + float(pessimism) * deviation
                )
            light = cost_row[MODEL_IDS[0]]
            cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], light * (1.0 + 1e-12))
            cost_row[MODEL_IDS[2]] = max(
                cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1.0 + 1e-12)
            )
            result.append(cost_row)
        return result

    quantile_heads = artifact.get("selection_cost_heads")
    if quantile_heads:
        base: List[Dict[str, float]] = []
        for row_index, row in enumerate(rows):
            cost_row = {
                model: math.exp(eval_ensemble(quantile_heads[model], row))
                for model in MODEL_IDS
            }
            light = max(cost_row[MODEL_IDS[0]], mean_costs[row_index][MODEL_IDS[0]])
            cost_row[MODEL_IDS[0]] = light
            cost_row[MODEL_IDS[1]] = max(cost_row[MODEL_IDS[1]], light * (1.0 + 1e-12))
            cost_row[MODEL_IDS[2]] = max(
                cost_row[MODEL_IDS[2]], cost_row[MODEL_IDS[1]] * (1.0 + 1e-12)
            )
            base.append(cost_row)
    else:
        base = [dict(row) for row in mean_costs]
    inflation = (artifact.get("selection_cost_inflation") or {}).get(tier)
    if inflation:
        base = [
            {model: row[model] * float(inflation[model]) for model in MODEL_IDS}
            for row in base
        ]
    return base


def _selection_cost(
    selection: Sequence[str], costs: Sequence[Mapping[str, float]]
) -> float:
    return sum(costs[i][model] for i, model in enumerate(selection))


def _select_for_lambda(
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    lam: float,
) -> List[str]:
    selection = []
    for score_row, cost_row in zip(scores, costs):
        best_model = None
        best_key = None
        for model in MODEL_IDS:
            key = (score_row[model] - lam * cost_row[model], -cost_row[model])
            if best_key is None or key > best_key:
                best_key = key
                best_model = model
        selection.append(best_model)
    return selection


def select_models(
    scores: Sequence[Mapping[str, float]],
    costs: Sequence[Mapping[str, float]],
    budget: float,
) -> List[str]:
    """Maximize predicted quality subject to predicted cost <= budget."""

    # λ=0에서 이미 예산 안이면 그대로 사용합니다.
    selection = _select_for_lambda(scores, costs, 0.0)
    if _selection_cost(selection, costs) <= budget:
        return selection

    # cost(λ)는 λ에 대해 비증가이므로 예산을 만족하는 최소 λ를 찾습니다.
    lo, hi = 0.0, 1.0
    while _selection_cost(_select_for_lambda(scores, costs, hi), costs) > budget:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if _selection_cost(_select_for_lambda(scores, costs, mid), costs) > budget:
            lo = mid
        else:
            hi = mid
    selection = _select_for_lambda(scores, costs, hi)

    # 남은 예산으로 Δscore/Δcost가 좋은 승격을 greedy하게 적용합니다.
    slack = budget - _selection_cost(selection, costs)
    upgrades = []
    for i, current in enumerate(selection):
        for model in MODEL_IDS:
            delta_score = scores[i][model] - scores[i][current]
            delta_cost = costs[i][model] - costs[i][current]
            if delta_score > 0 and delta_cost > 0:
                upgrades.append((delta_score / delta_cost, delta_score, delta_cost, i, model))
    upgrades.sort(key=lambda item: (-item[0], -item[1]))
    upgraded_cost: Dict[int, Tuple[float, float]] = {}
    for ratio, delta_score, delta_cost, i, model in upgrades:
        base = selection[i]
        # 이미 승격된 문항은 현재 선택 기준으로 다시 계산합니다.
        delta_score = scores[i][model] - scores[i][base]
        delta_cost = costs[i][model] - costs[i][base]
        if delta_score <= 0 or delta_cost <= 0:
            continue
        if delta_cost <= slack:
            selection[i] = model
            slack -= delta_cost
    return selection


def route(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: Mapping[str, Any],
    tier: str,
) -> Submission:
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    hash_bins = int(artifact["hash_bins"])
    texts = [episode_text(episode) for episode in inputs.episodes]
    rows = [extract_vector(episode, hash_bins) for episode in inputs.episodes]
    rows = augment_rows(artifact, rows)
    families = [detect_family(text) for text in texts]
    scores, costs = predict_heads(artifact, rows, families)
    light_total = sum(row[MODEL_IDS[0]] for row in costs)
    multiplier = float(policy.tiers[tier].budget_multiplier)
    safety = float(artifact["tier_safety_ratios"][tier])
    budget = light_total * multiplier * safety
    # 선택 제약에는 quantile/비관 보정 비용을 사용해 비용 과소예측
    # 문항으로의 쏠림(winner's curse)을 완화합니다. 예산 분모는 평균
    # 예측을 유지합니다.
    robust_costs = selection_costs(artifact, rows, tier, costs)
    selection = select_models(scores, robust_costs, budget)
    decisions = tuple(
        Decision(episode.episode_id, model)
        for episode, model in zip(inputs.episodes, selection)
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=decisions,
    )
    return parse_submission(submission_to_dict(submission))


BUNDLED_ARTIFACT_NAME = "gbm-artifact.v1.json"


def _validate_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 artifact 형식입니다.")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ProtocolError("artifact의 feature_version이 코드와 다릅니다.")
    return artifact


def load_artifact(path: Path) -> Mapping[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return _validate_artifact(json.load(handle))


def load_bundled_artifact() -> Mapping[str, Any]:
    from importlib import resources

    text = resources.read_text("ossp_router.resources", BUNDLED_ARTIFACT_NAME)
    return _validate_artifact(json.loads(text))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="학습된 GBM artifact로 한 등급의 선택 결과를 만듭니다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--policy", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        artifact = (
            load_artifact(args.artifact)
            if args.artifact is not None
            else load_bundled_artifact()
        )
        submission = route(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError, KeyError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {args.tier} 제출 파일을 생성했습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
