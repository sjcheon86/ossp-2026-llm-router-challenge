# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-only feature extraction shared by the GBM trainer and router.

표준 라이브러리만 사용합니다. 문항 ID, 위치, split 같은 메타데이터는 읽지
않고 프롬프트 본문만 사용합니다.
"""

from __future__ import annotations

import ast
import math
import re
from typing import List, Sequence, Tuple

from .heuristic import episode_text
from .protocol import Episode

FEATURE_VERSION = 3
DEFAULT_HASH_BINS = 256

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1

_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[A-Za-z가-힣]+")
_SENTENCE_END = re.compile(r"[.!?。！？]")
_CODE_MARKERS = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include)\b|[{};]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_MARKERS = re.compile(r"[=+\-*/^∑∫√≈≠≤≥<>]|\\(?:frac|sum|int|sqrt)\b")
_LATEX = re.compile(
    r"\\(?:frac|sum|int|sqrt|cdot|times|binom|pmod|mod|equiv|pi|alpha|beta|"
    r"theta|angle|triangle|overline|underline|left|right|begin|end)\b|\$"
)
_REASONING_WORDS = re.compile(
    r"\b(?:prove|derive|reason|analyze|explain why|algorithm|complexity|"
    r"증명|유도|추론|분석|알고리즘|복잡도)\b",
    re.IGNORECASE,
)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)
_MCQ = re.compile(
    r"(?:^|\s)\(?[A-E]\)\s|[①②③④⑤]|다음\s*중|보기|선택지|"
    r"\b(?:which of the following|multiple[- ]choice)\b",
    re.IGNORECASE | re.MULTILINE,
)
_CONTEST_MATH = re.compile(
    r"\b(?:find the (?:number|sum|value|remainder)|remainder when|"
    r"modulo|divisible|integer[s]? satisfying|positive integer|"
    r"나머지|약수|배수|자연수|정수)\b",
    re.IGNORECASE,
)
_STEP_BY_STEP = re.compile(r"step[- ]by[- ]step|단계별", re.IGNORECASE)
_FORMAT_OUTPUT = re.compile(
    r"\b(?:json|yaml|csv|markdown|table|format|형식|표로)\b", re.IGNORECASE
)
_CREATIVE = re.compile(
    r"\b(?:story|poem|essay|novel|creative|시|소설|이야기|수필|에세이)\b",
    re.IGNORECASE,
)
_HANGUL_RANGE = re.compile(r"[가-힣]")
_QUESTION_WORD = re.compile(
    r"\b(?:what|why|how|when|where|who|무엇|왜|어떻게|언제|어디|누구)\b",
    re.IGNORECASE,
)

DENSE_FEATURE_NAMES: Tuple[str, ...] = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "log_line_count",
    "log_max_line_length",
    "avg_word_length",
    "hangul_ratio",
    "ascii_letter_ratio",
    "digit_ratio",
    "symbol_ratio",
    "uppercase_ratio",
    "numeric_density",
    "long_context",
    "log_code_marker_count",
    "log_math_marker_count",
    "log_latex_count",
    "log_reasoning_marker_count",
    "formal_reasoning_count",
    "program_analysis_count",
    "log_multi_constraint_count",
    "simple_transform_count",
    "mcq_marker_count",
    "contest_math_count",
    "step_by_step",
    "format_output_count",
    "creative_count",
    "question_word_count",
    "log_question_mark_count",
    "has_system_message",
    "log_user_message_count",
    "log_assistant_message_count",
    "ends_with_question",
    "code_fence_count",
)


# --- 계산 가능한 구조·난이도 특징 -----------------------------------------
# 어휘(해시 n-gram)가 잡지 못하는 신호를 프롬프트 본문에서 직접 계산합니다.
# 수치 크기, 연산 수, Python AST 복잡도, 사실·규칙 수, 선택지 구조가
# 문항군과 난이도를 나타내며, 특히 모델별 비용(출력 길이) 예측에 쓰입니다.
# 문항 ID·split·출처 같은 메타데이터는 읽지 않습니다.

_NUMBER_LITERAL = re.compile(r"-?\d+(?:\.\d+)?")
_ARITH_OP = re.compile(r"[+\-*/^]")
_LATEX_OP = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|times|binom|pmod)")
_IF_THEN = re.compile(r"\bif\b[^.]*\bthen\b|\bIf\b[^.]*,", re.IGNORECASE)
_FACT_VERB = re.compile(
    r"\b(?:is|are|does not|likes|chases|eats|sees|visits)\b", re.IGNORECASE
)
_OPTION_MARKER = re.compile(
    r"(?:^|\n)\s*(?:\(?[A-E][).]|[①②③④⑤])\s+", re.MULTILINE
)
_CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_DEF_LINE = re.compile(r"^\s*def\s+\w+\s*\(", re.MULTILINE)
_CAPITALIZED = re.compile(r"\b[A-Z][a-z]{2,}\b")
_MATH_OPENER = re.compile(
    r"^\s*(?:What is|Calculate|Solve|Simplify|Differentiate|Round|Let |Suppose|"
    r"Divide|Multiply|Add|Subtract|Sort|Convert|Express|Find)"
)
_ASKS_NUMBER = re.compile(
    r"how many|what is the (?:number|value|sum|remainder)|몇 |얼마", re.IGNORECASE
)
_ASKS_PROOF = re.compile(r"\bprove\b|show that|증명", re.IGNORECASE)

FAMILIES: Tuple[str, ...] = (
    "code",
    "math_synth",
    "mcq",
    "korean_rc",
    "rules",
    "long_context",
    "other",
)

STRUCTURAL_FEATURE_NAMES: Tuple[str, ...] = (
    "log_max_number",
    "log_number_count",
    "max_number_digits",
    "log_total_digits",
    "has_decimal",
    "has_negative",
    "has_fraction",
    "log_arith_op_count",
    "log_latex_op_count",
    "paren_depth",
    "ast_parsed",
    "log_ast_nodes",
    "ast_depth",
    "ast_loops",
    "ast_branches",
    "log_ast_calls",
    "ast_functions",
    "log_fact_count",
    "log_if_then_count",
    "log_entity_count",
    "fact_entity_ratio",
    "option_count",
    "log_option_length",
    "options_numeric",
    "asks_number",
    "asks_proof",
) + tuple(f"family_{name}" for name in FAMILIES)

_STRUCTURAL_HEAD = 6000


def detect_family(text: str) -> str:
    """프롬프트 본문만 보고 문항군을 추정합니다 (정규식 기반 분류기)."""

    if _DEF_LINE.search(text) or "assert f(" in text:
        return "code"
    if len(text) < 400 and _MATH_OPENER.match(text):
        return "math_synth"
    if _OPTION_MARKER.search(text):
        return "mcq"
    if _HANGUL_RANGE.search(text) and len(text) > 300:
        return "korean_rc"
    if len(text) > 20_000:
        return "long_context"
    if text.count("The ") > 8 and _FACT_VERB.search(text):
        return "rules"
    return "other"


def _ast_statistics(text: str) -> Tuple[float, ...]:
    """프롬프트에 포함된 Python 코드를 실제로 파싱해 복잡도를 계산합니다."""

    match = _CODE_BLOCK.search(text)
    if match:
        source = match.group(1)
    elif _DEF_LINE.search(text):
        start = _DEF_LINE.search(text).start()
        end = text.find("assert", start)
        source = text[start : end if end > 0 else len(text)]
    else:
        return (0.0,) * 7
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return (0.0,) * 7
    counters = {"nodes": 0, "loops": 0, "branches": 0, "calls": 0, "functions": 0}

    def walk(node: ast.AST, level: int = 0) -> int:
        counters["nodes"] += 1
        if isinstance(node, (ast.For, ast.While, ast.comprehension)):
            counters["loops"] += 1
        if isinstance(node, (ast.If, ast.IfExp)):
            counters["branches"] += 1
        if isinstance(node, ast.Call):
            counters["calls"] += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            counters["functions"] += 1
        depths = [walk(child, level + 1) for child in ast.iter_child_nodes(node)]
        return max(depths) if depths else level

    try:
        depth = walk(tree)
    except RecursionError:
        return (0.0,) * 7
    return (
        1.0,
        math.log1p(counters["nodes"]),
        float(depth),
        float(counters["loops"]),
        float(counters["branches"]),
        math.log1p(counters["calls"]),
        float(counters["functions"]),
    )


def structural_features(text: str) -> List[float]:
    """수치·구조·문항군 특징을 반환합니다. 표준 라이브러리만 사용합니다."""

    head = text[:_STRUCTURAL_HEAD]
    literals = _NUMBER_LITERAL.findall(head)[:400]
    magnitudes = []
    digit_counts = []
    for literal in literals:
        try:
            magnitudes.append(abs(float(literal)))
        except ValueError:
            continue
        digit_counts.append(len(literal.lstrip("-").replace(".", "")))
    depth = current = 0
    for character in head:
        if character == "(":
            current += 1
            depth = max(depth, current)
        elif character == ")":
            current = max(0, current - 1)
    options = _OPTION_MARKER.split(head)[1:]
    entities = set(_CAPITALIZED.findall(head))
    facts = len(_FACT_VERB.findall(head))
    family = detect_family(text)
    joined_literals = "".join(literals)
    return [
        math.log1p(max(magnitudes) if magnitudes else 0.0),
        math.log1p(len(literals)),
        float(max(digit_counts) if digit_counts else 0),
        math.log1p(sum(digit_counts)),
        1.0 if "." in joined_literals else 0.0,
        1.0 if any(item.startswith("-") for item in literals) else 0.0,
        1.0 if ("/" in head or "\\frac" in head) else 0.0,
        math.log1p(len(_ARITH_OP.findall(head))),
        math.log1p(len(_LATEX_OP.findall(head))),
        float(depth),
        *_ast_statistics(text),
        math.log1p(facts),
        math.log1p(len(_IF_THEN.findall(head))),
        math.log1p(len(entities)),
        facts / max(1, len(entities)),
        float(len(options)),
        math.log1p(sum(len(item) for item in options) / max(1, len(options))),
        1.0
        if options and all(_NUMBER_LITERAL.match(item.strip()) for item in options[:3])
        else 0.0,
        1.0 if _ASKS_NUMBER.search(head) else 0.0,
        1.0 if _ASKS_PROOF.search(head) else 0.0,
    ] + [1.0 if family == name else 0.0 for name in FAMILIES]


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _normalized_tokens(text: str) -> List[str]:
    result = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        if normalized.isdecimal():
            normalized = "<number>"
        result.append(normalized)
    return result


def feature_names(hash_bins: int) -> Tuple[str, ...]:
    return (
        DENSE_FEATURE_NAMES
        + STRUCTURAL_FEATURE_NAMES
        + tuple(f"hash_bin_{index}" for index in range(hash_bins))
    )


def extract_vector(
    episode: Episode, hash_bins: int = DEFAULT_HASH_BINS
) -> List[float]:
    """Return the dense + signed hashed-ngram feature vector for one episode."""

    text = episode_text(episode)
    characters = len(text)
    nonspace = sum(not ch.isspace() for ch in text) or 1
    hangul = sum("가" <= ch <= "힣" for ch in text)
    ascii_letters = sum(("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text)
    uppercase = sum("A" <= ch <= "Z" for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    symbols = sum(
        (not ch.isalnum()) and (not ch.isspace()) and not ("가" <= ch <= "힣")
        for ch in text
    )
    lines = text.splitlines() or [""]
    words = _WORD.findall(text)
    word_count = len(words)
    total_word_len = sum(len(word) for word in words)

    if episode.prompt is not None:
        message_count = 1
        has_system = 0.0
        user_messages = 1
        assistant_messages = 0
    else:
        messages = episode.messages or ()
        message_count = len(messages)
        has_system = 1.0 if any(m.role == "system" for m in messages) else 0.0
        user_messages = sum(m.role == "user" for m in messages)
        assistant_messages = sum(m.role == "assistant" for m in messages)

    stripped = text.rstrip()
    dense = [
        math.log1p(characters),
        math.log1p(word_count),
        math.log1p(max(1, len(_SENTENCE_END.findall(text)))),
        math.log1p(message_count),
        math.log1p(len(lines)),
        math.log1p(max(len(line) for line in lines)),
        total_word_len / max(1, word_count),
        hangul / nonspace,
        ascii_letters / nonspace,
        digits / nonspace,
        symbols / nonspace,
        uppercase / max(1, ascii_letters),
        digits / nonspace,
        1.0 if characters >= 8_000 else 0.0,
        math.log1p(len(_CODE_MARKERS.findall(text))),
        math.log1p(len(_MATH_MARKERS.findall(text))),
        math.log1p(len(_LATEX.findall(text))),
        math.log1p(len(_REASONING_WORDS.findall(text))),
        float(len(_FORMAL_REASONING.findall(text))),
        float(len(_PROGRAM_ANALYSIS.findall(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(len(_SIMPLE_TRANSFORM.findall(text))),
        float(len(_MCQ.findall(text))),
        float(len(_CONTEST_MATH.findall(text))),
        1.0 if _STEP_BY_STEP.search(text) else 0.0,
        float(len(_FORMAT_OUTPUT.findall(text))),
        float(len(_CREATIVE.findall(text))),
        float(len(_QUESTION_WORD.findall(text))),
        math.log1p(text.count("?") + text.count("？")),
        has_system,
        math.log1p(user_messages),
        math.log1p(assistant_messages),
        1.0 if stripped.endswith(("?", "？")) else 0.0,
        float(text.count("```")),
    ]

    bins = [0.0] * hash_bins
    tokens = _normalized_tokens(text)
    grams = tokens + [
        f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)
    ]
    for gram in grams:
        digest = _stable_hash(gram)
        index = digest % hash_bins
        sign = 1.0 if (digest >> 63) & 1 else -1.0
        bins[index] += sign
    norm = math.sqrt(sum(value * value for value in bins))
    if norm > 0:
        bins = [value / norm for value in bins]
    return dense + structural_features(text) + bins


def extract_matrix(
    episodes: Sequence[Episode], hash_bins: int = DEFAULT_HASH_BINS
) -> List[List[float]]:
    return [extract_vector(episode, hash_bins) for episode in episodes]
