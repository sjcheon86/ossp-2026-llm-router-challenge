# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Prompt-only feature extraction shared by the GBM trainer and router.

표준 라이브러리만 사용합니다. 문항 ID, 위치, split 같은 메타데이터는 읽지
않고 프롬프트 본문만 사용합니다.
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

from .heuristic import episode_text
from .protocol import Episode

FEATURE_VERSION = 2
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
    return DENSE_FEATURE_NAMES + tuple(
        f"hash_bin_{index}" for index in range(hash_bins)
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
    return dense + bins


def extract_matrix(
    episodes: Sequence[Episode], hash_bins: int = DEFAULT_HASH_BINS
) -> List[List[float]]:
    return [extract_vector(episode, hash_bins) for episode in episodes]
