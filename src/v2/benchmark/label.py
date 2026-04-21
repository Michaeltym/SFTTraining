import re

from rapidfuzz import fuzz
from typing import Any

from src.v2.benchmark.types import BenchmarkItem, BenchmarkLabel, BenchmarkLabelNotes

NATURAL_LANGUAGE_PHRASE_PATTERN = r"^[a-z]+(?:\s+[a-z]+)+$"
FUZZY_MATCH_SCORE_THRESHOLD = 90


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def normalize_symbol(symbol: str) -> str:
    return normalize_text(symbol)


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-z_][a-z0-9_]*", normalize_text(text))


def contains_with_boundary(text: str, term: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text))


def should_use_fuzzy_match(text: str) -> bool:
    return bool(re.search(NATURAL_LANGUAGE_PHRASE_PATTERN, normalize_text(text)))


def tail_match_in_answer(
    tail: str, normalized_answer: str, answer_tokens: set[str]
) -> bool:
    """Check if a symbol's tail part appears in the answer as an API reference.

    Two-tier matching based on whether the tail contains underscores:

    Safe tails (contain underscores, e.g. __len__, zero_grad, from_numpy):
      Almost certainly API-specific names, so a bare token match is enough.

    Ambiguous tails (no underscores, e.g. tensor, cat, view, to, train):
      Could be common English words. Only accept code-style references
      to confirm the answer is using them as API names:
        - .tail    (dot-prefixed, e.g. ".view", ".backward")
        - tail()   (function call, e.g. "backward()", "eval()")
        - `tail`   (backtick-quoted, e.g. "`reshape`", "`view`")
      Bare token matches like "a new tensor" or "during training" are
      rejected to avoid false positives.
    """
    if "_" in tail:
        return tail in answer_tokens
    return (
        f".{tail}" in normalized_answer
        or f"{tail}()" in normalized_answer
        or f"{tail}(" in normalized_answer
        or f"`{tail}`" in normalized_answer
    )


def match_expected_symbols(
    expected_symbols: list[str],
    normalized_answer: str,
) -> list[str]:
    answer_tokens = set(tokenize_text(normalized_answer))
    matched = []

    for symbol in expected_symbols:
        normalized_symbol = normalize_symbol(symbol)
        normalized_symbol_tail = normalized_symbol.split(".")[-1]
        # Full match: the complete symbol (e.g. "torch.tensor") appears
        # in the answer with word boundaries.
        full_match = contains_with_boundary(normalized_answer, normalized_symbol)
        # Tail match: only the last part of the symbol appears, with
        # stricter rules for common-word tails to avoid false positives.
        tail = tail_match_in_answer(
            tail=normalized_symbol_tail,
            normalized_answer=normalized_answer,
            answer_tokens=answer_tokens,
        )
        if full_match or tail:
            matched.append(symbol)

    return matched


def fuzzy_match_requirement(
    requirement: str, normalized_answer: str
) -> tuple[bool, str]:
    answer_words = normalized_answer.split(" ")
    total_answer_words = len(answer_words)
    total_requirement_words = len(requirement.split(" "))

    for i in range(total_answer_words - total_requirement_words + 1):
        answer_span = " ".join(answer_words[i : i + total_requirement_words])
        if fuzz.ratio(requirement, answer_span) >= FUZZY_MATCH_SCORE_THRESHOLD:
            return True, answer_span
    return False, ""


def get_matched_forbidden_phrases(
    forbidden_phrases: list[str], normalized_answer: str
) -> list[str]:
    return [
        forbidden_phrase
        for forbidden_phrase in forbidden_phrases
        if normalize_text(forbidden_phrase) in normalized_answer
    ]


def get_matched_forbidden_regex(
    regex_patterns: list[str], normalized_answer: str
) -> list[str]:
    """Match forbidden regex patterns against the normalized answer.

    Patterns are matched against ``normalize_text()`` output, NOT the raw answer:
      - text is lowercased and any whitespace run is collapsed to a single space
      - leading / trailing whitespace is stripped
      - punctuation (``.``, ``[``, ``]``, backticks, commas, etc.) is preserved

    Write patterns for that normalized form. In particular:
      - do not rely on raw newlines, tabs, or multi-space runs -- they all collapse to one space
      - patterns are typically written lowercase; ``re.IGNORECASE`` is applied as a safety net
      - standard regex escapes still apply, e.g. ``torch\\.tensor`` to match the literal dot
    """
    return [
        regex_pattern
        for regex_pattern in regex_patterns
        if re.search(regex_pattern, normalized_answer, re.IGNORECASE)
    ]


def match_required_phrases(
    required_phrases: list[str], normalized_answer: str
) -> dict[str, Any]:
    exact_requirement_matches: list[str] = []
    fuzzy_requirement_matches: list[str] = []
    fuzzy_match_logs: list[str] = []

    for required_phrase in required_phrases:
        normalized_requirement = normalize_text(required_phrase)
        if should_use_fuzzy_match(normalized_requirement):
            matched, matched_answer_span = fuzzy_match_requirement(
                requirement=normalized_requirement,
                normalized_answer=normalized_answer,
            )
            if matched:
                fuzzy_requirement_matches.append(normalized_requirement)
                fuzzy_match_logs.append(
                    f"{matched_answer_span} ~= {normalized_requirement}"
                )
            continue

        if normalized_requirement in normalized_answer:
            exact_requirement_matches.append(normalized_requirement)

    missing_requirements = [
        requirement
        for requirement in required_phrases
        if normalize_text(requirement) not in exact_requirement_matches
        and normalize_text(requirement) not in fuzzy_requirement_matches
    ]

    return {
        "exact_matches": exact_requirement_matches,
        "fuzzy_matches": fuzzy_requirement_matches,
        "fuzzy_logs": fuzzy_match_logs,
        "missing": missing_requirements,
        "matched_count": len(exact_requirement_matches)
        + len(fuzzy_requirement_matches),
    }


def get_existing_any_of_group_match(
    matched_any_of_requirement_groups: list[dict[str, Any]], group: list[str]
) -> dict[str, Any] | None:
    return next(
        (d for d in matched_any_of_requirement_groups if d["group"] == group),
        None,
    )


def match_any_of_groups(
    required_any_of_groups: list[list[str]], normalized_answer: str
) -> dict[str, Any]:
    matched_any_of_requirement_groups: list[dict[str, Any]] = []

    for group in required_any_of_groups:
        for phrase in group:
            normalized_requirement = normalize_text(phrase)
            matched = False

            if should_use_fuzzy_match(normalized_requirement):
                matched, _ = fuzzy_match_requirement(
                    requirement=normalized_requirement,
                    normalized_answer=normalized_answer,
                )
            else:
                matched = normalized_requirement in normalized_answer

            if not matched:
                continue

            existing_group = get_existing_any_of_group_match(
                matched_any_of_requirement_groups=matched_any_of_requirement_groups,
                group=group,
            )
            if existing_group:
                existing_group["matched"].append(phrase)
            else:
                matched_any_of_requirement_groups.append(
                    {"group": group, "matched": [phrase]}
                )

    missing_any_of_groups = [
        group
        for group in required_any_of_groups
        if not any(
            match["group"] == group for match in matched_any_of_requirement_groups
        )
    ]

    return {
        "matched_groups": matched_any_of_requirement_groups,
        "missing_groups": missing_any_of_groups,
    }


def build_label_notes(
    phrase_match_result: dict[str, Any],
    any_of_match_result: dict[str, Any],
    matched_expected_symbols: list[str],
    expected_symbols: list[str],
    matched_forbidden_phrases: list[str],
    matched_forbidden_regex: list[str],
) -> BenchmarkLabelNotes:
    missing_expected_symbols = [
        symbol for symbol in expected_symbols if symbol not in matched_expected_symbols
    ]
    return {
        "must_include": {
            "exact_matches": phrase_match_result["exact_matches"],
            "fuzzy_matches": phrase_match_result["fuzzy_matches"],
            "fuzzy_logs": phrase_match_result["fuzzy_logs"],
            "missing": phrase_match_result["missing"],
        },
        "must_include_any_of": {
            "matched_groups": any_of_match_result["matched_groups"],
            "missing_groups": any_of_match_result["missing_groups"],
        },
        "must_not_include": {
            "matched_phrases": matched_forbidden_phrases,
            "matched_regex": matched_forbidden_regex,
        },
        "expected_symbols": {
            "matched": matched_expected_symbols,
            "missing": missing_expected_symbols,
        },
    }


def get_benchmark_label(
    item: BenchmarkItem, answer: str
) -> tuple[BenchmarkLabel, BenchmarkLabelNotes]:
    required_phrases = item["must_include"]
    required_any_of_groups = item.get("must_include_any_of") or []
    must_not_include_regex = item.get("must_not_include_regex") or []
    forbidden_phrases = item["must_not_include"]
    expected_symbols = item["expected_symbols"]
    normalized_answer = normalize_text(answer)

    matched_forbidden_phrases = get_matched_forbidden_phrases(
        forbidden_phrases=forbidden_phrases,
        normalized_answer=normalized_answer,
    )
    matched_forbidden_regex = get_matched_forbidden_regex(
        regex_patterns=must_not_include_regex,
        normalized_answer=normalized_answer,
    )

    phrase_match_result = match_required_phrases(
        required_phrases=required_phrases,
        normalized_answer=normalized_answer,
    )
    any_of_match_result = match_any_of_groups(
        required_any_of_groups=required_any_of_groups,
        normalized_answer=normalized_answer,
    )

    matched_expected_symbols = match_expected_symbols(
        expected_symbols=expected_symbols,
        normalized_answer=normalized_answer,
    )

    has_expected_symbol_match = (
        len(expected_symbols) == 0 or len(matched_expected_symbols) > 0
    )
    notes = build_label_notes(
        phrase_match_result=phrase_match_result,
        any_of_match_result=any_of_match_result,
        matched_expected_symbols=matched_expected_symbols,
        expected_symbols=expected_symbols,
        matched_forbidden_phrases=matched_forbidden_phrases,
        matched_forbidden_regex=matched_forbidden_regex,
    )

    # must_not_include (keyword) is a hard rule: any hit forces incorrect.
    # This preserves the historical semantics of the keyword layer and keeps
    # must_not_include and must_not_include_regex as two distinct severity tiers:
    #   - must_not_include        -> hard forbidden, hit => incorrect
    #   - must_not_include_regex  -> soft forbidden, hit => at most partially_correct
    if len(matched_forbidden_phrases) > 0:
        return "incorrect", notes

    if (
        (
            (
                len(required_phrases) > 0
                and phrase_match_result["matched_count"] == len(required_phrases)
            )
            or len(required_phrases) == 0
        )
        and len(required_any_of_groups) == len(any_of_match_result["matched_groups"])
        and has_expected_symbol_match
        and len(matched_forbidden_regex) == 0
    ):
        return "correct", notes
    if (
        phrase_match_result["matched_count"] > 0
        or len(any_of_match_result["matched_groups"]) > 0
        or len(matched_forbidden_regex) > 0
    ):
        return "partially_correct", notes
    return "incorrect", notes
