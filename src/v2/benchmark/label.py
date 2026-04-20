import re

from rapidfuzz import fuzz

from src.v2.benchmark.types import BenchmarkItem, BenchmarkLabel

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


def get_benchmark_label(
    item: BenchmarkItem, answer: str
) -> tuple[BenchmarkLabel, str]:
    required_phrases = item["must_include"]
    forbidden_phrases = item["must_not_include"]
    expected_symbols = item["expected_symbols"]
    normalized_answer = normalize_text(answer)

    for forbidden_phrase in forbidden_phrases:
        if normalize_text(forbidden_phrase) in normalized_answer:
            return "incorrect", f"matched must_not_include: {forbidden_phrase}"

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

    matched_requirement_count = len(exact_requirement_matches) + len(
        fuzzy_requirement_matches
    )
    missing_requirements = [
        requirement
        for requirement in required_phrases
        if normalize_text(requirement) not in exact_requirement_matches
        and normalize_text(requirement) not in fuzzy_requirement_matches
    ]

    matched_expected_symbols = match_expected_symbols(
        expected_symbols=expected_symbols,
        normalized_answer=normalized_answer,
    )

    has_expected_symbol_match = (
        len(expected_symbols) == 0 or len(matched_expected_symbols) > 0
    )
    exact_match_log = ", ".join(exact_requirement_matches) or "none"
    fuzzy_match_log = ", ".join(fuzzy_match_logs) or "none"
    missing_requirement_log = ", ".join(missing_requirements) or "none"
    matched_expected_symbols_log = ", ".join(matched_expected_symbols) or "none"
    missing_expected_symbols = [
        symbol for symbol in expected_symbols if symbol not in matched_expected_symbols
    ]
    missing_expected_symbols_log = ", ".join(missing_expected_symbols) or "none"

    if (
        matched_requirement_count == len(required_phrases)
        and len(required_phrases) > 0
        and has_expected_symbol_match
    ):
        return (
            "correct",
            f"exact matched: {exact_match_log}; fuzzy matched: {fuzzy_match_log}; missing: {missing_requirement_log}; matched expected symbols: {matched_expected_symbols_log}; missed expected symbols: {missing_expected_symbols_log}",
        )
    if matched_requirement_count > 0:
        return (
            "partially_correct",
            f"exact matched: {exact_match_log}; fuzzy matched: {fuzzy_match_log}; missing: {missing_requirement_log}; matched expected symbols: {matched_expected_symbols_log}; missed expected symbols: {missing_expected_symbols_log}",
        )
    return "incorrect", f"matched no must_include: {missing_requirement_log}"
