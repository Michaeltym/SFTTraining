import re

from rapidfuzz import fuzz

from src.v2.benchmark.types import BenchmarkItem, BenchmarkLabel

NATURAL_LANGUAGE_PHRASE_PATTERN = r"^[a-z]+(?:\s+[a-z]+)+$"
FUZZY_MATCH_SCORE_THRESHOLD = 90


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def should_use_fuzzy_match(text: str) -> bool:
    return bool(re.search(NATURAL_LANGUAGE_PHRASE_PATTERN, normalize_text(text)))


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


def get_benchmark_label(item: BenchmarkItem, answer: str) -> tuple[BenchmarkLabel, str]:
    required_phrases = item["must_include"]
    forbidden_phrases = item["must_not_include"]
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

    exact_match_log = ", ".join(exact_requirement_matches) or "none"
    fuzzy_match_log = ", ".join(fuzzy_match_logs) or "none"
    missing_requirement_log = ", ".join(missing_requirements) or "none"

    if matched_requirement_count == len(required_phrases) and len(required_phrases) > 0:
        return (
            "correct",
            f"exact matched: {exact_match_log}; fuzzy matched: {fuzzy_match_log}; missing: {missing_requirement_log}",
        )
    if matched_requirement_count > 0:
        return (
            "partially_correct",
            f"exact matched: {exact_match_log}; fuzzy matched: {fuzzy_match_log}; missing: {missing_requirement_log}",
        )
    return "incorrect", f"matched no must_include: {missing_requirement_log}"
