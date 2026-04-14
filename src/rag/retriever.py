from collections import Counter
from typing import TypedDict
import re
from src.config import (
    RAG_RETRIEVAL_COMPARISON_BONUS,
    RAG_RETRIEVAL_TAG_SYMBOL_BASE,
    RAG_RETRIEVAL_TAG_SYMBOL_STEP,
    RAG_RETRIEVAL_TAG_TOKEN_WEIGHT,
    RAG_RETRIEVAL_TEXT_SYMBOL_BASE,
    RAG_RETRIEVAL_TEXT_SYMBOL_STEP,
    RAG_RETRIEVAL_TEXT_TOKEN_WEIGHT,
    RAG_RETRIEVAL_TITLE_SYMBOL_BASE,
    RAG_RETRIEVAL_TITLE_SYMBOL_STEP,
    RAG_RETRIEVAL_TITLE_TOKEN_WEIGHT,
)
from src.rag.knowledge import KnowledgeItem


class RetrievedKnowledge(TypedDict):
    knowledge: KnowledgeItem
    weight: float


def extract_normalized_symbols(text: str) -> set[str]:
    symbols = set(re.findall(r"\b\w+\.\w+(?:\(\))?\b", text.lower()))
    return {s[:-2] if s.endswith("()") else s for s in symbols}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\.]+", text.lower())


def score_symbol_overlap(overlap_count: int, base_weight: int, step_weight: int) -> int:
    return overlap_count * (base_weight + overlap_count * step_weight)


def score_lexical_overlap(
    query_words: set[str],
    title_counter: Counter[str],
    tags_counter: Counter[str],
    text_counter: Counter[str],
) -> int:
    score = 0
    for word in query_words:
        score += (
            title_counter[word] * RAG_RETRIEVAL_TITLE_TOKEN_WEIGHT
            + text_counter[word] * RAG_RETRIEVAL_TEXT_TOKEN_WEIGHT
            + tags_counter[word] * RAG_RETRIEVAL_TAG_TOKEN_WEIGHT
        )
    return score


def score_knowledge_item(
    query_symbols: set[str],
    query_words: set[str],
    knowledge_item: KnowledgeItem,
) -> int:
    lowercased_tags = [tag.lower() for tag in knowledge_item["tags"]]
    title_counter = Counter(tokenize(knowledge_item["title"]))
    tags_counter = Counter(lowercased_tags)
    text_counter = Counter(tokenize(knowledge_item["text"]))

    title_symbols = extract_normalized_symbols(knowledge_item["title"])
    tags_symbols = extract_normalized_symbols(" ".join(lowercased_tags))
    text_symbols = extract_normalized_symbols(knowledge_item["text"])

    title_query_overlap = len(title_symbols & query_symbols)
    tags_query_overlap = len(tags_symbols & query_symbols)
    text_query_overlap = len(text_symbols & query_symbols)

    score = (
        score_symbol_overlap(
            title_query_overlap,
            RAG_RETRIEVAL_TITLE_SYMBOL_BASE,
            RAG_RETRIEVAL_TITLE_SYMBOL_STEP,
        )
        + score_symbol_overlap(
            tags_query_overlap,
            RAG_RETRIEVAL_TAG_SYMBOL_BASE,
            RAG_RETRIEVAL_TAG_SYMBOL_STEP,
        )
        + score_symbol_overlap(
            text_query_overlap,
            RAG_RETRIEVAL_TEXT_SYMBOL_BASE,
            RAG_RETRIEVAL_TEXT_SYMBOL_STEP,
        )
    )

    if len(query_symbols) >= 2:
        matched_query_symbols = (
            title_symbols | tags_symbols | text_symbols
        ) & query_symbols
        if len(matched_query_symbols) >= 2:
            score += len(matched_query_symbols) * RAG_RETRIEVAL_COMPARISON_BONUS

    score += score_lexical_overlap(
        query_words=query_words,
        title_counter=title_counter,
        tags_counter=tags_counter,
        text_counter=text_counter,
    )
    return score


def retrieve_top_k_knowledge(
    query: str, knowledge_items: list[KnowledgeItem], top_k: int
) -> list[RetrievedKnowledge]:
    query_symbols = extract_normalized_symbols(query)
    query_words = set(tokenize(query))
    knowledge_with_weight: list[RetrievedKnowledge] = []

    if not query_words:
        return []

    for knowledge_item in knowledge_items:
        knowledge_with_weight.append(
            {
                "knowledge": knowledge_item,
                "weight": score_knowledge_item(
                    query_symbols=query_symbols,
                    query_words=query_words,
                    knowledge_item=knowledge_item,
                ),
            }
        )
    return sorted(knowledge_with_weight, key=lambda x: x["weight"], reverse=True)[
        :top_k
    ]
