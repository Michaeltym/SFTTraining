from collections import Counter
from typing import TypedDict
import re
from src.rag.knowledge import KnowledgeItem


class RetrievedKnowledge(TypedDict):
    knowledge: KnowledgeItem
    weight: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_\.]+", text.lower())


def retrieve_top_k_knowledge(
    query: str, knowledge_items: list[KnowledgeItem], top_k: int
) -> list[RetrievedKnowledge]:
    query_words = set(tokenize(query))
    knowledge_with_weight: list[RetrievedKnowledge] = []

    if not query_words:
        return []

    for k in knowledge_items:
        title_counter = Counter(tokenize(k["title"]))
        tags_counter = Counter(tag.lower() for tag in k["tags"])
        text_counter = Counter(tokenize(k["text"]))

        knowledge_weight = 0
        for w in query_words:
            knowledge_weight += (
                title_counter[w] * 3 + text_counter[w] * 2 + tags_counter[w] * 5
            )
        knowledge_with_weight.append(
            {"knowledge": k, "weight": (knowledge_weight / len(query_words))}
        )
    return sorted(knowledge_with_weight, key=lambda x: x["weight"], reverse=True)[
        :top_k
    ]
