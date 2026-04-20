import re
from src.v2.corpus.types import CorpusChunk
from src.v2.retrieval.types import LexicalHit

TITLE_WEIGHT = 5
ALIASES_WEIGHT = 1
TEXT_WEIGHT = 0.5


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\.\(\)\_]", "", text.lower())).strip()


def tokenize_text(text: str) -> list[str]:
    return normalize_text(text).split()


def score_chunk_lexically(query_tokens: list[str], chunk: CorpusChunk) -> LexicalHit:
    title_tokens = tokenize_text(chunk["title"])
    text_tokens = tokenize_text(chunk["text"])
    aliases_tokens = [
        token for alias in chunk["aliases"] for token in tokenize_text(alias)
    ]
    title_score = len(set(title_tokens) & set(query_tokens)) * TITLE_WEIGHT
    aliases_score = len(set(aliases_tokens) & set(query_tokens)) * ALIASES_WEIGHT
    text_score = len(set(text_tokens) & set(query_tokens)) * TEXT_WEIGHT
    return {
        "chunk": chunk,
        "score": title_score + aliases_score + text_score,
        "score_breakdown": {
            "title": title_score,
            "aliases": aliases_score,
            "text": text_score,
        },
    }


def retrieve_by_lexical(
    query: str, corpus: list[CorpusChunk], top_k: int
) -> list[LexicalHit]:
    query_tokens = tokenize_text(query)
    lexical_hits = [
        score_chunk_lexically(query_tokens=query_tokens, chunk=chunk)
        for chunk in corpus
    ]
    return sorted(
        [lexical_hit for lexical_hit in lexical_hits if lexical_hit["score"] > 0],
        key=lambda score_chunk: score_chunk["score"],
        reverse=True,
    )[:top_k]
