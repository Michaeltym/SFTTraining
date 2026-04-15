from typing import Literal, TypedDict
from src.v2.corpus.types import CorpusChunk

MatchType = Literal["exact", "alias"]


class MatchedSymbol(TypedDict):
    query_symbol: str
    matched_symbol: str
    match_type: MatchType


class RetrievedDoc(TypedDict):
    doc_id: str
    title: str
    url: str
    matched_symbols: list[MatchedSymbol]
    sources: list[Literal["symbol", "lexical"]]
    score_lexical: float
    score_breakdown: dict[str, float]


class SymbolMatchedDoc(TypedDict):
    doc_id: str
    matched_symbols: list[MatchedSymbol]


class LexicalDebugHit(TypedDict):
    doc_id: str
    score: float


class RetrievalDebug(TypedDict):
    symbol_hit_doc_ids: list[str]
    lexical_top_k: list[LexicalDebugHit]
    dropped_by_cap: list[str]


class RetrievalResult(TypedDict):
    query: str
    matched_symbols: list[MatchedSymbol]
    retrieved_docs: list[RetrievedDoc]
    debug: RetrievalDebug


class LexicalHit(TypedDict):
    chunk: CorpusChunk
    score: float
    score_breakdown: dict[str, float]
