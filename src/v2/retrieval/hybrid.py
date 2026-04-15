from src.v2.retrieval.symbol import retrieve_by_symbol
from src.v2.retrieval.lexical import retrieve_by_lexical
from src.v2.corpus.types import CorpusChunk, SymbolIndex
from src.v2.retrieval.types import (
    LexicalHit,
    RetrievedDoc,
    RetrievalDebug,
    RetrievalResult,
)


def build_lexical_retrieved_doc(lexical_hit: LexicalHit) -> RetrievedDoc:
    chunk = lexical_hit["chunk"]
    retrieved_doc: RetrievedDoc = {
        "doc_id": chunk["doc_id"],
        "title": chunk["title"],
        "url": chunk["url"],
        "matched_symbols": [],
        "sources": ["lexical"],
        "score_lexical": lexical_hit["score"],
        "score_breakdown": lexical_hit["score_breakdown"],
    }
    return retrieved_doc


def merge_hybrid_docs(
    symbol_docs: list[RetrievedDoc], lexical_hits: list[RetrievedDoc], top_k: int
) -> list[RetrievedDoc]:
    merged_docs = symbol_docs.copy()
    merged_docs_lookup = {doc["doc_id"]: doc for doc in merged_docs}
    for lexical_doc in lexical_hits:
        doc_id = lexical_doc["doc_id"]
        if (
            merged_docs_lookup.get(doc_id)
            and "lexical" not in merged_docs_lookup[doc_id]["sources"]
        ):
            merged_docs_lookup[doc_id]["sources"].append("lexical")
            merged_docs_lookup[doc_id]["score_lexical"] = lexical_doc["score_lexical"]
            merged_docs_lookup[doc_id]["score_breakdown"] = lexical_doc[
                "score_breakdown"
            ]
        elif len(merged_docs) < top_k:
            merged_docs.append(lexical_doc)
            merged_docs_lookup[doc_id] = lexical_doc
    return merged_docs


def build_retrieval_debug(
    symbol_docs: list[RetrievedDoc],
    lexical_hits: list[RetrievedDoc],
    final_docs: list[RetrievedDoc],
) -> RetrievalDebug:
    final_docs_ids = [doc["doc_id"] for doc in final_docs]
    return {
        "symbol_hit_doc_ids": [doc["doc_id"] for doc in symbol_docs],
        "lexical_top_k": [
            {"doc_id": lexical_hit["doc_id"], "score": lexical_hit["score_lexical"]}
            for lexical_hit in lexical_hits
        ],
        "dropped_by_cap": list(
            dict.fromkeys(
                [
                    doc["doc_id"]
                    for doc in symbol_docs
                    if doc["doc_id"] not in final_docs_ids
                ]
                + [
                    lexical_hit["doc_id"]
                    for lexical_hit in lexical_hits
                    if lexical_hit["doc_id"] not in final_docs_ids
                ]
            )
        ),
    }


def retrieve_hybrid(
    query: str,
    corpus: list[CorpusChunk],
    symbol_index: SymbolIndex,
    top_k: int = 6,
    lexical_top_k: int = 10,
) -> RetrievalResult:
    symbol_result = retrieve_by_symbol(
        query=query, corpus=corpus, symbol_index=symbol_index
    )
    symbol_retrieved_docs = symbol_result["retrieved_docs"]
    lexical_result = retrieve_by_lexical(
        query=query, corpus=corpus, top_k=lexical_top_k
    )
    lexical_retrieved_docs = [
        build_lexical_retrieved_doc(lexical_hit=lexical_hit)
        for lexical_hit in lexical_result
    ]
    final_docs = merge_hybrid_docs(
        symbol_docs=symbol_retrieved_docs,
        lexical_hits=lexical_retrieved_docs,
        top_k=top_k,
    )
    return {
        "query": query,
        "retrieved_docs": final_docs,
        "debug": build_retrieval_debug(
            symbol_docs=symbol_retrieved_docs,
            lexical_hits=lexical_retrieved_docs,
            final_docs=final_docs,
        ),
        "matched_symbols": symbol_result["matched_symbols"],
    }
