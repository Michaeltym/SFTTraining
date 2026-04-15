from src.v2.corpus.build import extract_symbols, build_aliases
from src.v2.retrieval.load import load_corpus, load_symbol_index
from src.v2.corpus.types import (
    CorpusChunk,
    SymbolIndex,
)
from src.v2.retrieval.types import (
    RetrievalResult,
    MatchedSymbol,
    SymbolMatchedDoc,
    RetrievalDebug,
)


def extract_query_symbols(query: str) -> list[str]:
    return extract_symbols(query)


def append_unique_match(
    matches: list[MatchedSymbol], new_matches: list[MatchedSymbol]
) -> list[MatchedSymbol]:
    updated_matches = matches.copy()
    for match in new_matches:
        if match not in updated_matches:
            updated_matches.append(match)
    return updated_matches


def build_exact_match(symbol: str) -> MatchedSymbol:
    return {
        "query_symbol": symbol,
        "matched_symbol": symbol,
        "match_type": "exact",
    }


def build_alias_match(query_symbol: str, matched_symbol: str) -> MatchedSymbol:
    return {
        "query_symbol": query_symbol,
        "matched_symbol": matched_symbol,
        "match_type": "alias",
    }


def upsert_doc_match(
    docs: dict[str, SymbolMatchedDoc],
    doc_id: str,
    new_matches: list[MatchedSymbol],
) -> None:
    if doc_id not in docs:
        docs[doc_id] = {
            "doc_id": doc_id,
            "matched_symbols": new_matches.copy(),
        }
        return

    docs[doc_id]["matched_symbols"] = append_unique_match(
        matches=docs[doc_id]["matched_symbols"],
        new_matches=new_matches,
    )


def lookup_symbol_matches(
    query_symbols: list[str], symbol_index: SymbolIndex
) -> list[SymbolMatchedDoc]:
    docs: dict[str, SymbolMatchedDoc] = {}
    for query_symbol in query_symbols:
        exact_entry = symbol_index.get(query_symbol)
        if exact_entry:
            exact_match = build_exact_match(symbol=query_symbol)
            for doc_id in exact_entry["doc_ids"]:
                upsert_doc_match(
                    docs=docs,
                    doc_id=doc_id,
                    new_matches=[exact_match],
                )
            continue

        query_aliases = build_aliases(symbols=[query_symbol])
        for symbol_entry in symbol_index.values():
            matched_aliases = list(set(symbol_entry["aliases"]) & set(query_aliases))
            if not matched_aliases:
                continue

            alias_match = build_alias_match(
                query_symbol=query_symbol,
                matched_symbol=symbol_entry["symbol"],
            )
            for doc_id in symbol_entry["doc_ids"]:
                upsert_doc_match(
                    docs=docs,
                    doc_id=doc_id,
                    new_matches=[alias_match],
                )
    return [item for _, item in docs.items()]


def retrieve_by_symbol(
    query: str, corpus: list[CorpusChunk], symbol_index: SymbolIndex
) -> RetrievalResult:
    debug: RetrievalDebug = {
        "symbol_hit_doc_ids": [],
        "lexical_top_k": [],
        "dropped_by_cap": [],
    }
    result: RetrievalResult = {
        "query": query,
        "matched_symbols": [],
        "retrieved_docs": [],
        "debug": debug,
    }
    query_symbols = extract_query_symbols(query)
    matched_docs = lookup_symbol_matches(
        query_symbols=query_symbols,
        symbol_index=symbol_index,
    )
    chunks_by_id = {chunk["doc_id"]: chunk for chunk in corpus}
    for matched_doc in matched_docs:
        matched_symbols = matched_doc["matched_symbols"]
        doc_id = matched_doc["doc_id"]
        if doc_id not in result["debug"]["symbol_hit_doc_ids"]:
            result["debug"]["symbol_hit_doc_ids"].append(doc_id)
        if doc_id in chunks_by_id:
            chunk = chunks_by_id[doc_id]
            result["retrieved_docs"].append(
                {
                    **matched_doc,
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "sources": ["symbol"],
                    "score_lexical": 0.0,
                    "score_breakdown": {},
                }
            )
            result["matched_symbols"] = append_unique_match(
                matches=result["matched_symbols"], new_matches=matched_symbols
            )

    return result


if __name__ == "__main__":
    query = "what's is Tensor.backward"
    corpus = load_corpus()
    symbol_index = load_symbol_index()
    result = retrieve_by_symbol(query=query, corpus=corpus, symbol_index=symbol_index)
    print(result)
