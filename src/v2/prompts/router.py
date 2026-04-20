from src.config import HALLUCINATION_REFUSAL_THRESHOLD
from src.v2.retrieval.types import RetrievalResult
from src.v2.corpus.types import CorpusChunk
from src.v2.prompts.refusal import build_refusal_prompt
from src.v2.prompts.reasoning import build_reasoning_prompt
from src.v2.prompts.exact import build_exact_prompt
from src.v2.benchmark.label import normalize_text
from src.v2.prompts.types import BuildPromptResult
from src.v2.corpus.build import extract_symbols

EXACT_PROMPT_QUERY_KEYWORDS = [
    "pin_memory",
    "requires_grad",
    "shape",
    "dtype",
    "device",
    "shuffle",
    "one-hot",
    "class targets",
    "class target",
    "state_dict",
    "collate_fn",
    "validation",
    "torch.cat",
    "non-concatenated",
    "concatenation dimension",
    "same size",
]


def should_use_refusal_prompt(result: RetrievalResult) -> bool:
    retrieved_docs = result["retrieved_docs"]
    debug = result["debug"]
    symbol_hit_doc_ids = debug["symbol_hit_doc_ids"]
    lexical_top_k = debug["lexical_top_k"]

    return len(retrieved_docs) == 0 or (
        len(symbol_hit_doc_ids) == 0
        and (
            len(lexical_top_k) == 0
            or lexical_top_k[0]["score"] < HALLUCINATION_REFUSAL_THRESHOLD
        )
    )


def should_use_exact_prompt(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(keyword in normalized_query for keyword in EXACT_PROMPT_QUERY_KEYWORDS)


def build_context(
    result: RetrievalResult,
    corpus_lookup: dict[str, CorpusChunk],
    query: str,
    should_use_refusal: bool,
) -> str:
    query_symbols = extract_symbols(query)
    verification_block = ""
    if query_symbols:
        matched_symbols = list(
            dict.fromkeys([s["query_symbol"] for s in result["matched_symbols"]])
        )
        unconfirmed_symbols = [qs for qs in query_symbols if qs not in matched_symbols]
        verification_block = "\n".join(
            [
                "Query symbol verification:",
                f"- Confirmed in retrieved PyTorch docs: {', '.join(matched_symbols) if matched_symbols else 'None'}",
                f"- Not confirmed in retrieved PyTorch docs: {', '.join(unconfirmed_symbols) if unconfirmed_symbols else 'None'}",
            ]
        )
    if should_use_refusal:
        return verification_block or "No supporting context found."
    context_blocks = [verification_block] if query_symbols else []

    for index, doc in enumerate(result["retrieved_docs"], start=1):
        if doc["doc_id"] in corpus_lookup:
            chunk = corpus_lookup[doc["doc_id"]]
            context_blocks.append(
                "\n".join(
                    [
                        f"Fact {index}",
                        f"API: {chunk['title']}",
                        f"Details: {' '.join(chunk['text'].split()[:150])}",
                    ]
                )
            )

    context = (
        "\n\n".join(context_blocks)
        if context_blocks
        else "No supporting context found."
    )
    return context


def build_hybrid_prompt(
    result: RetrievalResult, query: str, corpus_lookup: dict[str, CorpusChunk]
) -> BuildPromptResult:

    should_use_refusal = should_use_refusal_prompt(result=result)
    should_use_exact = should_use_exact_prompt(query=query)
    context = build_context(
        result=result,
        corpus_lookup=corpus_lookup,
        query=query,
        should_use_refusal=should_use_refusal,
    )
    if should_use_refusal:
        return {
            "prompt": build_refusal_prompt(
                query=query,
                context=context,
            ),
            "should_use_refusal": should_use_refusal,
        }
    if should_use_exact:
        return {
            "prompt": build_exact_prompt(
                query=query,
                context=context,
            ),
            "should_use_refusal": False,
        }

    return {
        "prompt": build_reasoning_prompt(
            query=query,
            context=context,
        ),
        "should_use_refusal": False,
    }
