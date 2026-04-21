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
    "from_numpy",
    "numpy array",
    "share memory",
]


def should_use_refusal_prompt(
    result: RetrievalResult,
    matched_symbols: list[str],
    unconfirmed_symbols: list[str],
) -> bool:
    retrieved_docs = result["retrieved_docs"]
    debug = result["debug"]
    symbol_hit_doc_ids = debug["symbol_hit_doc_ids"]
    lexical_top_k = debug["lexical_top_k"]

    base_refusal = len(retrieved_docs) == 0 or (
        len(symbol_hit_doc_ids) == 0
        and (
            len(lexical_top_k) == 0
            or lexical_top_k[0]["score"] < HALLUCINATION_REFUSAL_THRESHOLD
        )
    )
    mixed_real_and_fake = len(matched_symbols) > 0 and len(unconfirmed_symbols) > 0
    return base_refusal or mixed_real_and_fake


def should_use_exact_prompt(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(keyword in normalized_query for keyword in EXACT_PROMPT_QUERY_KEYWORDS)


def build_context(
    result: RetrievalResult,
    corpus_lookup: dict[str, CorpusChunk],
    should_use_refusal: bool,
    query_symbols: list[str],
    matched_symbols: list[str],
    unconfirmed_symbols: list[str],
) -> str:
    verification_block = ""
    if query_symbols:
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

    query_symbols = extract_symbols(query)
    matched_symbols: list[str] = []
    unconfirmed_symbols: list[str] = []
    if query_symbols:
        matched_symbols = list(
            dict.fromkeys([s["query_symbol"] for s in result["matched_symbols"]])
        )
        unconfirmed_symbols = [qs for qs in query_symbols if qs not in matched_symbols]
    should_use_refusal = should_use_refusal_prompt(
        result=result,
        matched_symbols=matched_symbols,
        unconfirmed_symbols=unconfirmed_symbols,
    )
    should_use_exact = should_use_exact_prompt(query=query)
    context = build_context(
        result=result,
        corpus_lookup=corpus_lookup,
        should_use_refusal=should_use_refusal,
        query_symbols=query_symbols,
        matched_symbols=matched_symbols,
        unconfirmed_symbols=unconfirmed_symbols,
    )
    if should_use_refusal:
        return {
            "prompt": build_refusal_prompt(
                query=query,
                context=context,
                unconfirmed_symbols=unconfirmed_symbols,
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
            unconfirmed_symbols=unconfirmed_symbols,
        ),
        "should_use_refusal": False,
    }
