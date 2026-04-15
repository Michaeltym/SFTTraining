from typing import Any

from src.config import MAX_NEW_TOKENS
from src.rag.inference import run_inference
from src.v2.benchmark.types import BenchmarkAnswer, BenchmarkAnswerFn
from src.v2.retrieval.hybrid import retrieve_hybrid
from src.v2.retrieval.load import load_corpus, load_symbol_index
from src.v2.retrieval.types import RetrievedDoc
from src.v2.corpus.types import CorpusChunk


def build_plain_generation_answer_fn(
    model: Any,
    tokenizer: Any,
) -> BenchmarkAnswerFn:
    def answer_fn(question: str) -> BenchmarkAnswer:
        inputs = tokenizer(question, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )
        return {
            "answer": answer,
            "citations": [],
            "used_symbols": [],
            "abstained": False,
            "confidence_band": "high",
        }

    return answer_fn


def build_rag_answer_fn(model: Any, tokenizer: Any) -> BenchmarkAnswerFn:
    def answer_fn(question: str) -> BenchmarkAnswer:
        result = run_inference(model=model, tokenizer=tokenizer, query=question)
        return {
            "answer": result["answer"],
            "citations": [
                {"title": source["title"], "url": source["url"]}
                for source in result["sources"]
            ],
            "used_symbols": [],
            "abstained": False,
            "confidence_band": "high",
        }

    return answer_fn


def build_hybrid_prompt(
    retrieved_docs: list[RetrievedDoc],
    query: str,
    corpus_lookup: dict[str, CorpusChunk],
) -> str:
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        if doc["doc_id"] in corpus_lookup:
            chunk = corpus_lookup[doc["doc_id"]]
            context_blocks.append(
                "\n".join(
                    [
                        f"Context {i}:",
                        f"Title: {chunk['title']}",
                        f"URL: {chunk['url']}",
                        f"Source Type: {chunk['source_type']}",
                        f"Content: {chunk['text']}",
                    ]
                )
            )

    contexts = (
        "\n\n".join(context_blocks)
        if context_blocks
        else "No supporting context found."
    )
    instructions = [
        "You are a PyTorch API assistant.",
        "Answer the question using the provided context.",
        "If the context is insufficient, say you are not sure.",
        "Reply with only the answer.",
    ]
    return "\n".join(
        instructions
        + [
            "",
            f"Question: {query}",
            "",
            "Context:",
            "",
            contexts,
            "",
            "Answer:",
        ]
    )


def build_hybrid_answer_fn(model: Any, tokenizer: Any) -> BenchmarkAnswerFn:
    corpus = load_corpus()
    symbol_index = load_symbol_index()
    corpus_lookup = {chunk["doc_id"]: chunk for chunk in corpus}

    def answer_fn(question: str) -> BenchmarkAnswer:
        result = retrieve_hybrid(
            query=question, corpus=corpus, symbol_index=symbol_index
        )
        docs = result["retrieved_docs"]
        inputs = tokenizer(
            build_hybrid_prompt(
                retrieved_docs=docs, query=question, corpus_lookup=corpus_lookup
            ),
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],
            skip_special_tokens=True,
        )

        return {
            "answer": answer,
            "citations": [{"title": doc["title"], "url": doc["url"]} for doc in docs],
            "used_symbols": [
                matched["matched_symbol"] for matched in result["matched_symbols"]
            ],
            "abstained": False,
            "confidence_band": "high",
            "retrieval_debug": result["debug"],
        }

    return answer_fn
