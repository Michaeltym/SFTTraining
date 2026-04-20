from typing import Any

from src.config import MAX_NEW_TOKENS
from src.rag.inference import run_inference
from src.v2.benchmark.types import BenchmarkAnswer, BenchmarkAnswerFn
from src.v2.retrieval.hybrid import retrieve_hybrid
from src.v2.retrieval.load import load_corpus, load_symbol_index
from src.v2.retrieval.types import RetrievalResult
from src.v2.corpus.build import extract_symbols
from src.v2.prompts.router import build_hybrid_prompt


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


def build_hybrid_answer_fn(model: Any, tokenizer: Any) -> BenchmarkAnswerFn:
    corpus = load_corpus()
    symbol_index = load_symbol_index()
    corpus_lookup = {chunk["doc_id"]: chunk for chunk in corpus}

    def answer_fn(question: str) -> BenchmarkAnswer:
        result = retrieve_hybrid(
            query=question,
            corpus=corpus,
            symbol_index=symbol_index,
            corpus_lookup=corpus_lookup,
        )
        docs = result["retrieved_docs"]
        prompt_result = build_hybrid_prompt(
            result=result, query=question, corpus_lookup=corpus_lookup
        )
        should_use_refusal = prompt_result["should_use_refusal"]
        inputs = tokenizer(
            prompt_result["prompt"],
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
            "citations": (
                []
                if should_use_refusal
                else [{"title": doc["title"], "url": doc["url"]} for doc in docs]
            ),
            "used_symbols": extract_symbols(answer),
            "abstained": should_use_refusal,
            "confidence_band": "low" if should_use_refusal else "high",
            "retrieval_debug": result["debug"],
        }

    return answer_fn
