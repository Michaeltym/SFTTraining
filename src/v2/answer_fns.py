from typing import Any

from src.config import MAX_NEW_TOKENS
from src.rag.inference import run_inference
from src.v2.benchmark.types import BenchmarkAnswer, BenchmarkAnswerFn


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
