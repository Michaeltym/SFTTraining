from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import BENCHMARK_DATA_PATH, BENCHMARK_RESULTS_DIR, MAX_NEW_TOKENS
from src.model import get_model_name_slug
from src.tokenizer import AutoTokenizer
from src.v2.benchmark.data import load_benchmark_data
from src.v2.benchmark.label import get_benchmark_label
from src.v2.benchmark.save import save_benchmark_results
from src.v2.benchmark.summary import build_benchmark_summary
from src.v2.benchmark.types import (
    BenchmarkResultItem,
)


def run_benchmark(
    model: Any, tokenizer: AutoTokenizer, system_name: str, model_name: str, mode: str
) -> None:
    print(f"##### Run {system_name} benchmark #####")
    results: list[BenchmarkResultItem] = []
    benchmark_items = load_benchmark_data(file_path=str(BENCHMARK_DATA_PATH))
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    for benchmark_item in benchmark_items:
        answer_text = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=benchmark_item["question"],
        )
        label, notes = get_benchmark_label(item=benchmark_item, answer=answer_text)
        results.append(
            {
                **benchmark_item,
                "answer": answer_text,
                "citations": [],
                "used_symbols": [],
                "abstained": False,
                "confidence_band": "high",
                "label": label,
                "notes": notes,
            }
        )

    summary = build_benchmark_summary(results=results)
    model_name_slug = get_model_name_slug(model_name)
    output_path = (
        Path(BENCHMARK_RESULTS_DIR)
        / f"{mode}-{system_name}-{model_name_slug}-{timestamp}.json"
    )
    save_benchmark_results(
        results=results,
        path=output_path,
        model_name=model_name,
        evaluated_at=timestamp,
        checkpoint_path="",
        mode=mode,
        system_name=system_name,
        summary=summary,
    )
    print(
        "Benchmark summary: "
        f"total={summary['total']} "
        f"correct={summary['correct']} "
        f"partially_correct={summary['partially_correct']} "
        f"incorrect={summary['incorrect']}"
    )
    print(f"Saved benchmark results to {output_path}")


def generate_answer(model: Any, tokenizer: AutoTokenizer, question: str) -> str:
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )
