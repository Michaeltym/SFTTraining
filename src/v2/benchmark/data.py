from typing import Any, cast
from datasets import load_dataset

from src.v2.benchmark.types import BenchmarkItem


def load_benchmark_data(file_path: str) -> list[BenchmarkItem]:
    dataset = load_dataset("json", data_files=file_path, split="train")
    rows = cast(list[dict[str, Any]], dataset.to_list())
    benchmark_items: list[BenchmarkItem] = []

    for item in rows:
        benchmark_item: BenchmarkItem = {
            "id": item["id"],
            "question": item["question"],
            "category": item["category"],
            "gold_type": item["gold_type"],
            "expected_symbols": item["expected_symbols"],
            "must_include": item["must_include"],
            "must_include_any_of": item.get("must_include_any_of") or [],
            "must_not_include_regex": item.get("must_not_include_regex") or [],
            "must_not_include": item["must_not_include"],
            "requires_citation": item["requires_citation"],
            "difficulty": item["difficulty"],
        }
        benchmark_items.append(benchmark_item)

    return benchmark_items
