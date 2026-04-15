import json
from pathlib import Path
from src.v2.benchmark.types import (
    BenchmarkResultItem,
    BenchmarkRunResult,
    BenchmarkSummary,
)


def save_benchmark_results(
    results: list[BenchmarkResultItem],
    path: Path,
    model_name: str,
    evaluated_at: str,
    checkpoint_path: str,
    mode: str,
    system_name: str,
    summary: BenchmarkSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result: BenchmarkRunResult = {
        "mode": mode,
        "system_name": system_name,
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "evaluated_at": evaluated_at,
        "results": results,
        "summary": summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Benchmark result {str(path)} is saved.")
