import time
from datetime import datetime
from pathlib import Path

from src.config import BENCHMARK_DATA_PATH, BENCHMARK_NAME, BENCHMARK_RESULTS_DIR
from src.model import get_model_name_slug
from src.v2.benchmark.data import load_benchmark_data
from src.v2.benchmark.label import get_benchmark_label
from src.v2.benchmark.save import save_benchmark_results
from src.v2.benchmark.summary import build_benchmark_summary
from src.v2.benchmark.types import BenchmarkResultItem, BenchmarkAnswerFn


def run_benchmark(
    system_name: str,
    model_name: str,
    mode: str,
    answer_fn: BenchmarkAnswerFn,
) -> None:
    time_start = time.time()
    print(f"##### Run {system_name} benchmark ({BENCHMARK_NAME}) #####")
    results: list[BenchmarkResultItem] = []
    benchmark_items = load_benchmark_data(file_path=str(BENCHMARK_DATA_PATH))
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    total = len(benchmark_items)
    for index, benchmark_item in enumerate(benchmark_items, start=1):
        print(f"[{index}/{total}] {benchmark_item['id']}", flush=True)
        benchmark_answer = answer_fn(benchmark_item["question"])
        label, notes = get_benchmark_label(
            item=benchmark_item,
            answer=benchmark_answer["answer"],
        )
        result_item: BenchmarkResultItem = {
            **benchmark_item,
            **benchmark_answer,
            "label": label,
            "notes": notes,
        }

        if "retrieval_debug" in benchmark_answer:
            result_item["retrieval_debug"] = benchmark_answer["retrieval_debug"]

        results.append(result_item)

    summary = build_benchmark_summary(results=results)
    model_name_slug = get_model_name_slug(model_name)
    output_path = (
        Path(BENCHMARK_RESULTS_DIR)
        / f"{mode}-{system_name}-{model_name_slug}-{BENCHMARK_NAME}-{timestamp}.json"
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
        f"incorrect={summary['incorrect']} "
        f"time={time.time() - time_start:.2f}s",
    )
    print(f"Saved benchmark results to {output_path}")
