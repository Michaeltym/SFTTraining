"""Re-score existing benchmark result JSONs with the current scorer.

Use this when the scorer logic (``get_benchmark_label``) or the benchmark item
rules (``must_not_include``, ``must_not_include_regex``, etc.) change, and
you want to re-label historical runs without re-running inference.

Usage:
    python scripts/rescore_benchmark.py <path> [<path> ...]

For each input JSON, writes a rescored copy to
``<BENCHMARK_RESULTS_DIR>/rescored/<stem>-rescored.json`` and prints a
before/after summary delta so the impact of the scorer change is visible at a
glance. Inference-time fields (``answer``, ``citations``, ``used_symbols``,
``abstained``, ``confidence_band``, ``retrieval_debug``) are preserved
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running as a plain script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import BENCHMARK_DATA_PATH, BENCHMARK_RESULTS_DIR
from src.v2.benchmark.data import load_benchmark_data
from src.v2.benchmark.label import get_benchmark_label
from src.v2.benchmark.summary import build_benchmark_summary
from src.v2.benchmark.types import BenchmarkItem, BenchmarkResultItem


def _load_items_by_id(item_path: Path) -> dict[str, BenchmarkItem]:
    items = load_benchmark_data(file_path=str(item_path))
    return {item["id"]: item for item in items}


def _label_counts(results: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for r in results:
        counts[r["label"]] += 1
    return counts


def rescore_file(
    source_path: Path,
    items_by_id: dict[str, BenchmarkItem],
    output_dir: Path,
) -> Path:
    """Rescore a single benchmark result JSON.

    Returns the path of the rescored output file. Prints a before/after
    summary delta for quick inspection.
    """
    with source_path.open("r", encoding="utf-8") as f:
        original: dict[str, Any] = json.load(f)

    original_results: list[dict[str, Any]] = original["results"]

    before_counts = _label_counts(original_results)

    rescored_results: list[BenchmarkResultItem] = []
    changes: list[tuple[str, str, str]] = []

    for result in original_results:
        item_id = result["id"]
        if item_id not in items_by_id:
            raise KeyError(
                f"benchmark item id {item_id!r} from {source_path} "
                f"not found in {BENCHMARK_DATA_PATH}"
            )
        item = items_by_id[item_id]
        old_label = result["label"]

        new_label, new_notes = get_benchmark_label(
            item=item,
            answer=result["answer"],
        )
        # Preserve every inference-time field by shallow-copying the original
        # result dict and overlaying the freshly computed scorer fields.
        rescored: dict[str, Any] = dict(result)
        rescored["label"] = new_label
        rescored["notes"] = new_notes
        rescored_results.append(rescored)  # type: ignore[arg-type]

        if old_label != new_label:
            changes.append((item_id, old_label, new_label))

    after_counts = _label_counts(rescored_results)
    summary = build_benchmark_summary(results=rescored_results)

    rescored_at = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    output_payload: dict[str, Any] = {
        **{k: v for k, v in original.items() if k not in {"results", "summary"}},
        "results": rescored_results,
        "summary": summary,
        "rescored_at": rescored_at,
        "rescored_from": source_path.name,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_path.stem}-rescored.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    # Report.
    print(f"=== {source_path.name} ===")
    print(f"  before: {dict(before_counts)}")
    print(f"  after : {dict(after_counts)}")
    print(f"  summary: total={summary['total']} "
          f"correct={summary['correct']} "
          f"partially_correct={summary['partially_correct']} "
          f"incorrect={summary['incorrect']} "
          f"overall_accuracy={summary['overall_accuracy']:.3f}")
    if changes:
        print(f"  label changes ({len(changes)}):")
        for item_id, old, new in changes:
            print(f"    {item_id:32s} {old:20s} -> {new}")
    else:
        print("  label changes: none")
    print(f"  -> {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Benchmark result JSON files to rescore.",
    )
    parser.add_argument(
        "--items",
        type=Path,
        default=BENCHMARK_DATA_PATH,
        help=(
            "Benchmark item JSONL to load scorer rules from "
            f"(default: {BENCHMARK_DATA_PATH})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BENCHMARK_RESULTS_DIR / "rescored",
        help=(
            "Directory to write rescored result files to "
            f"(default: {BENCHMARK_RESULTS_DIR / 'rescored'})."
        ),
    )
    args = parser.parse_args(argv)

    if not args.items.exists():
        print(f"ERROR: items file not found: {args.items}", file=sys.stderr)
        return 2

    items_by_id = _load_items_by_id(args.items)

    for path in args.paths:
        if not path.exists():
            print(f"ERROR: result file not found: {path}", file=sys.stderr)
            return 2
        rescore_file(
            source_path=path,
            items_by_id=items_by_id,
            output_dir=args.output_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
