from src.v2.benchmark.types import (
    BenchmarkResultItem,
    BenchmarkSummary,
)


def build_benchmark_summary(results: list[BenchmarkResultItem]) -> BenchmarkSummary:
    total = len(results)
    correct = sum(1 for result in results if result["label"] == "correct")
    partially_correct = sum(
        1 for result in results if result["label"] == "partially_correct"
    )
    incorrect = sum(1 for result in results if result["label"] == "incorrect")
    overall_accuracy = correct / total if total > 0 else 0.0

    citation_required_results = [
        result for result in results if result["requires_citation"]
    ]
    citation_supported_results = [
        result for result in citation_required_results if len(result["citations"]) > 0
    ]
    citation_support_rate = (
        len(citation_supported_results) / len(citation_required_results)
        if len(citation_required_results) > 0
        else 0.0
    )

    hallucination_refusal_results = [
        result for result in results if result["category"] == "hallucination_refusal"
    ]
    hallucination_refusal_correct_results = [
        result
        for result in hallucination_refusal_results
        if result["label"] == "correct"
    ]
    hallucination_refusal_accuracy = (
        len(hallucination_refusal_correct_results) / len(hallucination_refusal_results)
        if len(hallucination_refusal_results) > 0
        else 0.0
    )

    categories = sorted({result["category"] for result in results})
    per_category_accuracy: dict[str, float] = {}
    for category in categories:
        category_results = [
            result for result in results if result["category"] == category
        ]
        category_correct_results = [
            result for result in category_results if result["label"] == "correct"
        ]
        per_category_accuracy[category] = (
            len(category_correct_results) / len(category_results)
            if len(category_results) > 0
            else 0.0
        )

    return {
        "total": total,
        "correct": correct,
        "partially_correct": partially_correct,
        "incorrect": incorrect,
        "overall_accuracy": overall_accuracy,
        "citation_support_rate": citation_support_rate,
        "hallucination_refusal_accuracy": hallucination_refusal_accuracy,
        "per_category_accuracy": per_category_accuracy,
    }
