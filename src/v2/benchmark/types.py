from typing import TypedDict, Literal, NotRequired
from collections.abc import Callable

from src.v2.retrieval.types import RetrievalDebug

BenchmarkCategory = Literal[
    "tensor_creation",
    "shape_ops",
    "dtype_device",
    "autograd",
    "nn_module_modes",
    "optim_training_loop",
    "data_loading",
    "debugging",
    "hallucination_refusal",
]

BenchmarkGoldType = Literal[
    "definition",
    "usage",
    "comparison",
    "shape_reasoning",
    "debugging",
    "hallucination_check",
]

BenchmarkDifficulty = Literal["easy", "medium", "hard"]

BenchmarkLabel = Literal["correct", "partially_correct", "incorrect"]

BenchmarkConfidenceBand = Literal["low", "medium", "high"]


class BenchmarkItem(TypedDict):
    id: str
    question: str
    category: BenchmarkCategory
    gold_type: BenchmarkGoldType
    expected_symbols: list[str]
    must_include: list[str]
    must_include_any_of: NotRequired[list[list[str]]]
    must_not_include: list[str]
    requires_citation: bool
    difficulty: BenchmarkDifficulty
    must_not_include_regex: NotRequired[list[str]]


class BenchmarkCitation(TypedDict):
    title: str
    url: str


class BenchmarkAnswer(TypedDict):
    answer: str
    citations: list[BenchmarkCitation]
    used_symbols: list[str]
    abstained: bool
    confidence_band: BenchmarkConfidenceBand
    retrieval_debug: NotRequired[RetrievalDebug]


BenchmarkAnswerFn = Callable[[str], BenchmarkAnswer]


class MustIncludeNotes(TypedDict):
    exact_matches: list[str]
    fuzzy_matches: list[str]
    fuzzy_logs: list[str]
    missing: list[str]


class AnyOfGroupMatch(TypedDict):
    group: list[str]
    matched: list[str]


class MustIncludeAnyOfNotes(TypedDict):
    matched_groups: list[AnyOfGroupMatch]
    missing_groups: list[list[str]]


class MustNotIncludeNotes(TypedDict):
    matched_phrases: list[str]
    matched_regex: list[str]


class ExpectedSymbolsNotes(TypedDict):
    matched: list[str]
    missing: list[str]


class BenchmarkLabelNotes(TypedDict):
    must_include: MustIncludeNotes
    must_include_any_of: MustIncludeAnyOfNotes
    must_not_include: MustNotIncludeNotes
    expected_symbols: ExpectedSymbolsNotes


class BenchmarkResultItem(BenchmarkItem):
    answer: str
    citations: list[BenchmarkCitation]
    used_symbols: list[str]
    abstained: bool
    confidence_band: BenchmarkConfidenceBand
    label: BenchmarkLabel
    notes: NotRequired[BenchmarkLabelNotes]
    retrieval_debug: NotRequired[RetrievalDebug]


class BenchmarkSummary(TypedDict):
    total: int
    correct: int
    partially_correct: int
    incorrect: int
    overall_accuracy: float
    citation_support_rate: float
    hallucination_refusal_accuracy: float
    per_category_accuracy: dict[str, float]


class BenchmarkRunResult(TypedDict):
    mode: str
    system_name: str
    model_name: str
    checkpoint_path: str
    evaluated_at: str
    summary: BenchmarkSummary
    results: list[BenchmarkResultItem]
