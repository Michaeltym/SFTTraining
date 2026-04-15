import json
from datasets import load_dataset
from typing import cast
from src.config import PYTORCH_CORPUS_OUTPUT_PATH, PYTORCH_SYMBOL_INDEX_OUTPUT_PATH
from src.v2.corpus.types import (
    CorpusChunk,
    SymbolIndex,
)


def load_corpus() -> list[CorpusChunk]:
    dataset = load_dataset(
        "json", data_files=str(PYTORCH_CORPUS_OUTPUT_PATH), split="train"
    )
    return cast(list[CorpusChunk], dataset.to_list())


def load_symbol_index() -> SymbolIndex:
    with open(str(PYTORCH_SYMBOL_INDEX_OUTPUT_PATH), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
