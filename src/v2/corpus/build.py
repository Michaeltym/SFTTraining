import re
import json
from datasets import load_dataset
from typing import cast
from src.config import (
    PYTORCH_DOCS_SOURCE_DIR,
    PYTORCH_CORPUS_OUTPUT_PATH,
    PYTORCH_SYMBOL_INDEX_OUTPUT_PATH,
)
from src.v2.corpus.types import (
    SourceDocument,
    CorpusChunk,
    SymbolIndexEntry,
    SymbolIndex,
)

SYMBOL_PATTERN = r"\b\w+(?:\.\w+)+(?:\(\))?\b"
NOISE_SYMBOL_PATTERN = r"\b\w\.\w+(?:\(\))?\b"
BROAD_SYMBOLS = ["torch.optim", "torch.tensor", "torch.utils", "torch.nn"]
ALLOWED_OBJECT_PREFIXES = {"loss", "optimizer", "model", "F"}
NOISE_OBJECT_PREFIXES = {"self", "obj", "foo", "bar"}


def build_corpus():
    docs = load_source_documents()
    chunks = [chunk for doc in docs for chunk in chunk_document(doc=doc)]
    symbol_index = build_symbol_index(chunks=chunks)
    write_corpus_jsonl(chunks=chunks)
    write_symbol_index_json(symbol_index=symbol_index)


def load_source_documents() -> list[SourceDocument]:
    data_files = [str(f) for f in PYTORCH_DOCS_SOURCE_DIR.glob("*.jsonl")]
    dataset = load_dataset("json", data_files=data_files, split="train")
    docs = cast(list[SourceDocument], dataset.to_list())
    return docs


def build_aliases(symbols: list[str]) -> list[str]:
    aliases: list[str] = []
    for symbol in symbols:
        parts = symbol.split(".")
        if len(parts) > 2:
            aliases.append(".".join(parts[-2:]))
            aliases.append(" ".join(parts[-2:]))
        if len(parts) > 1:
            aliases.append(parts[-1])

    return list(dict.fromkeys(aliases))


def is_broad_symbol(symbol: str) -> bool:
    return symbol.lower() in BROAD_SYMBOLS


def is_noise_symbol(symbol: str) -> bool:
    prefix = symbol.split(".", 1)[0].lower()
    if prefix in ALLOWED_OBJECT_PREFIXES:
        return False
    if prefix in NOISE_OBJECT_PREFIXES:
        return True
    return bool(re.fullmatch(NOISE_SYMBOL_PATTERN, symbol))


def chunk_document(doc: SourceDocument) -> list[CorpusChunk]:
    corpus_chunks: list[CorpusChunk] = []
    text = doc["text"]
    title = doc["title"]
    title_symbols = extract_symbols(title)
    text_symbols = extract_symbols(text)
    filtered_text_symbols = [
        symbol
        for symbol in text_symbols
        if not is_broad_symbol(symbol=symbol) and not is_noise_symbol(symbol=symbol)
    ]
    corpus_chunks.append(
        {
            **doc,
            "token_count": len(text.split()),
            "symbols": list(dict.fromkeys(title_symbols + filtered_text_symbols)),
            "aliases": build_aliases(title_symbols),
        }
    )
    return corpus_chunks


def extract_symbols(text: str) -> list[str]:
    return re.findall(SYMBOL_PATTERN, text)


def build_symbol_index(chunks: list[CorpusChunk]) -> SymbolIndex:
    symbol_index: dict[str, SymbolIndexEntry] = {}
    for chunk in chunks:
        doc_id = chunk["doc_id"]
        symbols = chunk["symbols"]
        for symbol in symbols:
            entry = symbol_index.get(symbol)
            if entry:
                entry["doc_ids"] = list(dict.fromkeys(entry["doc_ids"] + [doc_id]))
            else:
                new_entry: SymbolIndexEntry = {
                    "symbol": symbol,
                    "doc_ids": [doc_id],
                    "aliases": build_aliases([symbol]),
                }
                symbol_index[symbol] = new_entry
    return symbol_index


def write_corpus_jsonl(chunks: list[CorpusChunk]):
    PYTORCH_CORPUS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PYTORCH_CORPUS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )
    print("##### pytorch_corpus.jsonl saved #####")


def write_symbol_index_json(symbol_index: SymbolIndex):
    PYTORCH_SYMBOL_INDEX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PYTORCH_SYMBOL_INDEX_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(symbol_index, f, ensure_ascii=False, indent=2)
    print("##### pytorch_symbol_index.json saved #####")
