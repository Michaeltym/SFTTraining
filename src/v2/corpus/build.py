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

# The four rules below together decide which textual references count as a
# real PyTorch API symbol and therefore become symbol-index keys. Query-side
# extraction uses SYMBOL_PATTERN unfiltered so users can type
# `loss.backward()` and still route; the filters only apply when building
# index keys from chunk body text.

# Dotted identifier with at least two segments, optional trailing `()`.
# Matches both canonical API paths (`torch.cat`, `Tensor.backward`) and
# variable-name usage (`loss.backward()`, `x.shape`). Downstream filters
# narrow this down for indexing.
SYMBOL_PATTERN = r"\b\w+(?:\.\w+)+(?:\(\))?\b"

# Dotted symbol whose first segment is a single character. Almost always an
# example variable in code snippets (`x.shape`, `a.to`, `y.backward()`),
# never a real API path. Caught by `is_noise_symbol` so single-letter-prefix
# forms never become index keys.
NOISE_SYMBOL_PATTERN = r"\b\w\.\w+(?:\(\))?\b"

# Top-level namespace prefixes that are too general to route on. If
# indexed, a query containing `torch.optim` would exact-match every doc
# that mentions the namespace, flooding retrieval with top-level hits.
# Excluded from chunk symbol lists even when the body text mentions them
# literally. Canonical full paths (`torch.optim.SGD`,
# `torch.nn.Module.train`) still get indexed under their real names.
BROAD_SYMBOLS = ["torch.optim", "torch.tensor", "torch.utils", "torch.nn"]

# Variable-name prefixes used in PyTorch docs examples (`loss.backward()`,
# `optimizer.zero_grad()`, `model.train()`, `F.relu(x)`). These are
# placeholders users write in their own code, not real API paths. If
# allowed as symbol-index keys, a doc whose body happens to mention
# `loss.backward()` in an example becomes a symbol-hit target for the
# query "loss.backward" and drags unrelated docs (e.g.
# `api_docs_optimizer_zero_grad`) into retrieval. The canonical API forms
# (`Tensor.backward`, `Optimizer.zero_grad`, `Module.train`) stay in the
# index under their real names, and queries fall through to those via
# alias match on the short form (`backward`, `zero_grad`, `train`).
OBJECT_EXAMPLE_PREFIXES = {"loss", "optimizer", "model", "f"}

# Pedagogical placeholder names used in generic Python examples
# (`self.foo`, `obj.bar`, `foo.x`, `bar.y`). These never refer to real
# PyTorch APIs. Excluded from symbol-index keys so they cannot pollute
# retrieval regardless of how often they appear in docs body text.
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
    if prefix in OBJECT_EXAMPLE_PREFIXES:
        return True
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


if __name__ == "__main__":
    build_corpus()
