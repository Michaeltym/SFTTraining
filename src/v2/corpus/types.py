from typing import Literal, TypedDict


CorpusSourceType = Literal["api_docs", "tutorial", "recipe"]


class CorpusChunk(TypedDict):
    doc_id: str
    title: str
    url: str
    section_path: list[str]
    source_type: CorpusSourceType
    symbols: list[str]
    aliases: list[str]
    text: str
    token_count: int


class SourceDocument(TypedDict):
    doc_id: str
    title: str
    url: str
    source_type: CorpusSourceType
    section_path: list[str]
    text: str


# Example of a SymbolIndexEntry:
# {
#     "symbol": "torch.cat",
#     "aliases": ["cat"],
#     "doc_ids": ["compare_cat_vs_stack", "api_docs_torch_cat"],
# }
class SymbolIndexEntry(TypedDict):
    symbol: str
    aliases: list[str]
    doc_ids: list[str]


SymbolIndex = dict[str, SymbolIndexEntry]
