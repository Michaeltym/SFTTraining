from pathlib import Path
from typing import TypedDict
from src.config import KNOWLEDGE_DIR


class Metadata(TypedDict):
    id: str
    title: str
    url: str
    source_type: str
    tags: list[str]


class KnowledgeItem(Metadata):
    text: str


def load_knowledge() -> list[KnowledgeItem]:
    items: list[KnowledgeItem] = []
    files = sorted(
        [f for f in KNOWLEDGE_DIR.glob("*.md") if f.stem.lower() != "readme"]
    )
    for file_path in files:
        if file_path.is_file():
            metadata, text = parse_knowledge_file(path=file_path)
            items.append({**metadata, "text": text})
    return items


def parse_knowledge_file(path: Path) -> tuple[Metadata, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    raw_metadata: dict[str, str] = {}
    body_start: int | None = None

    for i, line in enumerate(lines):
        if not line.strip():
            body_start = i + 1
            break

        if ": " not in line:
            raise ValueError(f"Invalid metadata line in {path}: {line}")

        key, value = line.split(": ", 1)
        raw_metadata[key.strip()] = value.strip()

    if body_start is None:
        raise ValueError(
            f"Knowledge file {path} is missing a blank line between metadata and body."
        )

    metadata: Metadata = {
        "id": raw_metadata["id"],
        "title": raw_metadata["title"],
        "url": raw_metadata["url"],
        "source_type": raw_metadata["source_type"],
        "tags": [tag.strip() for tag in raw_metadata["tags"].split(",")],
    }

    text = "\n".join(lines[body_start:]).strip()
    return metadata, text
