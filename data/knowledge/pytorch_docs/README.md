# PyTorch Docs Knowledge Base

This directory stores a small curated knowledge base for the first RAG prototype.

Scope of v1:

- keep the document set intentionally small
- focus on the highest-value PyTorch failure cases seen in evaluation
- prefer official PyTorch documentation pages
- allow a small number of curated notes when official docs do not directly express the desired assistant behavior

Current file format:

- plain Markdown
- a small metadata header at the top of each file
- human-editable content

Expected metadata keys:

- `id`
- `title`
- `url`
- `source_type`
- `tags`

The first RAG implementation should treat each file as one knowledge item or split the body into smaller chunks if needed.
