# PyTorch API Assistant RAG Plan

## Goal

Build a small **RAG-based PyTorch API assistant** on top of the current best LoRA checkpoint.

The system should:

- answer PyTorch questions with external grounding
- retrieve relevant PyTorch documentation before generation
- reduce fake API hallucinations
- improve debugging and API-semantics answers
- return concise answers plus supporting sources

The short-term product direction is:

1. make the assistant more useful before building UI
2. add documentation grounding on top of the current best model
3. compare plain LoRA answers vs LoRA + RAG answers

## Current Direction

The project has already shown that:

- the current best LoRA checkpoint is better than the base model in some PyTorch-specific behaviors
- the model is still not reliable enough to answer arbitrary PyTorch questions on its own
- the remaining bottlenecks are:
  - fake API refusal
  - debugging semantics
  - factual stability on API details

Because of that, the next implementation direction is:

- keep the current best LoRA checkpoint
- add retrieval over a small PyTorch knowledge base
- use retrieved context to constrain generation
- validate usefulness in CLI before building a frontend

## Model Strategy

Use the current best checkpoint as the generator:

- base model: `Qwen/Qwen2.5-0.5B`
- dataset: `dataset_3`
- LoRA checkpoint: best `epoch 1` run

Why:

- it is the best adapted model available so far
- it already has some PyTorch-domain bias
- it is a better RAG generator candidate than the raw base model

Rules:

- do not treat the current LoRA checkpoint as sufficient on its own
- use retrieval to improve factuality and refusal behavior
- keep the best checkpoint fixed while building the first RAG prototype

## Knowledge Base Direction

Start with a **small curated PyTorch knowledge base**, not the full docs.

Prioritize coverage for current failure areas:

- `view`, `reshape`, `permute`, `contiguous`
- `.item()`
- `sum`, `mean`, `keepdim`
- `model.eval()` vs `torch.no_grad()`
- `from_numpy`, `as_tensor`, `tensor`
- fake API refusal rules and nearby valid APIs

Suggested storage:

- `data/knowledge/pytorch_docs/`

Each knowledge chunk should carry:

- `id`
- `title`
- `url`
- `text`

## Retrieval Strategy

### Phase 1: Simple Local Retrieval

Start with a simple lexical retriever.

Why:

- faster to implement
- easier to debug
- enough to test whether grounding improves usefulness
- avoids premature infrastructure work such as vector DB setup

The retriever should:

- normalize the query
- score chunks by keyword overlap / simple lexical matching
- return top-k chunks

### Phase 2: Optional Stronger Retrieval

Only if Phase 1 is clearly insufficient:

- add embeddings
- add vector search
- expand the document set

Do not start here.

## Prompting Strategy

RAG prompting should explicitly instruct the model to:

- answer using the provided context
- refuse to invent undocumented APIs
- say when the retrieved context is insufficient
- keep answers concise and API-focused

The response should ideally include:

- direct answer
- short explanation
- source titles or URLs

## Implementation Plan

### Phase 1: CLI RAG Prototype

Implement a minimal CLI path first.

Planned modules:

- `src/knowledge.py`
- `src/retriever.py`
- `src/rag_prompt.py`
- `src/rag_infer.py`

Responsibilities:

- `knowledge.py`
  - load and normalize knowledge files
  - produce chunk objects

- `retriever.py`
  - rank chunks for a question
  - return top-k supporting context

- `rag_prompt.py`
  - build the grounded generation prompt

- `rag_infer.py`
  - load best base model + adapter
  - retrieve context
  - generate answer
  - print answer and sources

### Phase 2: Evaluation

Compare:

- plain best-LoRA answers
- best-LoRA + RAG answers

Evaluation focus:

- fake API refusal
- `.item()`
- `eval()` vs `no_grad()`
- `view` / `reshape` / `contiguous`
- `from_numpy` / `as_tensor` / `tensor`

Use the existing fixed evaluation prompts where possible.

### Phase 3: Frontend

Only after the CLI RAG prototype is useful enough:

- expose the RAG path behind a simple web app
- show answer plus supporting sources
- keep the UI thin until usefulness is proven

## Experiment Policy

While building RAG:

- keep the model fixed
- keep the best checkpoint fixed
- keep the first knowledge base intentionally small
- change one retrieval variable at a time

Do not mix at once:

- new datasets
- new SFT runs
- new base models
- vector DB infrastructure
- frontend polish

## Immediate Next Steps

1. create a small curated PyTorch knowledge set
2. implement `knowledge.py`
3. implement `retriever.py`
4. implement `rag_prompt.py`
5. implement `rag_infer.py`
6. compare plain LoRA vs LoRA + RAG on existing eval prompts
