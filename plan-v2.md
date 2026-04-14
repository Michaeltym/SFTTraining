# Plan To Upgrade the Current Prototype into a "90% Coverage" PyTorch Assistant

## Summary

The main bottleneck in the current system is not simply that the local knowledge files are too small. The real issue is that the overall architecture has a very low ceiling: a small local LoRA model, a tiny knowledge base, and `top_k=1` lexical retrieval. That combination is enough for a narrow demo, but it is not enough to support "90% of core PyTorch questions."

This plan sets the target as follows:

- Success is not "it feels smarter"; success is "it reaches at least 90% acceptable-answer rate on a fixed core PyTorch benchmark"
- Phase 1 scope is limited to `Core PyTorch dev QA`
  covering `tensor / shape / dtype / device / autograd / nn / optim / data / training-vs-eval / debugging`
- The runtime shape is `Hybrid docs + strong model`
  The current `Qwen2.5-0.5B + LoRA` remains as a baseline, not the final primary answerer
- Phase 1 knowledge sources are limited to `official docs + tutorials + recipes`
  Forums and issues are not included early in v1

## Recommended v2

The recommended v2 architecture is:

`official PyTorch corpus + hybrid retrieval + stronger answer model + citations`

The system should be split into five layers:

1. `Corpus layer`
- Official PyTorch docs
- tutorials
- recipes
- structured chunking
- symbol index

2. `Retrieval layer`
- exact symbol match
- BM25 / lexical retrieval
- dense retrieval
- reranker
- final `top 4-6` context selection

3. `Answer layer`
- a stronger model produces the final answer
- the model is constrained to answer from retrieved context
- answers must include citations
- the system may abstain when evidence is insufficient

4. `Evaluation layer`
- fixed benchmark
- overall accuracy
- per-category accuracy
- refusal accuracy
- citation support rate

5. `Baseline layer`
- keep the current `base + LoRA + current RAG`
- use it only as a baseline and local fallback
- do not treat it as the main production path

Why this is the right v2 direction:

- PyTorch assistant quality depends on symbol precision, comparison handling, debugging coverage, grounded refusal, and answer compression
- the current architecture has structural limits in retrieval quality, knowledge coverage, and generator strength
- v2 should invest in better corpus coverage, better retrieval, and a stronger answer model instead of continuing to patch the small local generator

## Key Implementation Changes

### 1. Turn "90%" into a measurable target first

Add a formal benchmark instead of continuing to rely on the current small regression set.

Data and interface contract:

- Add `data/eval/benchmark_core_pytorch.jsonl`
- Each benchmark item must include at least:
  - `id`
  - `question`
  - `category`
  - `gold_type`
  - `expected_symbols`
  - `must_include`
  - `must_not_include`
  - `requires_citation`
  - `difficulty`
- Fixed category set:
  - `tensor_creation`
  - `shape_ops`
  - `dtype_device`
  - `autograd`
  - `nn_module_modes`
  - `optim_training_loop`
  - `data_loading`
  - `debugging`
  - `hallucination_refusal`

Evaluation labels must use a fixed three-way scale:

- `correct`
- `partially_correct`
- `incorrect`

The final report must always output:

- overall accuracy
- per-category accuracy
- citation support rate
- hallucination refusal accuracy
- top 20 failure cases

Phase 1 benchmark size is fixed at:

- dev set `200`
- test set `200`

The reason is simple: without this benchmark, every later gain from expanding knowledge, changing retrieval, or changing the model will be impossible to compare reliably.

### 2. Replace the current small hand-written knowledge base with a structured official corpus

Do not keep scaling this system by adding dozens of hand-written Markdown files. Instead, build an offline indexed PyTorch corpus from official sources.

New corpus output format:

- `data/output/cache/pytorch_corpus.jsonl`
- Each chunk must include at least:
  - `doc_id`
  - `title`
  - `url`
  - `section_path`
  - `source_type`
  - `symbols`
  - `aliases`
  - `text`
  - `token_count`

Fixed document sources:

- `PyTorch API docs`
- `PyTorch tutorials`
- `PyTorch recipes`

Fixed chunking strategy:

- Use `symbol` or `section` as the primary chunk unit
- Target chunk length: `200-500` words
- Preserve heading hierarchy and URL
- Create dedicated chunks for comparison-heavy topics
  such as `view vs reshape`, `train vs eval`, and `tensor vs as_tensor vs from_numpy`

Also generate a symbol index:

- `data/output/cache/pytorch_symbol_index.json`
- Used for exact symbol matching and alias normalization for `torch.xxx`, `nn.xxx`, and `Tensor.xxx`

Do not delete the current `data/knowledge/pytorch_docs/*.md` files, but change their role to:

- a small manual correction layer
- a place for refusal rules or high-value comparison notes that official docs do not state directly enough
- no longer the primary coverage mechanism

### 3. Upgrade retrieval from "simple lexical top-1" to "hybrid retrieval + reranking"

The current `lexical overlap + top_k=1` path can remain as a baseline, but it cannot remain the target system.

The target retrieval chain is fixed as:

1. `symbol exact match`
2. `BM25 / lexical retrieval`
3. `embedding retrieval`
4. `cross-encoder reranker`
5. Use `top 4-6` chunks as final context

Add a unified retrieval result structure:

- `query`
- `retrieved_docs`
- `score_breakdown`
- `matched_symbols`

Each `retrieved_doc` must include at least:

- `doc_id`
- `url`
- `title`
- `score_lexical`
- `score_dense`
- `score_rerank`
- `final_rank`

Fixed behavior rules:

- If the question contains explicit API symbols, `exact symbol match` must have strong priority
- Comparison questions must be allowed to return multiple relevant chunks, not forced back to `top_k=1`
- If retrieval evidence is insufficient, the answer layer must be allowed to `abstain / not sure`
- The retrieval module must support standalone debug output so it is easy to tell whether the problem is "failed to retrieve" or "retrieved correctly but answered badly"

### 4. Upgrade the answerer so a stronger model handles generation, and move the current LoRA model to baseline/local fallback

If the target is 90% coverage on core PyTorch QA, the current `Qwen2.5-0.5B + LoRA` should not remain the primary answerer. It can still be kept as:

- a baseline comparison system
- a low-cost local fallback
- an SFT learning experiment artifact

The primary answerer should be:

- a stronger general-purpose generation model
- given "question + reranked context + explicit answering constraints" as input
- required to return answers with citations

Use a unified answer output shape:

- `answer`
- `citations`
- `used_symbols`
- `abstained`
- `confidence_band`

Fixed answer rules:

- answer the question directly first
- only use supported information from retrieved evidence
- if evidence is insufficient, explicitly say so
- for debugging questions, provide `cause + minimal fix`
- for API/reference questions, provide `what it does + key args/shape/constraints`
- for hallucination questions, refuse directly and do not invent replacement API details

Use two prompt templates:

- `reference/debugging template`
- `fake-api refusal template`

Do not force every question through one generic prompt.

### 5. Iterate by failure slices instead of blindly expanding knowledge

After the first full benchmark run, only iterate based on failure distribution, in this fixed order:

1. `retrieval misses`
2. `citation unsupported answers`
3. `comparison confusion`
4. `debugging under-specification`
5. `generator verbosity / repetition`

In each iteration, only change one primary variable:

- corpus coverage
- chunk granularity
- retrieval weights
- reranker
- prompt
- primary answer model

Do not change model, retrieval, and benchmark all in the same iteration.

## Testing and Acceptance

### Core test scenarios

The benchmark must cover:

- API definition questions
- argument / return value questions
- shape reasoning
- dtype / device errors
- autograd / backward / gradient accumulation
- `model.train()` / `model.eval()` / `torch.no_grad()`
- `cat` / `stack` / `view` / `reshape` / `from_numpy` / `as_tensor`
- fake API refusal
- common debugging questions
  such as shape mismatch, device mismatch, dtype mismatch, and contiguity

### Engineering checks

Add automated checks for:

- corpus build succeeds and produces valid chunks
- symbol index hits common API names correctly
- retrieval hits expected documents on a fixed query set
- answer result JSON schema is stable
- benchmark runner outputs category metrics and failure samples

### Phase acceptance gates

Phase 1 is complete when:

- the benchmark is built
- the current system runs on it successfully
- the real error distribution is visible

Phase 2 is complete when:

- the official corpus index is built
- retrieval hit quality is clearly better than the current hand-written knowledge path
- retrieval debug output is available

Phase 3 is complete when:

- the `Core PyTorch dev QA` test set reaches:
  - `>= 90%` acceptable-answer rate
  - `>= 95%` fake API refusal accuracy
  - `>= 90%` citation support rate

## Assumptions and Default Decisions

- "90%" means the `Core PyTorch dev QA benchmark`, not the full long-tail PyTorch ecosystem
- v1 only covers `official docs + tutorials + recipes`
- The current `Qwen/Qwen2.5-0.5B + LoRA` remains as a baseline, not the final primary answerer
- This repository should prioritize `benchmark + corpus/index + retrieval + answer pipeline`, not further SFT dataset expansion first
- If real-world long-tail debugging needs to be added later, that should be a separate phase that introduces `forums/issues` as lower-trust sources with filtering
