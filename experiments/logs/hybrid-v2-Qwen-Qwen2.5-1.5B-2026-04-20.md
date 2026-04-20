## Run Info

- System Type: `v2 hybrid retrieval benchmark`
- Primary Generator Model: `Qwen/Qwen2.5-1.5B`
- Generator Type: `base model`
- Retriever Type: `symbol retrieval + lexical retrieval + hybrid merge`
- Prompt Type: `hybrid context prompt, top_k=3`
- Corpus Source Directory: [pytorch_docs](../../data/source/pytorch_docs)
- Corpus Cache: [pytorch_corpus.jsonl](../../data/output/cache/pytorch_corpus.jsonl)
- Symbol Index Cache: [pytorch_symbol_index.json](../../data/output/cache/pytorch_symbol_index.json)
- Core Benchmark Data: [benchmark_core_pytorch.jsonl](../../data/eval/benchmark_core_pytorch.jsonl)
- Smoke Benchmark Data: [benchmark_smoke_pytorch.jsonl](../../data/eval/benchmark_smoke_pytorch.jsonl)
- Timestamp: `2026-04-20`
- Smoke Baseline Entering Today (from 2026-04-19): `6 correct / 3 partially_correct / 3 incorrect = 0.5`
- Smoke Best After Today (Direction C): [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-212010.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-212010.json)
- Core Baseline Entering Today (from 2026-04-16): `25 correct / 8 partially_correct / 3 incorrect = 0.6944`
- Core Best After Today (Direction C + any_of + Tensor.permute): [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-20-212708.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-20-212708.json)

## Goal

Break through the `0.5` smoke plateau identified on 2026-04-19 by separating real model / retrieval limitations from scorer wording limitations, and by targeting the single clearly-policy-driven item `hallucination_refusal_003` without polluting the 11 other smoke items.

## Findings

### Session Part 1: `must_include_any_of` scorer extension (morning)

- Benchmark scoring was extended with:
  - `must_include_any_of`
  - semantics: each inner group is a set of equivalent acceptable phrases and at least one phrase per group must match
- The scorer implementation was updated to:
  - preserve the old `must_include` semantics
  - require all `must_include_any_of` groups for `correct`
  - count any-of group hits toward `partially_correct`
  - emit explicit notes for matched and missing any-of groups
- The benchmark data was updated for the two clearest wording-limitation cases in the smoke set:
  - `debugging_002`
  - `optim_training_loop_002`
- After that scorer and benchmark-data cleanup, the smoke benchmark improved from:
  - `6 correct / 3 partially_correct / 3 incorrect = 0.5`
  to:
  - `8 correct / 3 partially_correct / 1 incorrect = 0.6667`
  - file: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-200500.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-200500.json)
- This was a scorer-quality improvement, not a model or retrieval jump:
  - `debugging_002` now accepts `same shape` and `every dimension except`
  - `optim_training_loop_002` now accepts `class index`
- Both of those answers were manually checked and are acceptable as `correct`:
  - `optim_training_loop_002` is a genuine correct answer
  - `debugging_002` is slightly noisy in presentation, but its core constraint statement is correct
- The practical takeaway is that these two items were real scorer wording limitations, and fixing the benchmark was healthier than continuing to overfit prompt wording to them.
- The only remaining clearly interesting hard case in the smoke set that is not scorer wording is:
  - `hallucination_refusal_003`
  - it should be treated as a mixed real/fake API policy problem rather than a corpus coverage problem

### Session Part 2: Direction A / B / C iteration on `hallucination_refusal_003` (afternoon)

- Starting point for this sub-session:
  - smoke baseline `8 correct / 3 partially_correct / 1 incorrect = 0.6667`
  - the only remaining `incorrect` was `hallucination_refusal_003`
  - the diagnosis was that the reasoning prompt was forced to answer a question that mixes a real API (`optim.Adam`) and a fake API (`optim.AdamFast`) while the Facts block contained Adam / detach docs, creating fabrication pressure
- A guiding invariant was adopted for the whole sub-session:
  - change exactly one variable per iteration
  - preserve byte-equality of the generated prompt on all non-target items
  - since generation runs under greedy decoding, byte-equal prompts guarantee byte-equal answers, so untouched items must keep the exact same label

#### Direction A: inline the unconfirmed symbol into the reasoning prompt

- `src/v2/prompts/reasoning.py` now accepts `unconfirmed_symbols: list[str]` and conditionally inserts a literal `IMPORTANT: <backtick-quoted unconfirmed symbol>` block right before the Question line.
- The insertion only triggers when `unconfirmed_symbols` is non-empty, so all other smoke items stayed byte-equal.
- Result: smoke `9 correct / 3 partially_correct / 0 incorrect = 0.75`.
- But manual inspection showed this `correct` label on `hallucination_refusal_003` was a scorer false positive:
  - the answer still fabricated switching behavior for `optim.AdamFast` (borrowed from `Tensor.detach` in the context)
  - and also parroted the instruction string `you must state it is not a valid PyTorch API` literally, which is how the `must_include` phrase `not a valid` matched
- Conclusion on Direction A alone: headline improved but underlying behavior still wrong; do not stop here.

#### Direction B: narrow the refusal routing to also fire on mixed real/fake API queries

- `should_use_refusal_prompt` in `src/v2/prompts/router.py` now takes `matched_symbols` and `unconfirmed_symbols` and returns true when both are non-empty, in addition to the original conditions.
- `build_hybrid_prompt` was reordered so that query symbols are resolved before the refusal decision.
- Purpose: when the query contains both a real and a fake API, skip the reasoning template and use the refusal template, which does not stuff real API docs into the Facts block.
- Before enabling Direction B, the `benchmark_core_pytorch.jsonl` was scanned for other items that would fall into the `matched_symbols > 0 AND unconfirmed_symbols > 0` topology:
  - only two items matched that topology: `debugging_001` and `hallucination_refusal_003`
  - `debugging_001` previously got hurt by the earlier `has_unmatched_query_symbol` heuristic, so it was a known collateral risk
  - the root cause for `debugging_001` matching the mixed topology was that `Tensor.permute` was not in the symbol index, so `x.permute` resolved as unconfirmed even though it is a real API
  - this was a corpus coverage gap, not a routing bug
- To prevent collateral damage, a minimal source doc was added:
  - appended `api_docs_tensor_permute` (title `Tensor.permute`) to `data/source/pytorch_docs/shape_ops.jsonl`
  - rebuilt the corpus and symbol index with `python -m src.v2.corpus.build`
  - after rebuild, `Tensor.permute` is in the symbol index with alias `permute`
  - `debugging_001` now resolves `x.permute` as matched, so Direction B no longer routes it to refusal
  - the remaining mixed-topology item in both core and smoke is `hallucination_refusal_003` only
  - `data_loading_003` also routes to refusal but has no extracted query symbols, so its prompt stays byte-equal under Direction B and C
- Direction B alone, without Direction C, produced:
  - smoke `8 correct / 3 partially_correct / 1 incorrect = 0.6667`
  - `hallucination_refusal_003` flipped from the Direction A false-positive `correct` to `incorrect`
  - the actual answer on this run was `"Both optimizers are supported by the PyTorch library and can be used interchangeably"`, a different hallucination that asserts the fake API is real
- The refusal template at that time used abstract wording such as `"The retrieved PyTorch docs do not confirm this API"` and `"State clearly that this API does not exist in the retrieved PyTorch docs"`.
- In the mixed case `"this API"` was ambiguous between `optim.Adam` and `optim.AdamFast`, and the template gave no literal `not a valid` cue, so the model neither refused cleanly nor hit the scorer phrase. This motivated Direction C.

#### Direction C: inline the unconfirmed symbol into the refusal prompt

- `src/v2/prompts/refusal.py` now accepts `unconfirmed_symbols: list[str]` and conditionally inserts the same `IMPORTANT: <backtick-quoted unconfirmed symbol> ... not a valid PyTorch API ...` block that Direction A added to the reasoning prompt.
- `build_hybrid_prompt` passes `unconfirmed_symbols` to `build_refusal_prompt`.
- When `unconfirmed_symbols` is empty, the refusal prompt is byte-equal to the previous refusal prompt, so `data_loading_003` and any other refusal-branch item without unconfirmed query symbols stays unchanged.
- After Direction C the smoke result is:
  - `9 correct / 3 partially_correct / 0 incorrect = 0.75`
  - `hallucination_refusal_accuracy = 1.0`
  - file: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-212010.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-20-212010.json)
- The new `hallucination_refusal_003` answer under Direction C is a real refusal, not a scorer false positive:
  - `"You should not switch from optim.Adam to optim.AdamFast. optim.AdamFast is not a valid PyTorch API and cannot be verified from the retrieved PyTorch docs."`
  - no fabricated behavior for `optim.AdamFast`
  - the phrase `not a valid` appears as a fuzzy match to the scorer constraint, not as a literal echo of the instruction text
- The 11 other smoke items stayed byte-equal across Directions A, B, and C, matching the greedy decoding invariant:
  - their labels are identical across all three runs
  - this confirms that Direction C only affected the target topology and did not pollute normal items

### Scorer and routing lessons

- When a template uses abstract wording like `"this API"` in a context where multiple APIs appear, the model can drift; inlining the specific unconfirmed symbol name as a backticked token removed that ambiguity.
- `must_include` phrases like `"not a valid"` can be satisfied by literal instruction parroting, which is a scorer false positive that must be detected by reading the full answer, not just the label.
- `greedy decoding + byte-equality on unchanged items` is a strong regression harness and should continue to be used as the default iteration unit for prompt and router work.

### Session Part 3: Core verification (evening)

- The Direction C stack was rerun against the full `benchmark_core_pytorch.jsonl` (36 items).
- Result: `28 correct / 7 partially_correct / 1 incorrect = 0.7778`.
  - previous 1.5B + hybrid core best: `25 / 8 / 3 = 0.6944`
  - delta: `+3 correct`, `-1 partially_correct`, `-2 incorrect`
  - `hallucination_refusal_accuracy`: `0.75 -> 1.0`
  - `citation_support_rate`: `0.9667` (unchanged)
- Per-category accuracy on the new core run:
  - `autograd = 0.75`
  - `data_loading = 0.75`
  - `debugging = 1.0`
  - `dtype_device = 0.75`
  - `hallucination_refusal = 1.0`
  - `nn_module_modes = 0.75`
  - `optim_training_loop = 0.5`
  - `shape_ops = 0.75`
  - `tensor_creation = 0.75`
- Improvements (5 items):
  - `autograd_001`: `partially_correct -> correct`
  - `data_loading_002`: `partially_correct -> correct`
  - `debugging_002`: `incorrect -> correct` (benefited from `must_include_any_of`)
  - `hallucination_refusal_003`: `incorrect -> correct` (Direction C target, real behavioral refusal, not scorer false positive)
  - `optim_training_loop_002`: `incorrect -> correct` (benefited from `must_include_any_of`)
- Regressions (3 items):
  - `autograd_002`: `correct -> partially_correct`
    - answer fact is still correct, but the new generation says "gradient" instead of `.grad` so the `must_include` rule missed one literal token
    - scorer wording fragility, not a model or retrieval regression
    - remediation: migrate to `must_include_any_of`, do not change the prompt
  - `optim_training_loop_001`: `correct -> partially_correct`
    - new generation uses "sets ... to zero" rather than "clear"
    - same scorer wording fragility class as above
    - remediation: migrate to `must_include_any_of`, do not change the prompt
  - `tensor_creation_002`: `partially_correct -> incorrect`
    - this one is a real content regression, not a scorer issue
    - the new answer contradicts itself: it says `torch.from_numpy` creates a tensor from a NumPy ndarray while `torch.tensor` copies, then a sentence later claims `torch.from_numpy always makes a full copy`
    - that triggers `must_not_include: ["always copies"]`
    - likely cause: the corpus cache was rebuilt this session when `Tensor.permute` was appended to `shape_ops.jsonl`, which shifts BM25 idf and can change the top_k for unrelated items
    - do not diagnose via prompt churn; first compare `retrieval_debug` between the 2026-04-16 and 2026-04-20 core runs for this item
- Critical invariant check:
  - `debugging_001` held at `correct` (Tensor.permute corpus mitigation worked; the new answer additionally picked up `Tensor.reshape` as an expected symbol)
  - all four `hallucination_refusal` items are now `correct`, explaining the `hallucination_refusal_accuracy = 1.0` jump

## Current Conclusion

- The current best stable smoke result on `Qwen/Qwen2.5-1.5B base + hybrid` is:
  - `9 correct / 3 partially_correct / 0 incorrect = 0.75`
  - `hallucination_refusal_accuracy = 1.0`
  - this is a real behavioral improvement, not a scorer alignment effect
- The current stable prompt stack is:
  - `exact`
  - `reasoning` with optional inlined `IMPORTANT: <unconfirmed_symbols>` block
  - `refusal` with optional inlined `IMPORTANT: <unconfirmed_symbols>` block
  - `router` that also triggers refusal on mixed real/fake API topology (`matched > 0 AND unconfirmed > 0`)
- The corpus state was also updated during this session:
  - `Tensor.permute` is now in the symbol index via a new `api_docs_tensor_permute` source doc
  - without this, Direction B would have re-routed `debugging_001` in core and regressed it
- The scorer now supports `must_include_any_of` for acceptable-phrasing groups, which removed two scorer-wording false negatives (`debugging_002` and `optim_training_loop_002`) without any prompt or model change.
- The 1.5B + hybrid core best on record is now `28 / 7 / 1 = 0.7778`, up from `25 / 8 / 3 = 0.6944` on 2026-04-16. The smoke improvement generalized to core with net `+3 correct, -2 incorrect` and the Direction C target `hallucination_refusal_003` resolved as a real refusal.

## Next Step

- Migrate the two scorer-wording regressions to `must_include_any_of` rather than changing the prompt:
  - `autograd_002`: accept `.grad` or `gradient` (or equivalent) for the "named gradient buffer" group
  - `optim_training_loop_001`: accept `clear`, `reset`, or `sets ... to zero` (or equivalent) for the "zero out the grads" group
  - these are scorer-wording issues, not generation issues; do not iterate on the prompt for them
- Investigate `tensor_creation_002` regression at the retrieval layer first:
  - diff `retrieval_debug.retrieved_docs` between the 2026-04-16 and 2026-04-20 core runs for this item
  - determine whether adding `api_docs_tensor_permute` to `shape_ops.jsonl` shifted the BM25 ranking for unrelated tensor_creation queries
  - if yes, that is a corpus-rebuild side effect and should be logged as a known risk of the `one new doc -> global idf shift` dynamic
  - if the retrieved context is unchanged, reopen the investigation at the generation layer
- Smoke-only follow-up on the remaining 3 `partially_correct` items (`shape_ops_001`, `dtype_device_003`, `data_loading_003`) is lower priority:
  - these are not hallucination or routing issues
  - they are likely either model wording gaps or scorer `must_include` strictness on multi-keyword rules
  - do not launch a new prompt-churn branch on them without first checking whether they are scorer wording limitations
- Freeze the current Direction C prompt / router / refusal stack as the new 1.5B + hybrid mainline. Do not revisit refusal prompt wording unless a later run exposes a new mixed real/fake topology case.

## Other Important Info

- Direction A alone demonstrated how easy it is to hit a scorer false positive on hallucination refusal: the model can earn `not a valid` credit by literally echoing the instruction string while still fabricating API behavior above it. Manual answer review on refusal items should remain a required step for any result involving `hallucination_refusal`.
- Direction B and C depended on the `Tensor.permute` corpus addition to avoid collateral damage on `debugging_001`. Any future expansion of the mixed real/fake refusal route should re-scan `benchmark_core_pytorch.jsonl` for the `matched > 0 AND unconfirmed > 0` topology before rollout.
- The Direction A/B/C sequence is a good template for future prompt iterations: locate the failing item, enumerate candidate variables, enforce byte-equality on untouched items, and treat scorer headline number and behavioral correctness as two separate checks.
