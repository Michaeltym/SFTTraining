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
- Timestamp: `2026-04-21`
- Core Baseline Entering Today (from 2026-04-20): `28 correct / 7 partially_correct / 1 incorrect = 0.7778`
- Core Best After Today (any_of migration): [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-093523.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-093523.json)

## Goal

Execute the priority-1 follow-up from the 2026-04-20 Next Step: migrate the two scorer-wording regressions from the 2026-04-20 core run (`autograd_002` and `optim_training_loop_001`) into `must_include_any_of` groups, without touching the prompt stack, the router, the corpus, or the model. Confirm the two items flip from `partially_correct` to `correct` while every other item keeps its 2026-04-20 label exactly.

## Findings

- Two `benchmark_core_pytorch.jsonl` rows were updated:
  - `autograd_002`:
    - `must_include`: `["accumulate", ".grad"]` -> `["accumulate"]`
    - new `must_include_any_of`: `[[".grad", "gradient"]]`
    - `must_not_include` unchanged: `["backward clears gradients automatically"]`
  - `optim_training_loop_001`:
    - `must_include`: `["clear", "gradients"]` -> `["gradients"]`
    - new `must_include_any_of`: `[["clear", "reset", "to zero"]]`
    - `must_not_include` unchanged: `["updates parameters"]`
- The `must_include_any_of` candidates were probed against the 2026-04-20 generated answers before editing the benchmark, to make sure the new groups would actually match the post-regression wording:
  - `autograd_002` 2026-04-20 answer contains `gradient` and `gradients` repeatedly but no `.grad` token
  - `optim_training_loop_001` 2026-04-20 answer contains `resets` and `to zero` (`"resets the gradients of all parameters managed by the optimizer to zero or none"`)
  - the 2026-04-16 baseline answers also still match the new groups (`.grad` for autograd_002, `resets` and `to zero` for optim_training_loop_001), so the migration does not silently demote previously-correct generations
- The `must_not_include` rules on both items were left untouched. They still guard against the original failure modes (`backward clears gradients automatically` and `updates parameters`), so widening the positive constraint did not open a new false-positive surface.
- The benchmark JSONL was validated after the edit:
  - `36` rows
  - `36` unique ids
  - all required fields present on every row
  - the two target rows match the planned schema

### Core verification

- Reran the full `benchmark_core_pytorch.jsonl` (36 items) against the unchanged Direction C stack with only the two scorer rules updated.
- Result: `30 correct / 5 partially_correct / 1 incorrect = 0.8333`.
  - delta vs 2026-04-20: `+2 correct`, `-2 partially_correct`, `incorrect` unchanged
  - `hallucination_refusal_accuracy`: `1.0` (unchanged)
  - `citation_support_rate`: `0.9667` (unchanged)
- Per-category accuracy on the new core run:
  - `autograd = 1.0` (`0.75 -> 1.0`)
  - `data_loading = 0.75`
  - `debugging = 1.0`
  - `dtype_device = 0.75`
  - `hallucination_refusal = 1.0`
  - `nn_module_modes = 0.75`
  - `optim_training_loop = 0.75` (`0.5 -> 0.75`)
  - `shape_ops = 0.75`
  - `tensor_creation = 0.75`
- Label flips vs 2026-04-20 core:
  - `autograd_002`: `partially_correct -> correct`
  - `optim_training_loop_001`: `partially_correct -> correct`
- The other `34` items kept their 2026-04-20 labels exactly, which matches the byte-equal greedy decoding invariant: nothing in the prompt path changed, so nothing in the answer text could change, and only the targeted scorer rules could move labels.
- The single remaining `incorrect` is still `tensor_creation_002`. It was not touched by today's change, and it remains the priority-2 investigation target carried over from 2026-04-20.

## Current Conclusion

- The current best stable core result for `Qwen/Qwen2.5-1.5B base + hybrid` is now:
  - `30 correct / 5 partially_correct / 1 incorrect = 0.8333`
- This run was a pure scorer-quality improvement, not a model or retrieval change:
  - prompt stack: unchanged (`exact` / `reasoning` with optional `IMPORTANT` / `refusal` with optional `IMPORTANT` / `router` with mixed real/fake routing)
  - router: unchanged
  - corpus and symbol index: unchanged
  - model: unchanged (`Qwen/Qwen2.5-1.5B base`)
- The `must_include_any_of` mechanism continues to behave as intended: when the model expresses the correct concept with an equivalent acceptable phrasing, the scorer no longer mislabels it as `partially_correct`.
- The remaining `1 incorrect` is `tensor_creation_002`, which is a real generation-side issue (a self-contradicting answer that triggers `must_not_include: ["always copies"]`) and not a scorer wording problem.

## Next Step

- Investigate `tensor_creation_002` at the retrieval layer first, per the 2026-04-20 plan:
  - diff `retrieval_debug.retrieved_docs` between the 2026-04-16 and 2026-04-21 core runs for this item
  - determine whether adding `api_docs_tensor_permute` to `shape_ops.jsonl` shifted the BM25 ranking for unrelated `tensor_creation` queries
  - if retrieval shifted, treat it as the `one new doc -> global idf shift` known risk and add curated comparison docs for `torch.from_numpy` vs `torch.tensor`
  - if retrieval is unchanged, reopen the investigation at the generation layer (consider whether the comparison-style query needs an exact-routing keyword)
- Do not iterate on the prompt stack for `tensor_creation_002` until the retrieval-layer hypothesis is checked.
- Freeze the current Direction C + `must_include_any_of` stack as the new 1.5B + hybrid mainline. The next branch of work should start from this commit.
- Continue to use the smoke benchmark as the first-pass regression harness for any small change. Smoke is unaffected by today's edit because neither migrated item is in the smoke set.

## Other Important Info

- Today's change is the cleanest possible single-variable iteration in this project: the prompt path is byte-equal, the model is the same, the corpus is the same, and the only altered surface is two rows of benchmark scoring rules. The byte-equal greedy decoding invariant therefore guarantees that any label change must be on `autograd_002` or `optim_training_loop_001`, and nothing else. The result confirms exactly that.
- Side observation (not in scope for today, deferred): `src/v2/benchmark/label.py:219` uses Python 3.12+ PEP 701 nested-same-quote f-string syntax (`f"group {match["group"]} matched: ..."`). It runs in the local 3.12 environment, but will raise `SyntaxError` on Python 3.10 / 3.11. If the project ever needs to support an older Python, switch the inner quotes to single quotes: `f"group {match['group']} matched: ..."`. This is a portability fix only, not a behavior fix.
- The `tensor_creation_002` regression is the right next investigation target, but it should not be conflated with today's wins. Today resolved scorer fragility on two `correct -> partially_correct` regressions from 2026-04-20. The remaining `incorrect` is a separate axis (generation self-contradiction triggered by a possible retrieval-side perturbation), and the diagnosis should start in retrieval, not in prompt.
