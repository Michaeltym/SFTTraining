## Run Info

- System Type: `v2 hybrid retrieval benchmark`
- Primary Generator Model: `Qwen/Qwen2.5-1.5B`
- Comparison Generator Model: `Qwen/Qwen2.5-0.5B`
- 0.5B Generator Type: `SFT + LoRA checkpoint`
- 1.5B Generator Type: `base model`
- Retriever Type: `symbol retrieval + lexical retrieval + hybrid merge`
- Prompt Type: `hybrid context prompt, top_k=3`
- Corpus Source Directory: [pytorch_docs](../../data/source/pytorch_docs)
- Corpus Cache: [pytorch_corpus.jsonl](../../data/output/cache/pytorch_corpus.jsonl)
- Symbol Index Cache: [pytorch_symbol_index.json](../../data/output/cache/pytorch_symbol_index.json)
- Benchmark Data: [benchmark_core_pytorch.jsonl](../../data/eval/benchmark_core_pytorch.jsonl)
- Timestamp: `2026-04-16`
- Current Best 0.5B Hybrid Result: [hybrid-hybrid-Qwen-Qwen2.5-0.5B-2026-04-16-154623.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-0.5B-2026-04-16-154623.json)
- Current Best 1.5B Hybrid Result: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-164008.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-164008.json)

## Goal

Build a cleaner v2 retrieval pipeline on top of curated PyTorch docs and determine whether the current bottleneck is:

- corpus coverage
- retrieval wiring / ranking
- answer-layer model capacity

The key decision for this phase was whether the project should keep pushing the `0.5B + SFT` answerer or move to a stronger answerer with the current hybrid retriever.

## Findings

- The v2 hybrid retrieval path now works end to end:
  - source docs -> corpus build -> symbol index build -> hybrid retrieval -> benchmark result save
- The first major improvements came from expanding the curated docs by failure slice:
  - `tensor_creation`
  - `shape_ops`
  - `dtype_device`
  - `nn_module_modes`
  - `autograd`
  - `debugging`
- Early hybrid runs on the tiny initial corpus were misleadingly weak because the corpus only covered a few `optim_training_loop` docs. Once the corpus expanded, hybrid performance improved substantially.
- The current best `0.5B + SFT + hybrid` result is:
  - `17 correct / 7 partially_correct / 12 incorrect`
  - `overall_accuracy = 0.4722`
- Category strengths for the current best `0.5B + SFT + hybrid` run:
  - `autograd = 0.75`
  - `dtype_device = 0.75`
  - `nn_module_modes = 0.75`
  - `tensor_creation = 0.75`
- Persistent weak areas under the `0.5B + SFT` answerer:
  - `shape_ops = 0.25`
  - `debugging = 0.25`
  - `optim_training_loop = 0.25`
  - `hallucination_refusal = 0.0`
- Several experiments showed that the remaining bottleneck was no longer mainly retrieval coverage:
  - some questions already retrieved the right docs but the `0.5B` answerer still got the behavior or shape wrong
  - refusal prompting on `0.5B` was unstable and often damaged normal questions
- A direct stronger-answerer test was run with the same hybrid retrieval but a larger base model:
  - `Qwen/Qwen2.5-1.5B base + hybrid`
- That result clearly beat the current `0.5B + SFT + hybrid` system:
  - `21 correct / 5 partially_correct / 10 incorrect`
  - `overall_accuracy = 0.5833`
- Real gains from `1.5B base + hybrid` showed up in answer-layer-sensitive areas:
  - `autograd = 1.0`
  - `dtype_device = 1.0`
  - `nn_module_modes = 1.0`
  - `debugging = 0.75`
- This strongly suggests that the answer model, not retrieval, is now the main limiter.
- Benchmark scoring still has known limitations:
  - some `correct` labels are false positives because the scorer is rule-based
  - some `hallucination_refusal` answers are semantically correct refusals but do not match the current `must_include` wording
- Because of that, the absolute percentages are not perfect truth, but the model-capacity trend is still strong and consistent.
- A failed branch explored refusal prompting:
  - wiring bugs and later over-aggressive trigger rules temporarily degraded full-benchmark results
  - even after fixing the wiring, refusal prompting did not produce a net gain on the full benchmark
  - that line should not be treated as the main path right now
- Performance profiling showed that benchmark slowness is mostly generation time, not retrieval:
  - per-question total time was nearly identical to the generation-timed block
  - retrieval still has some optimization opportunities, but it is not the main latency source right now
- A `hybrid_with_base_model` mode was added so hybrid retrieval can be tested with a base model without requiring a checkpoint.
- `Qwen/Qwen2.5-3B` was also considered, but local runtime cost is high enough that it is not currently the preferred working setup.
- `1.5B + SFT/LoRA` would likely require running on RunPod or another remote machine, which is possible but adds significant iteration friction.
- A follow-up stronger-answerer comparison was also run with:
  - `Qwen/Qwen2.5-3B base + hybrid`
- The `3B` result did not beat `1.5B` overall:
  - `21 correct / 7 partially_correct / 8 incorrect`
  - `overall_accuracy = 0.5833`
- Compared with `1.5B base + hybrid`, `3B` had:
  - the same overall accuracy
  - better `hallucination_refusal`
  - but regressions on some normal API questions
- The practical conclusion is:
  - `3B` is not strong enough to justify becoming the main local answerer
  - `1.5B base + hybrid` remains the better balance of quality and local usability
- After aligning the `hallucination_refusal` benchmark wording from `does not exist` to the more stable phrase `not a valid`, the same `1.5B base + hybrid` system improved from:
  - `21 correct / 5 partially_correct / 10 incorrect`
  - `overall_accuracy = 0.5833`
  to:
  - `24 correct / 5 partially_correct / 7 incorrect`
  - `overall_accuracy = 0.6667`
- This increase mostly reflected scorer alignment on refusal items rather than a sudden retrieval or model jump.
- Additional curated docs were then added for:
  - `data_loading`
  - `optim_training_loop`
- Once those docs were actually included in the rebuilt corpus cache, `data_loading` improved:
  - `0.25 -> 0.5`
- `optim_training_loop` docs also helped, but that category quickly shifted from pure coverage problems to a mix of:
  - scorer wording issues
  - answer completeness / reasoning quality
- A prompt refactor split hybrid prompting into three files plus a router:
  - `exact`
  - `reasoning`
  - `refusal`
  - `router`
- This refactor was stable and did not itself degrade results.
- A prompt calibration pass then improved several reasoning-heavy categories:
  - `shape_ops: 0.25 -> 0.75`
  - `optim_training_loop: 0.25 -> 0.5`
- A later routing refinement kept:
  - the split prompt structure
  - the expanded `EXACT_PROMPT_QUERY_KEYWORDS`
- That routing refinement also confirmed that the exact/reasoning split could be kept without degrading the strong `1.5B` path.
- The strongest stable `1.5B base + hybrid` result after those changes is:
  - `25 correct / 8 partially_correct / 3 incorrect`
  - `overall_accuracy = 0.6944`
  - file: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-212420.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-212420.json)
- The stable category profile for this best local run is:
  - `autograd = 0.75`
  - `data_loading = 0.5`
  - `debugging = 0.75`
  - `dtype_device = 0.75`
  - `hallucination_refusal = 0.75`
  - `nn_module_modes = 0.75`
  - `optim_training_loop = 0.5`
  - `shape_ops = 0.75`
  - `tensor_creation = 0.75`
- A later experiment added a query-symbol heuristic called `has_unmatched_query_symbol` to force refusal on mixed real/fake API queries.
- That change was a net negative:
  - it did not reliably fix `hallucination_refusal_003`
  - it did damage normal questions such as `debugging_001`
- That heuristic was rolled back.
- The prompt router keeps the expanded `EXACT_PROMPT_QUERY_KEYWORDS`, but does not use the aggressive unmatched-symbol refusal rule.
- A follow-up run after removing `has_unmatched_query_symbol` confirmed that:
  - the expanded exact-routing keywords can stay
  - the stable best result remains `25 / 8 / 3`
  - file: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-213423.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-2026-04-16-213423.json)

## Current Conclusion

- The current best local working system is:
  - `Qwen/Qwen2.5-1.5B base + v2 hybrid retrieval`
- The current best stable benchmark result for that system is:
  - `25 correct / 8 partially_correct / 3 incorrect`
  - `overall_accuracy = 0.6944`
- It is more practical than trying to force `1.5B + SFT` locally.
- It is also stronger than the current `0.5B + SFT + hybrid` system on the benchmark.
- `Qwen/Qwen2.5-3B base + hybrid` does not provide enough additional value to replace `1.5B` as the main local path.
- The project should treat `1.5B base + hybrid` as the main local answerer for the next phase.
- The `0.5B + SFT + hybrid` path remains useful as a comparison baseline, but it should not be the main answerer anymore.
- The current prompt structure should remain:
  - `exact`
  - `reasoning`
  - `refusal`
  - `router`
- The current router should keep:
  - the prompt split
  - the expanded exact-routing keywords
- The current router should not re-enable:
  - `has_unmatched_query_symbol`
- The current benchmark interpretation should also remember:
  - `debugging_002` and `optim_training_loop_002` look like scorer wording limitations rather than model limitations
  - these should be revisited either by scorer improvement or by curated accepted-phrase lists, not by more prompt churn

## Next Step

- Freeze the current local mainline as:
  - `Qwen/Qwen2.5-1.5B base + hybrid`
- Continue targeted improvements on top of that system, but do not return to broad corpus expansion by default.
- The next iteration should focus on the remaining hard cases rather than adding many more docs.
- In particular:
  - inspect the remaining `incorrect` items one by one
  - treat `hallucination_refusal_003` as a mixed real/fake API policy problem, not a corpus coverage problem
  - avoid aggressive refusal heuristics on the mainline
- Freeze the current mainline before opening any new branch of work.
- Do not spend more mainline iteration budget trying to rescue already-fixed scorer wording issues through prompt churn.
- Do not spend more mainline iteration budget on refusal prompting for `0.5B`.
- If later needed, run a more formal remote experiment:
  - `Qwen/Qwen2.5-1.5B + SFT/LoRA + hybrid`
- For future benchmark quality:
  - keep using manual spot checks on high-risk comparison/debugging items
  - do not overfit benchmark `must_not_include` to one model snapshot

## Other Important Info

- Benchmark scoring limitations should be remembered when reading the current numbers:
  - `tensor_creation_002` and similar comparison questions can still expose rule-based scorer blind spots
  - `hallucination_refusal` still undercounts mixed real/fake API failures such as `hallucination_refusal_003`
- The current best run is not completely “clean truth”:
  - some remaining `correct` labels are still vulnerable to scorer shallow matching
  - some remaining `partial` labels are likely wording/scorer strictness rather than retrieval failures
- The repo `AGENTS.md` now explicitly records that benchmark rules should not be tailored to catch one current generated sentence pattern.
- Local practicality matters here:
  - `1.5B base` is strong enough to justify using it now
  - `1.5B + SFT` and `3B` remain optional later experiments, not current blockers
