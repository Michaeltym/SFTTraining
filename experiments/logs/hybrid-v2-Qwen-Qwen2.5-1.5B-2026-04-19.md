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
- Timestamp: `2026-04-19`
- Previous 1.5B Hybrid Best (core, from 2026-04-16): `25 correct / 8 partially_correct / 3 incorrect = 0.6944`
- New Smoke Baseline (2026-04-19): [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-19-210429.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-19-210429.json)

## Goal

Add a smaller, faster regression slice so that small prompt, router, and scorer iterations can be validated without running the full 36-item core benchmark every time. Determine whether the recent prompt/router stack has actually reached a plateau on hard cases.

## Findings

- A smoke benchmark track was added for faster iteration:
  - file: [benchmark_smoke_pytorch.jsonl](../../data/eval/benchmark_smoke_pytorch.jsonl)
  - size: `12` items
  - purpose: a fixed hard-case regression slice, not a miniature full-benchmark proxy
- The smoke items were selected from the core benchmark categories with an emphasis on cases that had been persistently tricky or sensitive to prompt wording.
- The smoke benchmark confirmed that the prompt/router stack had reached a local plateau:
  - several prompt hygiene tweaks changed wording but did not materially improve the smoke score
  - the stable smoke baseline on 2026-04-19 was:
    - `6 correct / 3 partially_correct / 3 incorrect`
    - `overall_accuracy = 0.5`
    - file: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-19-210429.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-smoke-2026-04-19-210429.json)
- That plateau made it clear that the next highest-ROI step was benchmark quality, not more prompt churn:
  - some items were labelled `incorrect` even though the generated answer was semantically acceptable
  - the `must_include` rule in the scorer was too brittle for cases where multiple wording variants are all acceptable

## Current Conclusion

- The 1.5B + hybrid stack has entered a plateau on the smoke slice at `6 / 3 / 3 = 0.5`.
- The binding constraint is probably not retrieval or model capacity; it is scorer strictness on hard cases.
- Smoke is intended as a fast first-pass regression harness and not a replacement for the 36-item core benchmark.

## Next Step

- Extend the benchmark scoring schema to allow acceptable equivalent phrasings:
  - introduce `must_include_any_of` groups
  - treat hits on any phrase within a group as satisfying that group
- Identify the smoke items whose `must_include` rules are the clearest wording-limitation cases and migrate them first.
- Use the smoke benchmark as the default first-pass regression set for small prompt / routing / scorer changes.
- Do not spend more mainline iteration budget trying to rescue already-fixed scorer wording issues through prompt churn.

## Other Important Info

- Smoke is a regression slice, not a proxy for the full benchmark. Aggregate percentages from smoke cannot be compared directly with core.
- The smoke baseline file should be preserved as the `before scorer cleanup` checkpoint for comparison in any later scorer-change log.
