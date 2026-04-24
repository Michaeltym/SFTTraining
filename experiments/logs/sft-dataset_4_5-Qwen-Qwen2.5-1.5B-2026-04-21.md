## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Datasets this log covers: `dataset_4` (stratified 50-row mix), `dataset_5` (= dataset_4 + mandatory rows for argmax / dropout-alone)
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT` (same recipe as the pilot)
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1`
- Train rows: `50`; Val rows: `12`
- LoRA Config (unchanged from pilot): `r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`
- Trainable params: `18,464,768 / 1,562,179,072` (`1.1820%`)
- Device: `mps` (Apple Silicon)
- Post-SFT checkpoints:
  - [Qwen-Qwen2.5-1.5B-dataset_4-8-0.0001.pt](../../data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_4-8-0.0001.pt)
  - [Qwen-Qwen2.5-1.5B-dataset_5-8-0.0001.pt](../../data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_5-8-0.0001.pt)
- Benchmark results (hybrid mode, post-fix retrieval):
  - `dataset_4` run: [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-205027.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-205027.json)
  - `dataset_5` run: [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-212112.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-212112.json)
- Baseline reference (post-fix base + hybrid): [rescored `171427`](../eval_results/benchmark/rescored/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-171427-rescored.json) `29/7/0 = 0.806`
- Timestamp: `2026-04-21`

## Goal

The pilot log (`sft-dataset_3-Qwen-Qwen2.5-1.5B-pilot-2026-04-21.md`) closed with a concrete diagnosis: the 50-row `dataset_3` slice at `seed=42` was dominated by `shape_ops` (56%) and `debugging` (42%) and had zero coverage of `autograd`, and the per-category SFT delta tracked that mix almost exactly. The next experiment was supposed to hold size fixed at 50 rows and **rebalance categories** to isolate `mix` as the single changing variable.

This log covers two stacked attempts:

- **`dataset_4` (mix hypothesis test).** Build a category-stratified 50-row slice across the nine benchmark categories using a rule-based classifier, so every category except `data_loading` has non-zero coverage. Budget-per-category hand-picked from the dataset_3 pool sizes.
- **`dataset_5` (targeted FP intervention).** After the dataset_4 benchmark revealed two recurring false-positives (`nn_module_modes_003` dropout-vs-BN, `debugging_004` argmax-returns-scalar), add a mandatory-row pre-fill mechanism so specific training rows that directly counter those FPs are guaranteed to appear in the slice, and add a small dropout-alone addendum file so dropout has training rows that don't bundle it with batch normalization.

Both runs use the exact same LoRA recipe as the pilot. Retrieval stack is the same post-fix stack the pilot's Next Step #5 landed on.

## Findings

### Build scripts and data files

- `scripts/build_dataset_4.py`: rule-based classifier into the nine benchmark categories, then category-stratified sampling with a fixed seed. Writes `data/raw/training/dataset_4.jsonl` and `data/raw/validation/dataset_4.jsonl`.
- `scripts/build_dataset_5.py`: forked from `build_dataset_4.py`. Adds:
  - `SRC_ADDENDUM = data/raw/training/dataset_3_addendum.jsonl` — merged into the training pool with `(input, output)` dedup so repeat rows are not double-counted.
  - `MANDATORY_INPUT_SUBSTRINGS` — each category pre-fills its budget with rows whose `input` matches any listed substring, then random-samples the remaining budget. Hard-check raises if any mandatory substring did not match at least one row in the final slice.
- `data/raw/training/dataset_3_addendum.jsonl`: two new rows explaining dropout alone (no batch-norm co-mention), used as the mandatory pre-fill for the nn_module_modes budget.

### Category distribution (dataset_4 and dataset_5)

Both datasets use the same per-category budget (sum = 50). `dataset_5` differs only in which specific rows fill each category.

| category               | budget | source coverage (dataset_3) |
|------------------------|--------|------------------------------|
| `shape_ops`            | 12     | 130                          |
| `debugging`            | 8      | 24                           |
| `autograd`             | 7      | 11                           |
| `tensor_creation`      | 6      | 6                            |
| `nn_module_modes`      | 6      | 6 (+2 in addendum for ds5)   |
| `hallucination_refusal`| 6      | 19                           |
| `optim_training_loop`  | 4      | 4                            |
| `dtype_device`         | 1      | 1                            |
| `data_loading`         | 0      | 0 real coverage              |

Known gap: `data_loading` stays at 0 because `dataset_3` has no real `data_loading` rows — the only `DataLoader`-mentioning rows are fake-API refusal rows that legitimately belong in `hallucination_refusal`.

### `dataset_4` benchmark result (run `205027`)

- Scorer (pre-patch): `29 correct / 7 partially_correct / 0 incorrect = 0.8056`
- Per-category accuracy:

  | category | ds4 | base (171427) | Δ |
  |---|---|---|---|
  | `autograd`           | 0.875 | 0.750 | **+0.125** |
  | `debugging`          | 0.750 | 1.000 | **-0.250** |
  | `dtype_device`       | 1.000 | 1.000 | 0 |
  | `hallucination_refusal` | 1.000 | 1.000 | 0 |
  | `nn_module_modes`    | 0.500 | 0.750 | **-0.250** |
  | `optim_training_loop`| 0.750 | 0.750 | 0 |
  | `shape_ops`          | 0.750 | 0.750 | 0 |
  | `tensor_creation`    | 1.000 | 0.750 | **+0.250** |
  | `data_loading`       | 0.750 | 0.750 | 0 |

- Mix hypothesis, partial confirmation:
  - Autograd recovered (`+0.125` vs baseline, much better than the pilot's `-0.25`). Stratification delivered what it promised for the zero-coverage category.
  - Shape_ops held flat instead of regressing — consistent with the pilot's "over-sampling drives regression" explanation now that shape_ops is 24% of the slice instead of 56%.
  - BUT: two categories regressed vs baseline, `debugging` (`-0.25`) and `nn_module_modes` (`-0.25`). Both are traceable to specific FP phrasings (below).

### `dataset_4` FP/FN audit

Per the `Benchmark Scorer Audit Rule` in CLAUDE.md, walked all `correct` and `partially_correct` labels for hidden factual errors.

- `nn_module_modes_003` (scorer: correct) — **false positive**. Answer: "After calling model.eval(), dropout layers are disabled […] which changes the behavior of dropout layers to use their stored running statistics instead of computing statistics from the current batch." The first clause is correct, the second clause inverts dropout and BN semantics. Scorer passed because `dropout` and `disabled` were present.
- `debugging_004` (scorer: partially_correct) — **FP-at-partial severity**. Answer: "torch.argmax(x, dim=1) returns the index of the overall maximum in the flattened tensor as a scalar tensor […]". That is `torch.argmax(x)` behavior, not `argmax(x, dim=1)`. The existing `must_not_include_regex: "flattened tensor as a scalar"` fires and demotes to partial, but the severity is hand-audit-incorrect.

Root cause, per coverage check against `dataset_3`:

- Dropout FP: every dropout row in dataset_3 bundles dropout with batch normalization (`"dropout and batch normalization both..."`). The model never sees dropout explained on its own, so it has no signal that `running statistics` is a BN-only concept.
- Argmax FP: `dataset_4` happens to contain **zero** argmax training rows at `seed=42`, even though dataset_3 has 13 argmax rows in its `shape_ops` pool. `dim=1` shape semantics are never in the training signal for this slice.

### `dataset_5` intervention — mandatory row mechanism

`scripts/build_dataset_5.py` pre-fills each category's budget from rows matching any of:

```python
MANDATORY_INPUT_SUBSTRINGS = (
    "shape does torch.argmax(x, dim=0) return",
    "shape does torch.argmax(x, dim=1) return",
    "What does a dropout layer do during training vs evaluation",
    "Why does dropout not use running statistics",
)
```

Final `dataset_5` slice: 5 mandatory rows (3 argmax in `shape_ops` — dim=0, dim=1, dim=3 picked randomly once the two mandatory dims had pre-filled; 2 dropout-alone in `nn_module_modes`) + 45 random rows, total 50. Distribution match with `dataset_4` by category (no budget change), different specific rows inside `shape_ops` and `nn_module_modes`.

### `dataset_5` benchmark result (run `212112`)

- Scorer (pre-patch): `29/7/0 = 0.8056` — **identical to dataset_4 at the scorer level**.
- Per-category accuracy:

  | category | ds5 | ds4 | Δ |
  |---|---|---|---|
  | `autograd`           | 1.000 | 0.875 | **+0.125** |
  | `nn_module_modes`    | 1.000 | 0.500 | **+0.500** (at the scorer level — two dropout-adjacent rows now pass) |
  | `data_loading`       | 0.500 | 0.750 | **-0.250** |
  | `debugging`          | 0.500 | 0.750 | **-0.250** |
  | `shape_ops`          | 0.500 | 0.750 | **-0.250** |
  | `optim_training_loop`| 0.750 | 0.750 | 0 |
  | `dtype_device`       | 1.000 | 1.000 | 0 |
  | `hallucination_refusal` | 1.000 | 1.000 | 0 |
  | `tensor_creation`    | 1.000 | 1.000 | 0 |

- Scorer total unchanged (29/7/0) is because six individual rows flipped — three up, three down:
  - UP: `autograd_003` partial→correct, `nn_module_modes_001` partial→correct, `nn_module_modes_004` partial→correct
  - DOWN: `data_loading_002` correct→partial, `debugging_002` correct→partial, `shape_ops_001` correct→partial

### `dataset_5` FP/FN audit — the targeted fixes did not land

- `nn_module_modes_003` (scorer: correct) — **same wrong answer as ds4**. Answer still ends with "…changes the behavior of dropout layers to use their stored running statistics instead of computing statistics from the current batch." Two dropout-alone training rows were insufficient to override the base model's bundled-with-BN prior in 1 epoch of LoRA SFT.
- `debugging_004` (scorer: partially_correct) — **same wrong answer as ds4**. "…index of the overall maximum in the flattened tensor as a scalar tensor…". Three mandatory argmax training rows did not flip the `dim=1` explanation either.

Hand-regrade: dataset_5 ≈ `27/7/2 = 0.75` (same as dataset_4). The scorer over-reports by ~0.03–0.06.

### Scorer patches (option 丙 per CLAUDE.md's Benchmark Scorer Audit Rule)

Because the same FP pattern now shows up in **two** SFT runs (ds4 and ds5), the rule mandates promoting the pattern to a benchmark-item rule instead of re-running training. Two-row patch applied to `data/eval/benchmark_core_pytorch.jsonl`:

- `nn_module_modes_003`:
  - `must_not_include` (hard, forces `incorrect`) — added six precise-phrase entries: `"dropout layers to use their stored running statistics"`, `"dropout layers to use running statistics"`, `"dropout layers use running statistics"`, `"dropout uses running statistics"`, `"dropout uses their running statistics"`, `"dropout uses the running statistics"`.
  - `must_not_include_regex` (soft, demotes to partial) — added one pattern as a phrasing-variant catch-all: `"dropout (layers?\\s+)?(to\\s+)?(uses?|using)\\s+(their\\s+|its\\s+)?(stored\\s+)?(the\\s+)?running (statistics|stats)"`. Verified that negations like `"dropout does not use running statistics"` do **not** match (the tokens between `dropout` and `use` are `does not`, which is neither `layers?` nor `to`, so the regex rejects).
- `debugging_004`:
  - Promoted `"flattened tensor as a scalar"` from `must_not_include_regex` (soft) to `must_not_include` (hard). Existing `"argmax returns the maximum values"` left in place.

### Rescore impact across the three archived runs

Run via a minimal self-contained rescorer (`/sessions/vigilant-upbeat-ptolemy/scripts/rescore_minimal.py`) because the project venv lives on the Mac host and the sandbox Python 3.10 cannot import the project's `NotRequired`-using TypedDicts. Logic is a 1:1 port of `src/v2/benchmark/label.py`; rapidfuzz installed for fuzzy-match parity.

| Run                                   | Pre-patch     | Post-patch      | Label changes |
|---------------------------------------|---------------|------------------|---------------|
| baseline `171427` (rescored)          | `29/7/0 = 0.806` | `29/7/0 = 0.806` | **none**       |
| `dataset_4` SFT (`205027`)            | `29/7/0 = 0.806` | `28/6/2 = 0.778` | `nn_module_modes_003` correct→incorrect, `debugging_004` partial→incorrect |
| `dataset_5` SFT (`212112`)            | `29/7/0 = 0.806` | `28/6/2 = 0.778` | same two flips |

Critically, the baseline answers for both patched items are substantively correct and are **not** affected by the patch:

- baseline `nn_module_modes_003` answer: "After calling model.eval(), dropout layers are disabled and pass all inputs through without zeroing any elements. This is because the evaluation mode disables dropout layers, which randomly zero out a fraction of input elements at each forward pass to prevent overfitting." — no forbidden phrase hits, stays `correct`.
- baseline `debugging_004` answer: explains `torch.max` for values and `torch.argmax` for indices correctly. — no forbidden phrase hits, stays `correct`.

The patch is surgical: it only flips answers that contain the specific wrong phrasings SFT produced.

### Where this leaves the scorecard

Apples-to-apples against baseline on the **patched** scorer:

| Run                       | accuracy | vs baseline |
|---------------------------|----------|-------------|
| baseline `171427` (post-fix base + hybrid) | 0.806 | — |
| `dataset_4` SFT           | 0.778 | **-0.028** |
| `dataset_5` SFT           | 0.778 | **-0.028** |

So the clean read is: neither SFT run beats the post-fix baseline. The mandatory-row intervention in ds5 did not produce a net gain; it reshuffled which specific items pass the scorer without moving the total.

### What this tells us about mandatory rows in 1-epoch LoRA SFT

The ds5 failure mode is informative independent of the scorecard: **2–3 counter-evidence rows, scattered among 45 other rows, are not enough to override a base-model bias in 1 epoch of LoRA SFT at `r=16, alpha=32, lr=1e-4`**. The base model has seen the wrong pattern (dropout ↔ running statistics, `argmax(dim=1)` ↔ scalar) many times in pretraining, and a handful of one-off contradictions don't dent that prior when they're not phrased as direct refutations.

This motivates the `dataset_6` plan below.

## Next Step

**Option B from the dataset_5 debrief: direct-negation anti-pattern rows.** Instead of adding more generic explainer rows, add training rows whose input question is the exact wrong belief we want to overwrite, and whose output answer opens with "No, …" and then gives the correct mechanism. Stronger training signal per row than a generic explainer, and directly addresses the two known FPs.

Target additions to a new `data/raw/training/dataset_6_addendum.jsonl`:

- **3 dropout anti-pattern rows** refuting the BN-running-stats conflation.
- **3 argmax anti-pattern rows** refuting the scalar / flatten-first misconception.

Build flow: `scripts/build_dataset_6.py` forks `build_dataset_5.py` and reads three pool sources — `dataset_3.jsonl` + `dataset_3_addendum.jsonl` + `dataset_6_addendum.jsonl` — with dedup. `MANDATORY_INPUT_SUBSTRINGS` extended to include the six new rows' question prefixes so they all make the final slice. Category budget unchanged (sum=50) — the anti-pattern rows displace a few random picks in `nn_module_modes` and `shape_ops`, which is the intended variable.

After training, compare against `171427` baseline (`29/7/0 = 0.806 on patched scorer`) and `205027` / `212112` ds4/ds5 runs (`28/6/2 = 0.778 on patched scorer`). Decision framework:

- If `dataset_6` flips `nn_module_modes_003` and `debugging_004` to answers that pass the patched scorer cleanly (no forbidden phrases triggered), the intervention worked at the SFT level and we have a reproducible recipe for "direct-negation anti-pattern rows close known FPs".
- If `dataset_6` still produces the same wrong answers, the `2–3 counter-evidence rows in 1 epoch of LoRA` floor is lower than we hoped, and the next step is either more epochs or re-framing the problem away from SFT-as-FP-fix.

## Other Important Info

- Both ds4 and ds5 training runs use the same `MODEL_NAME / BATCH_SIZE / LEARNING_RATE`, so checkpoints are discriminated only by `DATASET_NAME`. Separate checkpoint files exist under `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_4-8-0.0001.pt` and `-dataset_5-8-0.0001.pt`.
- `dataset_6` will overwrite `Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt` if rerun with the same config; archive existing benchmark JSONs for any A/B across dataset_6 variants.
- The rescoring of archived runs is **forward-only** (writes `<stem>-rescored.json` copies, original files untouched). To reproduce any table in this log, use `scripts/rescore_benchmark.py` against the archived JSONs when running on the Mac host (requires project venv). In the Cowork sandbox, use the minimal rescorer at `/sessions/vigilant-upbeat-ptolemy/scripts/rescore_minimal.py`.
- Patched benchmark items (`nn_module_modes_003`, `debugging_004`) now have broader forbidden-pattern coverage. Future runs that produce novel FP phrasings will not be caught; the audit rule still applies. When a new FP appears in the audit, add it to the appropriate `must_not_include` (hard) or `must_not_include_regex` (soft) list per the `Benchmark Scorer Audit Rule` decision tree.
- Key file deltas from this log:
  - Added: `scripts/build_dataset_4.py`, `scripts/build_dataset_5.py`
  - Added: `data/raw/training/dataset_4.jsonl`, `data/raw/validation/dataset_4.jsonl`
  - Added: `data/raw/training/dataset_5.jsonl`, `data/raw/validation/dataset_5.jsonl`
  - Added: `data/raw/training/dataset_3_addendum.jsonl`
  - Modified: `data/eval/benchmark_core_pytorch.jsonl` (two rows patched)
  - Modified: `CLAUDE.md` (added `Benchmark Scorer Audit Rule` under `Evaluation Rules`)
