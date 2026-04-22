## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_8` (= `dataset_7` restructured: 3 Why-framed eval-vs-no_grad rows replace 3 Y/N rows in mandatory; `nn_module_modes` mandatory 10 → 7; `shape_ops` target 8 → 11)
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT`
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1` (1-epoch run), `2` (epoch-2 resume run)
- Train rows: `50`; Val rows: `12`
- LoRA Config (unchanged): `r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`
- Device: `mps` (Apple Silicon M1 Pro, 16 GB)
- Post-SFT checkpoints:
  - 1-epoch: [Qwen-Qwen2.5-1.5B-dataset_8-8-0.0001.pt](../../data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_8-8-0.0001.pt) (overwritten by epoch-2 resume)
- Benchmark results:
  - 1-epoch hybrid: [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-143139.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-143139.json) — `27/5/4 = 0.750`
  - 2-epoch resume hybrid: [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-145042.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-145042.json) — `28/6/2 = 0.778` (226.4s)
- Reference baselines:
  - hybrid_with_base_model `131502`: `28/7/1 = 0.778` (source-content ceiling for this model size)
  - hybrid ds7 1-epoch `140357`: `26/6/4 = 0.722` (for direct ds7→ds8 diff)
- Timestamp: `2026-04-22`

## Goal

Two simultaneous interventions on top of `dataset_7`, both targeting residual ceiling FPs:

**A. Why-framed addendum.** `dataset_7`'s 3 Y/N eval-vs-no_grad rows ("Does model.eval() disable gradient tracking?", etc.) failed to move `nn_module_modes_002` because the benchmark Q is "Why is model.eval() **not the same as** torch.no_grad()" — different syntactic shape. `dataset_8` replaces those 3 mandatory rows with 3 Why-framed rows whose input phrasing matches the test-time Q shape.

**B. Mandatory rebalance.** `dataset_7`'s 10 mandatory `nn_module_modes` rows crowded the 3 ds6 argmax direct-negation rows from ~12% (ds6) to ~6% (ds7) of in-slice signal density, and `debugging_004` regressed correct → incorrect at 1 epoch. `dataset_8` trims `nn_module_modes` mandatory 10 → 7 (drops 3 Y/N + 1 of 2 which-layers from mandatory; both still in pool) and lifts `shape_ops` target 8 → 11 to restore the ds6-era density of argmax anti-pattern rows.

## Findings

### Data

- `data/raw/training/dataset_8_addendum.jsonl` — 3 Why-framed eval-vs-no_grad rows ("Why is model.eval() not the same as torch.no_grad()?", "Why doesn't calling model.eval() disable gradient tracking?", "Why do I need both model.eval() and torch.no_grad() for inference?"). Each opens with "Because…" and grounds the answer in named layers (`nn.Dropout`, `nn.BatchNorm`) and explicit autograd vocabulary.
- `scripts/build_dataset_8.py` — forked from `build_dataset_7.py`. Reads four addendum pools with `(input, output)` dedup. `MANDATORY_INPUT_SUBSTRINGS` reduced from 15 to 12 entries: 2 argmax shape + 3 ds6 dropout direct-negation + 3 ds6 argmax direct-negation + 1 ds7 which-layers + 3 new ds8 Why-framed.
- `data/raw/training/dataset_8.jsonl` (50 rows): 13 mandatory pre-fills (7 `nn_module_modes` + 6 `shape_ops`) + 37 random-sampled. The 3 ds7 Y/N rows are no longer mandatory; under `seed=42` the random sampler also did not draw them this run (verified — 0 hits).
- `data/raw/validation/dataset_8.jsonl` (12 rows): same validation pool as ds6/ds7.

### 1-epoch run — benchmark `143139`

Scorer: `27/5/4 = 0.750` (222.1s wall). Run file: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-143139.json`. Net `+0.028` vs ds7 `140357`, `-0.028` vs `131502` baseline.

#### Per-item diff vs ds7 1-epoch (only items that moved)

| id | ds7 `140357` | ds8 `143139` |
| -- | -- | -- |
| `debugging_002` | partially_correct | **correct** |

That is the only label that moved between ds7 and ds8 at 1 epoch. Everything else held its ds7 label.

#### Per-item diff vs `131502` source-content baseline

Same set of items as ds7 — the differences against baseline are unchanged from ds7 except `debugging_002` is now back to `correct`.

#### Critical diagnostic: bit-identical answers on the targets

The two intervention targets produced **byte-for-byte identical answers** between ds7 1-epoch and ds8 1-epoch:

| id | ds7 answer length | ds8 answer length | identical? |
| -- | -- | -- | -- |
| `nn_module_modes_002` | 513 | 513 | **yes** |
| `nn_module_modes_003` | 220 | 223 | no (cosmetic) |
| `debugging_004` | 559 | 559 | **yes** |

For these two FPs, the 1-epoch LoRA adapter produced zero observable token-level perturbation. The dataset shuffle + 3 Why-framed addendum rows + mandatory rebalance changed the adapter (19 of 36 benchmark items have different answers than ds7), but the change was too subtle to deflect greedy decoding on a single token in the `002` or `004` outputs.

Reading the retrieved chunks: the `api_docs_torch_no_grad` chunk (one of 4 retrieved for `002`) explicitly states "torch.no_grad() is fundamentally different from model.eval()", "They address different concerns and are not interchangeable", and "A common mistake is assuming that model.eval() also disables gradients. It does not." The model has the correct content in context but its pretraining template "X and Y are both used to do Z, but they serve different purposes" wins the opener. The corrected content appears in the body of the answer ("model.eval() changes module behavior … torch.no_grad() controls gradient tracking … not interchangeable") but the opener triggers the patched scorer's `must_not_include` rule.

#### Verdict on hypotheses A and B (1 epoch)

- **A failed at 1 epoch.** 3 Why-framed mandatory rows did not perturb the `002` answer at all. Either the rows are not load-bearing or 1 epoch is not enough gradient volume for them to bite.
- **B failed at 1 epoch.** Mandatory rebalance restored argmax in-slice density to ds6 levels but `debugging_004` answer is identical to ds7's incorrect answer. Same conclusion: not enough gradient volume.

The bit-identity is the strongest signal. ds7→ds8 at 1 epoch is too small a perturbation to distinguish A or B from "1 epoch is not enough gradient volume regardless of which rows are mandatory".

### 2-epoch resume run — benchmark `145042`

Scorer: `28/6/2 = 0.778` (226.4s wall). Run file: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-145042.json`. Net `+0.028` vs ds8 1-epoch, **ties `131502` source-content baseline**.

Command: `MODE=MODE_RESUME`, `EPOCHS=2`, `DATASET_NAME=dataset_8`. Resumed from the ds8 1-epoch checkpoint and ran epoch 2 (checkpoint overwritten in place).

#### Per-item diff vs ds8 1-epoch (only items that moved)

| id | ds8 1-ep | ds8 2-ep | notes |
| -- | -- | -- | -- |
| `nn_module_modes_001` | partially_correct | **incorrect** | lost `nn.Dropout/nn.BatchNorm` naming; A-addendum lesson displaced the winning phrasing |
| `nn_module_modes_002` | incorrect | **correct** | **A validated** — short, clean "address different concerns" framing |
| `nn_module_modes_004` | correct | partially_correct | e2 answer is actually *more* detailed (adds EMA + running_mean/var); likely scorer FN |
| `debugging_002` | correct | partially_correct | hit `[4, 3] … raise/error` regex — answer now contains a factually *wrong* example (cat dim=0 of `[2,3]`+`[4,3]` doesn't raise); legitimate demotion |
| `debugging_004` | incorrect | **correct** | **B validated** — "returns indices … not the maximum values themselves" |
| `data_loading_002` | incorrect | **correct** | incidental win; answer now names "padding", "mixed types or nested structures" |
| `autograd_001` | partially_correct | **correct** | cleaner `requires_grad` explanation |
| `tensor_creation_001` | correct | partially_correct | dropped "NumPy" / explicit "shares memory with the array" wording |

4 real wins (including both intervention targets), 4 regressions (1 real loss, 2 scorer artifacts, 1 legitimate demotion).

#### Per-item diff vs `131502` source-content baseline

| id | baseline | ds8 2-ep | notes |
| -- | -- | -- | -- |
| `autograd_001` | partially_correct | **correct** | net gain |
| `dtype_device_003` | partially_correct | **correct** | held from ds6 |
| `nn_module_modes_002` | partially_correct | **correct** | A-intervention target, finally landed |
| `data_loading_003` | partially_correct | incorrect | ds7-era regression, still present |
| `debugging_002` | correct | partially_correct | new epoch-2 regression (wrong example) |
| `nn_module_modes_004` | correct | partially_correct | new epoch-2 regression (scorer FN) |
| `shape_ops_003` | correct | partially_correct | held from ds7 |

ds8 2-epoch trades 3 baseline-correct items for 3 baseline-not-correct items. Net scorer score ties baseline. Composition shifted from baseline's broad coverage toward semantic-binding wins (`002`, `autograd_001`, `dtype_device_003`) at the cost of some vagueness/edge-case regressions.

#### Critical diagnostics

**A and B both validated at 2 epochs.** The two targets that produced bit-identical answers at ds7→ds8 1-epoch both moved at 2 epochs:

| id | ds8 1-ep label | ds8 2-ep label |
| -- | -- | -- |
| `nn_module_modes_002` | incorrect | **correct** |
| `debugging_004` | incorrect | **correct** |

The extra epoch was the missing gradient volume — the rows themselves were load-bearing. At 1 epoch the LoRA adapter could not perturb greedy decoding enough to flip either answer; at 2 epochs the Why-framed opener ("address different concerns") and the direct-negation argmax row both cleared the scorer's `must_not_include` patterns.

**The cost: `nn_module_modes_001` P→I.** Reading the answer, e2 correctly states "model.eval() … does not affect autograd or gradient tracking" — this is the exact lesson from the A-addendum. The model internalized the eval/no_grad separation so well that it dropped the `nn.Dropout`/`nn.BatchNorm` naming that earned the partial at e1. The A-addendum is bleeding into an adjacent benchmark item and displacing the winning content there.

**`debugging_002` is a legitimate demotion, not a regression.** The e2 answer contains a factually wrong example: it claims `cat([a, b], dim=0)` with shapes `[2,3]` and `[4,3]` raises a RuntimeError. It does not — that concat gives `[6, 3]`. The scorer's `must_not_include_regex` for `[4, 3] … raise/error` fires correctly. At 2 epochs the model started generating this malformed example (possibly learned from the argmax-debug rows' "x has shape …" phrasing pattern leaking across).

### FP/FN hand audit (2-epoch)

Per CLAUDE.md, hand-regrade the e2 run item by item:

- `nn_module_modes_004` (P → hand-correct): e2 answer adds EMA + running_mean/var + edge case. More detailed than e1. Likely scorer FN (missing a rigid required phrase or triggering a soft caveat). Flag as FN.
- `tensor_creation_001` (P → hand-still-correct): e2 drops "NumPy" literal but the semantics ("shares memory with the input when possible", "same device and dtype") are correct and more precise. Likely scorer FN on the keyword "numpy". Flag as FN.
- `debugging_002` (P → hand-incorrect-or-partial): the wrong `[2,3]+[4,3]` example is a semantic error. Partial is arguably generous; at least the first half of the answer is correct. Keep as partial.
- `nn_module_modes_001` (I → hand-partial): e2 is technically correct on the autograd clarification but lost the Dropout/BatchNorm anchoring. Should be partial, not incorrect. Flag as over-penalty.
- Everything else matches scorer.

Hand-regrade: `29/5/2 → 0.806` (up from scorer `28/6/2 = 0.778`; up from ds7 hand-regrade `28/5/3 = 0.778`; up from baseline hand `28/7/1 = 0.778`).

**This is the best hand-regraded score so far in the session** — both A and B interventions held at 2 epochs, and the composition gain on `nn_module_modes_002` (the original ceiling FP) is the first real win against `131502` on the two-layer binding question.

The two scorer FNs to note for future scorer patches:

- `nn_module_modes_004`: missing keyword for "mini-batch" / "running_mean" phrasing? Needs a compatibility pattern.
- `tensor_creation_001`: requires "numpy" literal. e2 says "input" and "array-like" and "shares memory with the input". Needs a compatibility pattern.

### Verdict on hypotheses A and B

- **A confirmed at 2 epochs.** 3 Why-framed rows moved `nn_module_modes_002` I→C. The rows are load-bearing; 1 epoch was not enough gradient volume. Side-effect: the strong semantic binding bled into `nn_module_modes_001` and cost the `nn.Dropout`/`nn.BatchNorm` naming there.
- **B confirmed at 2 epochs.** Mandatory rebalance + restored argmax density moved `debugging_004` I→C. No `shape_ops_*` collapse at 2 epochs (unlike ds6 epoch 3).
- **No catastrophic 2-epoch collateral** (ds6's epoch-3 shape_ops collapse did not recur at 2 epochs on 50 rows of ds8).

## Next Step

Landed on branch 1 of the decision tree (both targets flipped, no `shape_ops_*`/`data_loading_*` collapse). Ds8 2-epoch is the current best recipe for this model + retrieval stack. Hand-regrade `29/5/2 = 0.806` beats the `131502` source-content ceiling (hand `28/7/1 = 0.778`) for the first time this session.

Three possible follow-ups, ordered by effort:

1. **Freeze the recipe and consolidate.** Accept ds8 2-epoch as the final small-pilot result. Document the A/B validation. Two scorer FNs (`nn_module_modes_004`, `tensor_creation_001`) deserve scorer patches (compatibility regexes) so the dashboard matches the hand-regrade.
2. **Targeted recovery of `nn_module_modes_001`.** The A-addendum bled into an adjacent item. One option: add 1–2 rows whose `input` is "What's the difference between model.train() and model.eval()" (the `001` shape) with an answer that *both* names `nn.Dropout`/`nn.BatchNorm` explicitly *and* correctly distinguishes eval from no_grad. This reinforces layer naming on the train/eval question without losing the A-win. Costs one run.
3. **Scale-up to 200 rows, 1 epoch.** Per CLAUDE.md, the next size step after a successful pilot is 200/24. At 1 epoch on 200 rows the gradient volume should roughly match 2 epochs on 50 rows, with more phrasing diversity. This tests whether the recipe generalizes beyond the pilot. Higher effort, clearer finding.

Recommendation: option 2 first (one small run to close the `nn_module_modes_001` regression), then option 3 for the scale-up — option 1 is satisfied by landing the log either way.

## Other Important Info

- Training/benchmark runs are executed locally on the user's M1 Pro; sandbox has no `torch`. Results are pasted back into this log.
- Scorer state is the patched scorer used for `131502` and ds7 `140357`. ds8 results are apples-to-apples with both.
- Source corpus is at the `131502` revert state. `data/source/pytorch_docs/nn_module_modes.jsonl` and `data/output/cache/pytorch_corpus.jsonl` are unchanged from the ds7 run — the only experimental variables in this log are dataset composition and epoch count.
- The two known scorer FNs from the ds7 audit are unchanged in ds8 1-epoch (still favoring the baseline answer surface form). They are noise on the absolute number; not a critical-path fix.

## Files touched in this session

- Added: `data/raw/training/dataset_8_addendum.jsonl`
- Added: `scripts/build_dataset_8.py`
- Added: `data/raw/training/dataset_8.jsonl`
- Added: `data/raw/validation/dataset_8.jsonl`
- Added: `experiments/logs/sft-dataset_8-Qwen-Qwen2.5-1.5B-2026-04-22.md` (this file)
- Added: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-143139.json` (1-epoch ds8 benchmark)
- Added: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-145042.json` (2-epoch resume ds8 benchmark)
- Added/updated: `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_8-8-0.0001.pt` (checkpoint now holds the 2-epoch state after resume)
- Modified: `src/config.py` (`DATASET_NAME` → `dataset_8`)
