## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_9` (= `dataset_8` + 2 ds9 addendum rows targeting the `nn_module_modes_001` Q shape; `nn_module_modes` target 7 → 9; `shape_ops` target 11 → 9 to hold total at 50)
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT`
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1` (1-epoch run), `2` (epoch-2 resume run)
- Train rows: `50`; Val rows: `12`
- LoRA Config (unchanged): `r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`
- Device: `mps` (Apple Silicon M1 Pro, 16 GB)
- Post-SFT checkpoints:
  - 1-epoch: *pending*
  - 2-epoch resume: *pending*
- Benchmark results:
  - 1-epoch hybrid: *pending*
  - 2-epoch resume hybrid: *pending*
- Reference baselines (under patched scorer, rescored):
  - `131502` baseline hybrid_with_base_model: `28/5/3 = 0.778` (was `28/7/1 = 0.778` before rescore; correct count unchanged)
  - `143139` ds8 1-ep hybrid: `27/5/4 = 0.750` (unchanged by rescore)
  - `145042` ds8 2-ep hybrid: `30/4/2 = 0.833` (was `28/6/2 = 0.778` before rescore; `tensor_creation_001` P→C and `nn_module_modes_004` P→C under patched rules)
- Timestamp: `2026-04-22`

## Goal

One targeted intervention on top of `dataset_8`, plus a scorer-patch pass that lands independently.

**001-recovery.** At ds8 epoch 2 the A-addendum moved `nn_module_modes_002` I→C but bled into `nn_module_modes_001` (P→I), because the model over-applied the eval-vs-no_grad distinction to the train-vs-eval Q shape and dropped the winning `nn.Dropout` / `nn.BatchNorm` layer naming. `dataset_9` adds 2 mandatory rows whose `input` matches the `001` Q shape ("What is the difference between model.train() and model.eval()?", "How does model.train() differ from model.eval() in practice?") and whose `output` explicitly names `nn.Dropout` and `nn.BatchNorm` AND includes a short, non-opener eval/no_grad clarifier. Purpose:
  1. Restore `001` from I to at least P — ideally C.
  2. Preserve `002` C (the short, non-opener no_grad clarifier re-teaches the A lesson without letting it dominate).
  3. Do not regress `debugging_004` — shape_ops mandatory rows are unchanged; argmax density in shape_ops is 6/9 ≈ 67% (higher than ds8's 5/11 ≈ 45%).

**Scorer patches (independent).** `tensor_creation_001` and `nn_module_modes_004` were scorer FNs on ds8 ep2 — the answers were correct but the rules were too rigid. Two rule changes landed in `data/eval/benchmark_core_pytorch.jsonl`:
  - `tensor_creation_001`: `reuse` moved out of hard `must_include` and into a `must_include_any_of` group with `shares memory`, `share memory`, `sharing memory`, `memory sharing` — accepts the NL variant.
  - `nn_module_modes_004`: `expected_symbols` cleared (was `[BatchNorm, Module.train, Module.eval]`, too strict); replaced with `must_include_any_of: [["batch normalization", "batchnorm", "nn.batchnorm"]]`. `must_include ["running statistics"]` kept.

Rescored ds7 1-ep (no change), ds8 1-ep (no change), ds8 2-ep (+2 correct), baseline (0 correct delta, P→I composition shift from pre-existing patches).

## Findings

### Data

- `data/raw/training/dataset_9_addendum.jsonl` — 2 new rows, both with the `nn_module_modes_001` Q shape and layer-named `output`.
- `scripts/build_dataset_9.py` — forked from `build_dataset_8.py`. Adds `SRC_ADDENDUM_V5 = dataset_9_addendum.jsonl`. `MANDATORY_INPUT_SUBSTRINGS` grows 12 → 14 (2 new ds9 rows added). `TRAIN_TARGETS` nn_module_modes 7 → 9 (to seat 9 mandatory rows: 3 dropout direct-neg + 3 Why-framed + 1 which-layers + 2 ds9 recovery), shape_ops 11 → 9 (to hold total at 50; argmax mandatory rows unchanged so density goes up).
- `data/raw/training/dataset_9.jsonl` (50 rows): 9 nn_module_modes (all mandatory) + 9 shape_ops (6 mandatory argmax + 3 random) + 7 autograd + 8 debugging + 6 tensor_creation + 6 hallucination_refusal + 4 optim_training_loop + 1 dtype_device + 0 data_loading.
- `data/raw/validation/dataset_9.jsonl` (12 rows): same val pool as ds6/7/8.

### Scorer patches

- `data/eval/benchmark_core_pytorch.jsonl` — two items edited:
  - `tensor_creation_001`: `must_include: [copy, reuse]` → `must_include: [copy]` + `must_include_any_of: [[reuse, shares memory, share memory, sharing memory, memory sharing]]`.
  - `nn_module_modes_004`: `expected_symbols: [BatchNorm, Module.train, Module.eval]` → `expected_symbols: []` + `must_include_any_of: [[batch normalization, batchnorm, nn.batchnorm]]`.

Rescore pass (`scripts/rescore_benchmark.py`) produced:

| Run | Before rescore | After rescore | Δ correct |
| -- | -- | -- | -- |
| `131502` baseline | 28/7/1 = 0.778 | 28/5/3 = 0.778 | 0 |
| `140357` ds7 1-ep | 26/6/4 = 0.722 | 26/6/4 = 0.722 | 0 |
| `143139` ds8 1-ep | 27/5/4 = 0.750 | 27/5/4 = 0.750 | 0 |
| `145042` ds8 2-ep | 28/6/2 = 0.778 | **30/4/2 = 0.833** | +2 |

Baseline's 2 P→I moves (`nn_module_modes_002`, `data_loading_003`) were not caused by the two new ds9-related patches — they come from the earlier scorer-patch passes (tasks #19, #23) that were never re-applied to `131502`. Correct count unchanged, so `overall_accuracy` (correct/total) is stable at 0.778.

ds8 2-ep's 2 P→C promotions (`tensor_creation_001`, `nn_module_modes_004`) are the intended effect — both answers were hand-regrade correct and the dashboard now agrees.

### 1-epoch run (`154414`) — 27/6/3 = 0.750

Command: `MODE=MODE_TRAIN`, `EPOCHS=1`, `DATASET_NAME=dataset_9`. Checkpoint: `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_9-8-0.0001.pt`. Benchmark summary: `total=36 correct=27 partially_correct=6 incorrect=3 time=233.44s`.

Diff vs ds8 1-ep rescored (27/5/4 = 0.750):

| Item | ds8 1-ep | ds9 1-ep | Δ |
| -- | -- | -- | -- |
| `data_loading_002` | I | C | +1 C |
| `nn_module_modes_001` | P | P | same (held at P — early recovery signal since ds8 2-ep dropped it to I) |
| `nn_module_modes_002` | I | I | same (expected; 2-ep only per ds8 pattern) |
| `nn_module_modes_004` | C | C | same (under patched scorer) |
| `debugging_002` | C | P | −1 (mild phrasing drift) |
| `debugging_004` | I | I | same |

Net: +1 / −1 same total, but the shape of failures is different. The read: ds9 1-ep is already **holding `001` at P** (ds8 1-ep held at P too, but ds8 ep2 blew it to I — ds9's mandatory rows are now carrying the 001 Q shape at 1 epoch).

### 2-epoch resume run (`160319`) — 27/7/2 = 0.750

Command: `MODE=MODE_RESUME`, `EPOCHS=2`, `DATASET_NAME=dataset_9`. Benchmark summary: `total=36 correct=27 partially_correct=7 incorrect=2 time=224.71s`.

Diff vs ds8 2-ep rescored (30/4/2 = 0.833):

| Item | ds8 2-ep | ds9 2-ep | Δ | Note |
| -- | -- | -- | -- | -- |
| `tensor_creation_002` | C | P | C→P | mild phrasing drift; answer is factually fine |
| `nn_module_modes_001` | I | P | **I→P** | **goal met** — 001 recovered |
| `nn_module_modes_002` | C | C | same | held; "address different concerns" opener preserved |
| `nn_module_modes_003` | C | P | **C→P** | **contamination** — Dropout got BN semantics |
| `nn_module_modes_004` | C | C | same | held under patched scorer |
| `debugging_004` | C | I | **C→I** | argmax direct-negation signal weakened |
| `debugging_002` | P | P | same | still phrasing-drifted, not the prior C |

Diff vs ds9 1-ep (27/6/3 = 0.750):

| Item | ds9 1-ep | ds9 2-ep | Δ |
| -- | -- | -- | -- |
| `tensor_creation_002` | C | P | −1 C |
| `autograd_001` | P | C | +1 C |
| `nn_module_modes_002` | I | C | +1 C (A lesson lands at ep2 as expected) |
| `nn_module_modes_003` | C | P | −1 C (contamination) |

Net ds9 2-ep: 27 correct (same as 1-ep), but *composition* rotated — 002 promoted I→C, 003 demoted C→P, 004 held C (under patched scorer).

### Critical diagnostic — nn_module_modes_003 contamination

Actual ds9 2-ep answer to "What changes for dropout layers after calling model.eval()?":

> After calling model.eval(), nn.Dropout layers stop computing statistics from the current batch and instead **normalize using the running mean and running variance** that were accumulated during training. ...

The model applied **BN semantics to Dropout** — "normalize using running mean and running variance" is a BN-only concept. Dropout does not use running statistics; in eval mode it is simply a no-op.

Root cause: the two ds9 addendum rows mention `nn.Dropout` and `nn.BatchNorm` in the same sentence context, with "running_mean / running_var" bound to the `model.eval()` branch. At 2 epochs the association strengthens to the point where the model treats `model.eval() → running statistics` as a general rule that applies to any stateful layer, including Dropout.

At 1 epoch this didn't trigger (003 was still C). The contamination is specifically an epoch-2 over-application effect, and the ds9 2-ep signature mirrors the ds8 2-ep pattern (A lesson bled into 001) — just one category over. This is the **same failure mode, different target**: 001 recovery rows are now the bleed source.

### Critical diagnostic — debugging_004 regression

ds9 2-ep answer reverted to the pretraining template:

> torch.argmax(x, dim=1) returns the index of the overall maximum in **the flattened tensor as a scalar tensor**, not the maximum values themselves. ...

This is the exact wrong shape claim that ds8 2-ep had corrected. The shape_ops 11→9 trim removed 2 non-argmax random rows (argmax mandatory unchanged; density 5/11 → 6/9 = 67%), so density is up but *absolute argmax gradient volume is flat*. Meanwhile the 2 new ds9 nn_module_modes addendum rows added ~700 new tokens of `model.eval() / model.train()` signal to the batch. Net effect: argmax direct-negation gradient was crowded out by the recovery rows in the shuffled training sequence.

### FP/FN hand audit (ds9 2-ep)

| Item | Scorer | Audit | Cause |
| -- | -- | -- | -- |
| `nn_module_modes_001` | P | P | factually recovered, but answer says "updates running mean and variance buffers in evaluation mode" which is *wrong* (BN updates in train, not eval). Still an upgrade over ds8 2-ep I. |
| `nn_module_modes_003` | P | **I** | "normalize using running mean and running variance" applied to Dropout is a factual error. Audit should demote this to I, not P. |
| `tensor_creation_002` | P | C | "torch.tensor always makes a full copy" and "torch.from_numpy ... sharing memory ... avoiding a copy" are both correct. Scorer phrasing strictness, not a factual problem. |
| `debugging_002` | P | P | confirmed: the wrong `[2,3]+[4,3]` example didn't surface, but broadcast-rule framing is slightly off. |
| `debugging_004` | I | I | confirmed: "flattened tensor as a scalar tensor" is the direct-negation target. |

Hand-regrade: `27/6/3 = 0.750` (nn_module_modes_003 P→I, tensor_creation_002 P→C, one swap each direction).

Under *hand-regrade* ds8 2-ep was 29/5/2 = 0.806; ds9 2-ep is 27/6/3 = 0.750. **ds8 2-ep beats ds9 2-ep by 2 correct under both the scorer and the hand regrade.**

### Verdict

- **001 recovery: partial win.** I→P (scorer and audit agree). Goal met at the P level, not C. The answer has a subtle factual slip ("updates running mean and variance buffers in evaluation mode") but matches the winning shape.
- **002 hold: yes.** C preserved.
- **004 hold: yes.** C preserved under patched scorer.
- **003 contamination: loss.** C→P (audit-demote to I). This is new collateral from the recovery rows.
- **debugging_004 regression: loss.** C→I. argmax signal crowded out.

At 50 rows / r=16 / 2 epochs the model is in a zero-sum capacity regime: a 2-row swap to recover 001 has cost 2 correct elsewhere (003 + debugging_004). The ds8 2-ep recipe (0.833) remains the pilot best.

## Next Step

Decision after ds9 results: the bottleneck is **capacity**, not phrasing. ds9's 001-recovery rows work, but at 50 rows they crowd out Dropout semantics (003) and argmax direct-negation (debugging_004). The clean next step per CLAUDE.md is the pilot → scale-up transition:

1. **Accept ds8 2-ep as pilot terminus. Scale up to 200 rows / 1 epoch** (CLAUDE.md defaults for scaled runs). 4x the data at half the epochs keeps total gradient volume roughly constant but eliminates the 2-epoch over-application that caused both ds8's 001 regression and ds9's 003 contamination.
2. **Revise addendum for ds10**: keep `001` Q-shape rows (named `nn.Dropout` + `nn.BatchNorm`) but strip the specific "running_mean / running_var" BN details — those are the exact phrases that bled into 003. Reference BN behavior at high level only.
3. ds10 is the new recipe. If it does not beat 0.833 at 1 epoch, the next lever is a targeted 2-epoch resume on the 200-row set (not more data).

## Other Important Info

- Training/benchmark runs are executed locally on the user's M1 Pro; sandbox has no `torch`. Results are pasted back into this log.
- Scorer state is patched: `tensor_creation_001` and `nn_module_modes_004` now accept the NL phrasing variants. All ds7/ds8/baseline numbers are rescored for apples-to-apples comparison.
- Source corpus is at the `131502` revert state (unchanged from ds7/ds8). The only experimental variables in ds9 vs ds8 are the 2 new mandatory rows + the two target-count adjustments.
- 50-row ds9 at 2 epochs matches the ds8 2-ep recipe that validated A and B. No LR / batch / LoRA change.

## Files touched in this session

- Added: `data/raw/training/dataset_9_addendum.jsonl` (2 rows)
- Added: `scripts/build_dataset_9.py`
- Added: `data/raw/training/dataset_9.jsonl` (50 rows)
- Added: `data/raw/validation/dataset_9.jsonl` (12 rows)
- Added: `experiments/logs/sft-dataset_9-Qwen-Qwen2.5-1.5B-2026-04-22.md` (this file)
- Modified: `data/eval/benchmark_core_pytorch.jsonl` (`tensor_creation_001` and `nn_module_modes_004` rules relaxed for NL variants)
- Added: `experiments/eval_results/benchmark/rescored/*.json` (4 rescored prior-run files)
- Modified: `src/config.py` (`DATASET_NAME` → `dataset_9`, `EPOCHS=1`, `MODE=MODE_TRAIN`)
