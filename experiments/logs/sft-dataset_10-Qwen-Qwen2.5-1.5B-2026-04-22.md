## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_10` (200 train / 24 val — CLAUDE.md scaled-run defaults)
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT`
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1` (1-epoch run), optional `2` (resume if 1-ep underperforms)
- Train rows: `200`; Val rows: `24`
- LoRA Config (unchanged): `r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`
- Device: `mps` (Apple Silicon M1 Pro, 16 GB)
- Post-SFT checkpoints:
  - 1-epoch: *pending*
  - 2-epoch resume (conditional): *pending*
- Benchmark results:
  - 1-epoch hybrid: *pending*
  - 2-epoch resume hybrid (conditional): *pending*
- Reference baselines (patched scorer, rescored):
  - `131502` baseline hybrid_with_base_model: `28/5/3 = 0.778`
  - `143139` ds8 1-ep hybrid: `27/5/4 = 0.750`
  - `145042` ds8 2-ep hybrid: `30/4/2 = 0.833` ← current recipe ceiling
  - `154414` ds9 1-ep hybrid: `27/6/3 = 0.750`
  - `160319` ds9 2-ep hybrid: `27/7/2 = 0.750`
- Timestamp: `2026-04-22`

## Goal

Transition from the 50-row pilot into the 200-row scaled recipe per
`CLAUDE.md`. The ds9 pilot proved two things:

1. **A targeted 001-recovery row works.** ds9 added 2 rows matching the
   `nn_module_modes_001` Q shape with `nn.Dropout` / `nn.BatchNorm` named
   explicitly, and `001` moved I → P at 2 epochs.
2. **50 rows is a zero-sum capacity budget.** Every row added for a
   recovery target took signal away from somewhere else. ds9 ep2 gained
   `001` P but lost `003` C (Dropout<-BN contamination from the
   `running_mean / running_var` tokens in the ds9 addendum) and
   `debugging_004` C (argmax direct-negation gradient crowded out by the
   2 new rows).

ds10 attacks both of those at once:

- **More data.** 200 rows gives every capability its own room. All 10
  autograd rows, all 24 debugging rows, all 21 nn_module_modes rows, all
  19 hallucination_refusal rows, all 6 tensor_creation rows, all 4
  optim rows, and the full dtype_device row. Shape_ops is filled to 115
  from a pool of 133 — this keeps argmax density unchanged vs ds9 while
  adding the remaining broadcast / reshape / softmax / transpose rows.
- **1 epoch, not 2.** 50 rows × 2 epochs = 100 samples; 200 rows × 1
  epoch = 200 samples. Roughly 2x the total gradient volume, spread
  across 4x more distinct inputs. This removes the 2-epoch
  over-application effect responsible for both the ds8 2-ep 001
  regression and the ds9 2-ep 003 contamination.
- **Contamination-safe addendum.** `dataset_10_addendum.jsonl` carries
  the same two 001-recovery inputs but drops `running_mean` and
  `running_var` from the output. Both rows still name `nn.Dropout` and
  `nn.BatchNorm`; BN behavior is summarized as "uses the current
  mini-batch in train mode and the statistics it built up during
  training in eval mode" — no `running_*` token binding to the
  `model.eval()` branch. The ds9 v5 addendum is intentionally omitted
  from the pool.

Primary hypothesis: ds10 1-ep beats ds8 2-ep (0.833) on the
scorer-patched hybrid benchmark.

Fall-back decision rules are in `Next Step`.

## Findings

### Data

- Added: `data/raw/training/dataset_10_addendum.jsonl` (2 rows). Same
  input substrings as ds9 addendum; outputs name `nn.Dropout` and
  `nn.BatchNorm` without `running_mean` / `running_var`.
- Added: `scripts/build_dataset_10.py`. Forked from `build_dataset_9.py`
  with these changes:
  - `SRC_ADDENDUM_V5` (ds9 addendum) deliberately omitted from the pool
    merge.
  - `SRC_ADDENDUM_V6` = ds10 addendum added.
  - `TRAIN_TARGETS` sum = 200; every non-shape_ops category takes its
    full pool; shape_ops capped at 115.
  - `VAL_TOTAL` = 24 (entire pool), `VAL_CAPS` per category raised so
    the whole pool is retained.
  - Hard assertion: v6 addendum rows must not contain `running_mean`,
    `running_var`, `running statistics`, `running buffers`, or
    `running mean`/`running variance`. Build raises if violated.
- Added: `data/raw/training/dataset_10.jsonl` (200 rows). Category
  distribution: shape_ops=115, debugging=24, nn_module_modes=21,
  hallucination_refusal=19, autograd=10, tensor_creation=6,
  optim_training_loop=4, dtype_device=1, data_loading=0. Mandatory
  pre-fills: 15 (9 nn_module_modes + 6 shape_ops).
- Added: `data/raw/validation/dataset_10.jsonl` (24 rows = full val
  pool).
- Verified: 0 ds9 addendum rows leaked into ds10; 0 occurrences of the
  exact v5-signature phrase `stored running_mean / running_var` in the
  selected rows. Existing ds6 direct-negation rows *do* mention
  `running statistics` (they are the rows teaching that Dropout does
  NOT use running statistics) — that is intentional and not a
  contamination source. ds8 2-ep held 003 C with those same rows
  present; the marginal contamination at ds9 2-ep came from the new
  positive-attribution rows, which ds10 removes.

### 1-epoch run (`165020`) — 28/4/4 = 0.778

Command: `MODE=MODE_TRAIN`, `EPOCHS=1`, `DATASET_NAME=dataset_10`.
Checkpoint: `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_10-8-0.0001.pt`.
Benchmark summary: `total=36 correct=28 partially_correct=4 incorrect=4 time=165.13s`.

Diff vs ds8 2-ep rescored (30/4/2 = 0.833):

| Item | ds8 2-ep | ds10 1-ep | Δ | Note |
| -- | -- | -- | -- | -- |
| `debugging_002` | P | **C** | +1 C | chronic P at ds7/ds8/ds9 finally flipped — `cat` vs `stack` distinction landed |
| `nn_module_modes_002` | C | P | −1 C | A lesson not fully landed at 1 ep (same pattern as ds8 1-ep → ds8 2-ep) |
| `optim_training_loop_004` | P | **I** | −1 C equivalent (was P, now I) | 4 rows / 200 = 2% density vs 4/50 = 8% on ds8; gradient accumulation answer degraded |
| `shape_ops_004` | C | **I** | −1 C | `flatten(start_dim=2)` on `[3,2,4,5]` → said `[3, 8, 5]` (wrong — should be `[3, 2, 20]`) |
| `data_loading_004` | C | P | −1 C | minor phrasing drift |

Net vs ds8 2-ep: +1 C, −3 C, and 2 new I — that is 28/4/4 vs 30/4/2.

Diff vs ds9 2-ep (27/7/2 = 0.750):

| Item | ds9 2-ep | ds10 1-ep | Δ |
| -- | -- | -- | -- |
| `tensor_creation_002` | P | C | +1 |
| `debugging_002` | P | C | +1 |
| `debugging_004` | I | **C** | **+1 (major)** — argmax direct-negation held after the ds9 crowding loss |
| `nn_module_modes_001` | P | I | −1 |
| `nn_module_modes_002` | C | P | −1 |
| `nn_module_modes_003` | P | **C** | **+1 (goal)** — no Dropout<-BN contamination |
| `optim_training_loop_004` | P | I | −1 |
| `shape_ops_004` | C | I | −1 |

Net vs ds9 2-ep: +4 C, −4 C; summary count barely moves (27 → 28) but the composition is different — ds10 traded 2 nn_module_modes wins (001 P, 002 C) for 2 different wins (003 clean C, debugging_004 C) plus one P→C on debugging_002.

### Critical diagnostic — 3 hypothesized wins confirmed

1. **`nn_module_modes_003` held C with no BN contamination.** The v6
   addendum's phrasing fix worked. The answer:
   > After calling model.eval(), dropout layers are disabled and do not
   > zero elements during inference. nn.Dropout is turned off, and
   > nn.BatchNorm stops computing running statistics.
   There is no "normalize using running mean" language leaking into
   Dropout. A mild factual imprecision ("BN stops computing running
   statistics" — in eval it uses stored ones, doesn't update them) but
   directionally correct and scorer C. **Goal met.**
2. **`debugging_004` recovered to C.** Answer leads with
   "torch.argmax(x, dim=1) returns indices, not values" — no "flattened
   tensor as a scalar" pretraining template. Argmax direct-negation
   signal held at 200-row scale.
3. **`debugging_002` broke the chronic-P ceiling.** ds7/ds8/ds9 all had
   this at P. At 200 rows with all 24 debugging rows present, the
   `cat` vs `stack` distinction landed fully.
4. **`nn_module_modes_004` held C** under patched scorer.

### Critical diagnostic — density-driven regressions

| Item | Density | Outcome |
| -- | -- | -- |
| `nn_module_modes_001` | 2 recovery rows / 200 = 1% | I. At ds9 1-ep density was 2/50 = 4%, which held P. Below the threshold needed for this Q shape. |
| `nn_module_modes_002` | 3 Why-framed rows / 200 = 1.5% | P. Pattern matches ds8 1-ep → 2-ep: this A lesson needs 2 epochs to fully land at any density. |
| `optim_training_loop_004` | 4 rows / 200 = 2% | I (down from ds8 2-ep P). All 4 optim rows are present, but density is 4x lower than ds8's 4/50. The 1-epoch pass isn't enough to compensate. |
| `shape_ops_004` | flatten rows diluted in 115 shape_ops | I. ds8 2-ep got this C at 11 shape_ops rows × 2 epochs. At 115 × 1 epoch the specific `flatten(start_dim=k)` pattern didn't land. |

The read: **the 3 hypothesized wins (003, 004, 002-debugging) are genuine
recipe wins from scale + clean phrasing. The regressions are a pure
density-per-epoch effect — low-frequency-specific rows need more passes.**

### FP/FN hand audit (ds10 1-ep)

| Item | Scorer | Audit | Cause |
| -- | -- | -- | -- |
| `nn_module_modes_001` | I | I | confirmed — answer drops `nn.Dropout` / `nn.BatchNorm` names, relapses to the "does not affect autograd" opener. |
| `nn_module_modes_002` | P | P | confirmed — correct content, weaker opener. |
| `nn_module_modes_003` | C | C | answer has mild "BN stops computing running statistics" imprecision, but directionally right. No Dropout<-BN contamination. Keep C. |
| `optim_training_loop_004` | I | I | confirmed — claims accumulation "reduces number of forward passes" (wrong, it increases them). |
| `shape_ops_004` | I | I | confirmed — `[3, 8, 5]` is the wrong shape. Correct is `[3, 2, 20]`. |
| `data_loading_004` | P | C | could be upgraded — the answer is substantively correct. Keep P to match scorer; not worth a rule change. |
| `debugging_002` | C | C | confirmed good answer. |
| `debugging_004` | C | C | confirmed good answer. |

Hand-regrade: `28/4/4 = 0.778` (no changes from scorer). Under
hand-regrade ds8 2-ep is 29/5/2 = 0.806.

### Verdict

- **ds10 1-ep scorer: 0.778.** Below ds8 2-ep ceiling (0.833) by 2 C,
  tied with baseline (0.778).
- **Hand-regrade: 0.778.** Below ds8 2-ep hand-regrade (0.806).
- **Recipe wins (real):** 003 clean C (no contamination), 004 argmax C
  (density-safe), debugging_002 C (chronic-P broken).
- **Recipe losses (density at 1 ep):** 001 (recovery row density fell
  from 4% to 1%), 002 (A lesson needs 2 ep), optim_004 (density fell
  4x), shape_ops_004 (specific pattern diluted across 115 rows).

The structural takeaway: **at 200 rows, 1 epoch is undertrained for
low-frequency-specific rows that ds8 captured at 50 rows × 2 epochs.**
The gradient volume argument (200 × 1 = 200 samples vs 50 × 2 = 100)
holds in aggregate but fails per-row for rare patterns.

### 2-epoch resume run (`170716`) — 26/8/2 = 0.722

Command: `MODE=MODE_RESUME`, `EPOCHS=2`, `DATASET_NAME=dataset_10`.
Benchmark summary: `total=36 correct=26 partially_correct=8 incorrect=2 time=184.03s`.

**ds10 2-ep is worse than ds10 1-ep (-2 C, same total).** Under every
comparison (ds8 2-ep = 0.833, ds10 1-ep = 0.778, baseline = 0.778)
ds10 2-ep (0.722) is the weakest result on the board except for ds7.

Diff vs ds10 1-ep (28/4/4):

| Item | ds10 1-ep | ds10 2-ep | Δ | Note |
| -- | -- | -- | -- | -- |
| `nn_module_modes_001` | I | **P** | +1 (goal) | recovery finally landed at 2 ep, layer names back |
| `nn_module_modes_002` | P | **C** | +1 | A lesson landed (expected at 2 ep) |
| `data_loading_003` | I | P | +1 | weak — "pinning memory to the device" is loose but directionally right |
| `data_loading_004` | P | **C** | +1 | phrasing cleaned up |
| `nn_module_modes_003` | C | **P** | **−1** | **contamination re-emerged** — Dropout got BN semantics again |
| `autograd_003` | C | P | −1 | content still correct; scorer phrasing-strict |
| `data_loading_002` | C | P | −1 | answer mostly right, "concatenate along new first dimension" phrasing is weird |
| `debugging_002` | C | **P** | −1 | **new factual error**: "cat creates a new dimension, stack joins along existing" is backwards |

Diff vs ds8 2-ep rescored (30/4/2 = 0.833):

| Item | ds8 2-ep | ds10 2-ep | Δ |
| -- | -- | -- | -- |
| `autograd_003` | C | P | −1 |
| `data_loading_002` | C | P | −1 |
| `data_loading_003` | I | P | +1 (minor) |
| `data_loading_004` | C | C | same |
| `nn_module_modes_001` | I | P | +1 (goal at 2 ep) |
| `nn_module_modes_003` | C | P | −1 (contamination) |
| `optim_training_loop_004` | P | I | −1 |
| `shape_ops_004` | C | I | −1 |
| all others | same | same | |

Net vs ds8 2-ep: +2 soft moves, −5 strong moves. **-4 C, -0 I** vs ds8.

### Critical diagnostic — nn_module_modes_003 contamination is structural at 2-ep

Actual ds10 2-ep answer to "What changes for dropout layers after
calling model.eval()?":

> After calling model.eval(), dropout layers **stop computing
> statistics and normalizing using running mean and variance**. They
> pass inputs through unchanged and do not scale elements.

**This is the same error as ds9 2-ep**, just a different trigger path.
ds9 2-ep was triggered by the v5 addendum's explicit binding of
`running_mean / running_var` to the `model.eval()` branch in the
001-recovery Q shape. ds10 2-ep is triggered by the 6 pre-existing
rows in ds2..v4 that positively bind `running_*` to BN eval mode
(`Why does dropout not use running statistics...`, `Does dropout use
running statistics...`, etc.). All 6 are structured as "dropout
negation + BN positive binding", which at 50-row ds8 was fine (003
held C at 2 ep), but at 200-row ds10 the Q-shape-matching density
drops and **the BN positive-binding clause crosses a threshold where
Dropout inherits it at 2 epochs**.

**Key mechanism**: at 2 epochs on 200 rows = 400 samples, the per-row
gradient contribution grows, and the model starts pattern-matching
across sentences instead of within them. A sentence that says
"Dropout has no running statistics — BN uses running_mean and
running_var in eval" gets re-chunked into "model.eval() ⇒
running_mean / running_var" after two passes.

### Other regressions

- **`debugging_002` C → P with a new factual inversion**: "cat creates
  a new dimension and increases the rank, while stack joins along an
  existing dimension" — this is the exact inverse of the correct API
  semantics (cat concatenates along an existing dim; stack creates a
  new dim). ds10 1-ep had this row's answer correct. The inversion
  is another over-application artifact at 2 epochs.
- **`optim_training_loop_004` still I**: second pass did not help;
  4 rows / 200 = 2% density is structural under-coverage for this
  specific capability regardless of epochs.
- **`shape_ops_004` still I**: same story; flatten-specific pattern
  at 200-row scale is under-sampled.

### FP/FN hand audit (ds10 2-ep)

| Item | Scorer | Audit | Cause |
| -- | -- | -- | -- |
| `nn_module_modes_003` | P | **I** | "stop computing statistics and normalizing using running mean and variance" applied to Dropout is a factual error — demote. |
| `autograd_003` | P | **C** | content correct; scorer phrasing-strict FN. |
| `debugging_002` | P | P or I | "cat creates a new dimension" is factually backwards; borderline between P and I. Keep P (scorer gave credit for the correct lead-in). |
| `nn_module_modes_001` | P | P | confirmed (names layers, one clause weird). |
| `data_loading_002` | P | P | confirmed (right idea, mushy phrasing). |
| `data_loading_003` | P | P | confirmed (partial). |
| `data_loading_004` | C | C | confirmed. |

Hand-regrade: 26/8/2 = 0.722 → swaps cancel (autograd_003 P→C cancels
nn_module_modes_003 P→I), so hand-regrade stays at **26/8/2 = 0.722**.

### Full recipe arc

| Recipe | Score | Correct | Notes |
| -- | -- | -- | -- |
| `131502` baseline | 0.778 | 28 | source-content only |
| ds7 1-ep | 0.722 | 26 | |
| ds8 1-ep | 0.750 | 27 | |
| **ds8 2-ep** | **0.833** | **30** | **← recipe ceiling** |
| ds9 1-ep | 0.750 | 27 | |
| ds9 2-ep | 0.750 | 27 | 003 contamination, 004 regression |
| ds10 1-ep | 0.778 | 28 | 003 clean, 004 recovered |
| ds10 2-ep | 0.722 | 26 | 003 contamination re-emerged |

**ds8 2-ep at 0.833 is the recipe ceiling on this model (Qwen2.5-1.5B),
this benchmark (36 items, patched scorer), and this knowledge corpus
(131502 revert state).** Neither the 200-row scale-up at 1 ep nor at
2 ep beat it.

### Verdict

- **ds10 1-ep is the best scale-up result (0.778)** — it held 003 C,
  recovered debugging_004 C, broke the debugging_002 chronic-P.
- **ds10 2-ep underperforms** because:
  1. The 2-epoch over-application effect reappears at 200-row scale,
     and the 6 `running_*` positive-binding rows (not the v6 addendum)
     are now the contamination source for 003.
  2. Low-density capabilities (`optim_004`, `shape_ops_004`) never
     recovered even with 2 passes — their pool-level density is too
     low.
  3. Paradoxical C→P drift on adjacent items (`debugging_002`,
     `data_loading_002`, `autograd_003`) as the model over-smooths
     answers.
- **The recipe has stopped scaling** on this model at this
  benchmark. Further gains on these 36 items require a different
  lever than train-more-rows.

## Next Step

Pilot/scale-up arc is closed. Remaining levers, in order of expected
yield:

1. **Benchmark + scorer expansion**. 36 items is noisy; several
   recent P labels were hand-regraded C. Adding items + relaxing
   rigid `must_include` rules is cheaper than training gains and
   directly moves the signal-to-noise floor.
2. **Source-content / knowledge patches** (CLAUDE.md RAG rules). The
   `131502` revert corpus is fixed. Targeted patches on the train/eval
   doc and the flatten / gradient-accumulation docs should move
   `nn_module_modes_001`, `shape_ops_004`, `optim_training_loop_004`
   on the hybrid benchmark without retraining.
3. **Larger base model** (Qwen2.5-3B or 7B). At 1.5B the LoRA is
   visibly struggling with cross-sentence pattern over-application.
   A 3B base with the ds8 2-ep recipe is the natural next data point.
4. **If we insist on another 200-row LoRA run**: rewrite the 6
   positive-binding BN rows to avoid `running_mean` / `running_var`
   tokens (same treatment as v6 addendum). This would unblock 003
   at 2 epochs. But expected yield is +1 to +2 correct on a 36-item
   benchmark — probably below the benchmark noise floor.

Recommended immediate next step: **(1) benchmark + scorer expansion**.
Concretely: double the benchmark to ~70 items, add variants for
`argmax` / `flatten` / `cat vs stack` / `eval vs no_grad` / `gradient
accumulation`, and make the scorer's `must_include_any_of` rules the
default rather than the exception. Then re-measure ds8 2-ep vs the
source-content baseline on the expanded benchmark.

### FP/FN hand audit

*Pending benchmark results.* Audit checklist:
  - `nn_module_modes_001`: does the answer still name `nn.Dropout` and
    `nn.BatchNorm` (not just "stateful layers")? Does it avoid the ds9
    ep2 slip "updates running mean and variance buffers in evaluation
    mode" (BN updates in train, not eval)?
  - `nn_module_modes_003`: critical check — does Dropout pick up any
    BN-only semantics (e.g. "normalize using ..." or "statistics")?
    Under ds10's phrasing-safe addendum this should not happen.
  - `nn_module_modes_002`: does the "address different concerns" opener
    still win?
  - `nn_module_modes_004`: does the BN-mode answer still mention
    "running statistics" and "batch normalization" (both scorer
    required under patched rules)?
  - `debugging_004`: does the "flattened tensor as a scalar" wrong
    shape template re-surface? At 200 rows the argmax direct-negation
    rows are 3/200 = 1.5% instead of 3/50 = 6%, so signal density is
    *lower*. This is the main risk factor for ds10; watch it closely.
  - `tensor_creation_002`: minor — ds9 2-ep drifted C→P due to
    phrasing; at 200 rows with all 6 tensor_creation rows present
    this should be C.
  - `data_loading_003`: has been I under every SFT recipe; the pool has
    0 data_loading rows, so it stays untrained. Not a ds10 concern.

## Next Step

After the 1-epoch benchmark lands:

1. **If ds10 1-ep ≥ 0.833 AND no new regressions vs ds8 2-ep**: ds10
   is the new recipe. Move on to next capability gap (probably
   `data_loading` since the pool is empty, or `tensor_creation_002`
   phrasing drift).
2. **If ds10 1-ep < 0.833 but ≥ 0.806 (hand-regrade ds8 2-ep)**: run
   2-epoch resume. Watch 003 carefully — if it holds C, ds10 2-ep is
   the new recipe. If 003 regresses again, the per-token contamination
   at 2 epochs is structural (not just about the v5 phrasing) and the
   next lever is a source-content patch on the train/eval docs.
3. **If ds10 1-ep < 0.806**: the 200-row scale did not transfer the
   ds8 gains. Inspect training loss curve, check whether the argmax
   direct-negation rows are diluted at 3/200 density, consider raising
   their multiplicity (e.g. duplicate the 3 rows to 6 rows without
   adding total count) before scaling further.

## Other Important Info

- Training/benchmark runs are executed locally on the user's M1 Pro;
  sandbox has no `torch`. Results pasted back into this log.
- Scorer state unchanged from ds9: `tensor_creation_001` and
  `nn_module_modes_004` accept the NL phrasing variants (patched
  2026-04-22 earlier in the day). All baselines listed above are
  rescored.
- Source corpus is at the `131502` revert state (unchanged from
  ds7/ds8/ds9). The experimental variables in ds10 vs ds9 are: 200
  rows instead of 50, 1 epoch instead of 2, v6 addendum instead of v5.
- val pool is exactly 24 rows; ds10 val = entire pool.
- `MAX_TRAIN_ROWS` and `MAX_VAL_ROWS` in `src/config.py` are both set
  to `None` so the SFT loader uses the full 200/24 files without
  resampling.

## Files touched in this session

- Added: `data/raw/training/dataset_10_addendum.jsonl` (2 rows,
  contamination-safe phrasing)
- Added: `scripts/build_dataset_10.py`
- Added: `data/raw/training/dataset_10.jsonl` (200 rows)
- Added: `data/raw/validation/dataset_10.jsonl` (24 rows)
- Added: `experiments/logs/sft-dataset_10-Qwen-Qwen2.5-1.5B-2026-04-22.md`
  (this file)
- Modified: `src/config.py`: `DATASET_NAME → dataset_10`,
  `EPOCHS = 1`, `MAX_TRAIN_ROWS = None`, `MAX_VAL_ROWS = None`,
  `MODE = MODE_TRAIN`.
- Finalized: `experiments/logs/sft-dataset_9-Qwen-Qwen2.5-1.5B-2026-04-22.md`
  (ds9 1-ep and 2-ep results + FP/FN audit + verdict that motivated
  ds10)

## Benchmark expansion (scorer + item set)

With the pilot/scale-up arc closed at ds8 2-ep (0.833 ceiling) and ds10
2-ep regressing to 0.722 (pre-soften) under extra LoRA capacity, the
next lever is not more training — it is **more benchmark**.

### What changed

Benchmark file: 36 → 60 items. Backup of the previous file at
`data/eval/benchmark_core_pytorch_v1.jsonl`.

**Softens (4 items, scorer relaxations)** — all four were chronic
non-C under strict `must_include`. In each case the strict phrase was
swapped for a `must_include_any_of` synonym group so the scorer
credits valid phrasing variants.

| id | before `must_include` | after |
|---|---|---|
| `shape_ops_001` | `["contiguous", "copy"]` | `["contiguous"]` + any-of `[copy, clone, duplicate, new storage, new memory, makes a copy, makes a new]` |
| `nn_module_modes_001` | `["dropout", "batch normalization"]` | `[]` + any-of dropout + any-of batch-norm (incl. `batchnorm`, `batch norm`, `nn.batchnorm`) |
| `optim_training_loop_004` | `["effective batch size"]` | `[]` + any-of 9 synonyms (`larger effective batch`, `simulate a larger batch`, `virtual batch size`, …) |
| `data_loading_003` | `["host memory", "cuda"]` | `["cuda"]` + any-of `[host memory, pinned memory, page-locked, page locked, host side memory, host-side memory, cpu memory]` |

**New items (24, distributed across wobbly capabilities)**

| category | before → after | what the new items target |
|---|---|---|
| `shape_ops` | 4 → 10 (+6) | `flatten()` basic, `flatten(start_dim=1)`, `cat` vs `stack`, `softmax(dim=…)`, `permute([2,3,4] → (2,0,1))`, `argmax()` vs `argmax(dim=0)` |
| `nn_module_modes` | 4 → 9 (+5) | `eval()` does NOT disable grad, forget-eval consequences, BN inference uses running stats, Dropout inactive at inference, which layers change in eval |
| `debugging` | 4 → 9 (+5) | "element 0 does not require grad", device mismatch, scalar-only `.backward()`, `cat` size mismatch 3 vs 4, broadcast `[2,3]+[4,3]` fail |
| `optim_training_loop` | 4 → 8 (+4) | `zero_grad()` ordering, what `.backward()` computes, `backward` vs `step`, grad accumulation impl |
| `autograd` | 4 → 8 (+4) | `.detach()` return, `no_grad` for inference, when `requires_grad=True` matters, why `.grad` accumulates |
| `tensor_creation` | 4 → 4 (+0) | unchanged |
| `dtype_device` | 4 → 4 (+0) | unchanged |
| `hallucination_refusal` | 4 → 4 (+0) | unchanged |
| `data_loading` | 4 → 4 (+0) | unchanged |
| **total** | **36 → 60** | |

Distribution is deliberately heaviest on `shape_ops` (where ds10 2-ep
acquired a `cat` / `stack` factual inversion) and `nn_module_modes`
(where Dropout<-BN contamination is the 2-epoch signature), so future
runs can be scored against a denser signal for those two wobbly areas.

### Rescored comparison (36-item subset, softened scorer)

The softened scorer is applied to the four anchor runs using the
existing 36-item subset only (the new 24 items require a live
benchmark run and are NOT in the numbers below).

| Run | C/P/I (old scorer) → (softened) | old acc → new acc | Δ |
|---|---|---|---|
| baseline `131502` | 28/7/1 → 28/5/3 | 0.778 → 0.778 | ±0.000 |
| ds8 2-ep `145042` | 28/6/2 → 30/4/2 | 0.778 → **0.833** | +0.056 |
| ds10 1-ep `165020` | 28/4/4 → 28/4/4 | 0.778 → 0.778 | ±0.000 |
| ds10 2-ep `170716` | 26/8/2 → 27/7/2 | 0.722 → 0.750 | +0.028 |

Label-change log:
- Baseline: 2 P→I (`nn_module_modes_002` and `data_loading_003`). Both
  are genuine factual errors the new `must_not_include` rules now
  catch — the baseline conflated `model.eval()` with `no_grad`, and
  claimed `pin_memory` pins memory on the GPU. These were unearned
  partial credits under the old scorer and the new grading correctly
  demotes them to `incorrect`.
- ds8 2-ep: 2 P→C (`tensor_creation_001`, `nn_module_modes_004`). Both
  are synonym credit — the model wrote valid NL phrasings that the
  soften now accepts.
- ds10 1-ep: no label changes. Already aligned with the new scorer.
- ds10 2-ep: 1 P→C (`nn_module_modes_001`). Softening of the rigid
  dropout + batch-normalization `must_include` accepts the model's
  `batchnorm` phrasing.

### Findings from the soften pass

1. **ds8 2-ep advantage over baseline widens from +0.000 to +0.056.**
   Under the old scorer, baseline and ds8 2-ep looked identical on
   correct count. Once unearned baseline partials are demoted and
   ds8's valid synonym answers are promoted, the real SFT gain
   becomes visible. This matches the earlier hand-regrade (ds8 2-ep
   = 0.833) and validates the ds8 recipe as the legitimate ceiling.
2. **ds10 2-ep stays below baseline** (0.750 < 0.778). Softening
   narrows the gap but does not invert it. The contamination and
   factual-inversion costs documented in the ds10 2-ep audit are
   real, not scorer artifacts.
3. **No SFT run exceeded ds8 2-ep under the softened scorer.** The
   ceiling is confirmed under a fairer grading rule.
4. **Two of the four softens were sanity repairs, not favoritism.**
   Removing the rigid `["contiguous", "copy"]` from `shape_ops_001`
   and `["effective batch size"]` from `optim_training_loop_004` is
   a fix for scorer FN pressure documented in earlier runs;
   `nn_module_modes_001` and `data_loading_003` softens are
   synonym-admission fixes.

### Next step for the expanded benchmark

The 24 new items have been written to
`data/eval/benchmark_core_pytorch.jsonl` but have **no scores yet** —
no SFT or baseline answer has been generated for them. The next run
(user-side, on M1 Pro MPS) should:

1. Run the hybrid benchmark against the four anchor checkpoints on
   the new 60-item file:
   - `baseline` (hybrid_with_base_model mode)
   - ds8 2-ep checkpoint
   - ds10 1-ep checkpoint
   - ds10 2-ep checkpoint
2. Report C/P/I on the full 60-item set.
3. Do the FP/FN hand audit per CLAUDE.md on the new items
   specifically (24 fresh items, likely to surface new scorer blind
   spots).
4. Use the denser `shape_ops` and `nn_module_modes` coverage to
   decide the next lever: is it a knowledge-chunk patch, another
   targeted SFT dataset, or a base model change?

A good rerun order: ds8 2-ep first (since it's the ceiling), then
baseline (for the denominator), then the two ds10 runs (to confirm
the regression story still holds on the wider item set).

### Files touched

- Modified: `data/eval/benchmark_core_pytorch.jsonl` (36 → 60 items)
- Added: `data/eval/benchmark_core_pytorch_v1.jsonl` (backup of
  the 36-item version)
- Added: `scripts/expand_benchmark.py` (SOFTENS dict + NEW_ITEMS
  list + validator + one-shot main)
- Added: `scripts/_rescore_shim.py` (Python 3.10 `typing.NotRequired`
  compat shim so `rescore_benchmark.py` can run in the sandbox)
- Added: `experiments/eval_results/benchmark/rescored/` — four
  rescored JSONs for the anchor runs on the softened scorer
