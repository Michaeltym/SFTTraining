## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_3`
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT` (pilot slice)
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1`
- Train rows used: `50` (cap `MAX_TRAIN_ROWS=50`, seeded with `PILOT_SHUFFLE_SEED=42`)
- Val rows used: `12` (cap `MAX_VAL_ROWS=12`, same seed)
- LoRA Config:
  - `r = 16`
  - `alpha = 32`
  - `dropout = 0.05`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Trainable params: `18,464,768 / 1,562,179,072` (`1.1820%`)
- Device: `mps` (Apple Silicon)
- Train time: `221.64s` (7 steps, ~32s/step average)
- Train loss (running avg): `1.8094`
- Val loss: `1.4175`
- Post-SFT checkpoint: [Qwen-Qwen2.5-1.5B-dataset_3-8-0.0001.pt](../../data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_3-8-0.0001.pt)
- Post-SFT benchmark (hybrid mode): [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-111202.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-111202.json)
- Today's mainline base+hybrid reference: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-101108.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-101108.json) (31/5/0 = 0.8611)
- Timestamp: `2026-04-21`

## Goal

First SFT pilot on top of the day's mainline retrieval + routing stack
(see `hybrid-v2-Qwen-Qwen2.5-1.5B-2026-04-21.md` for Session A/B/C).

The question for this pilot was narrow: **does running a short LoRA SFT on a
50-row slice of `dataset_3` produce a measurable change on `benchmark_core`
relative to base + hybrid?**

This was a pilot, not a scale-up. The target was signal detection, not
best-possible model quality. A single epoch on 50 rows was chosen so the run
fits into one MPS training session and so we do not change both data size and
epoch count in the same experiment.

Secondary goal: validate the `torch.mps.empty_cache()` fix added to
`src/training.py` after observing catastrophic MPS slowdown (`8s -> 45s -> 444s`
per step without the fix).

## Findings

### Raw result

- SFT + hybrid: `32 correct / 4 partially_correct / 0 incorrect = 0.8889`
- base + hybrid (today mainline): `31 / 5 / 0 = 0.8611`
- Surface delta: `+1 correct`, `-1 partially_correct`, `0 incorrect`.

### MPS performance fix works

- With `torch.mps.empty_cache()` after `optimizer.step()`, step times
  stabilised in the `28-42s` range across the 7 training steps.
- No exponential growth. Previous run without the fix had gone `8s -> 53s ->
  497s` before being killed.
- Step 7 was `8.77s` because it was a partial batch of 2 rows (50 / 8 = 6 full
  batches + 1 partial), not a regression.
- Validation loop did not show the slowdown pattern either (2.03s for 12 rows
  under `torch.no_grad()`).

### Training signal

- `val_loss (1.4175) < train_loss running_avg (1.8094)`. This is unusual but
  not alarming for a 1-epoch pilot: the train running average includes the
  first few steps where the LoRA adapters had not adapted at all, while the
  validation is evaluated on the final weights. Per-step train losses
  (~`1.69 / 1.62 / 1.79 / 2.09 / 1.58 / 1.16 / 2.71` reconstructed from the
  running average) are noisy over only 7 steps so no clear trend.
- Training did not diverge. Checkpoint saved on the first (and only) epoch.

### Benchmark result needs false-positive correction

Reviewed all 36 answers in the SFT + hybrid run, and for a fair comparison
also reviewed all 36 answers in the base + hybrid mainline run, against the
same standard: **scorer labels `correct`, but the answer contains an
internal contradiction, a wrong worked example, or a wrong factual
statement about the API.**

#### False positives in SFT + hybrid (32/4/0)

- `autograd_001` [correct] — prescribes `optimizer.zero_grad()` as the fix for
  "tensor does not require gradients". The real fix is to ensure a leaf
  tensor with `requires_grad=True` is in the computation graph. Principle
  before that sentence is correct, so scorer saw `requires_grad` and
  `autograd` and passed.
- `debugging_002` [correct] — worked example claims `torch.cat([a, b],
  dim=0)` with `a.shape = [2, 3]` and `b.shape = [4, 3]` raises. That cat
  call is legal and returns `[6, 3]`. Principle ("same shape in every
  dimension except concat dim") is correct.
- `debugging_004` [correct] — describes `torch.argmax(x, dim=1)` as returning
  "the index of the overall maximum in the flattened tensor as a scalar
  tensor". That is the behaviour of `torch.argmax(x)` (no `dim`). With
  `dim=1`, argmax returns indices along `dim=1` with the other dims kept.
- `shape_ops_003` [correct] — says `torch.cat is stricter than torch.stack`
  then repeats the same "same shape except concat dim" rule for both,
  contradicting itself. Answer truncated at 128 tokens.

Also recorded from the partial-credit bucket (not FPs, but factually wrong
material that the scorer partially caught):

- `tensor_creation_001` [partially_correct] — says `as_tensor` is "safer to
  modify the original data without affecting the tensor". Backwards:
  `as_tensor` shares memory, so mutating the numpy array **does** affect the
  tensor. Then the last sentence says the opposite, correctly. Internal
  contradiction.
- `data_loading_003` [partially_correct] — says `pin_memory` "ensures the
  data is already on the GPU memory". Wrong: `pin_memory` page-locks host
  CPU memory. Does not move data to GPU.

#### False positives in base + hybrid (31/5/0)

- `autograd_001` [correct] — **same FP as SFT run**. Recommends
  `optimizer.zero_grad()` as the fix. This is shared between base and SFT,
  so it is most likely an artifact of the RAG passage / prompt template, not
  an SFT-introduced error.
- `tensor_creation_001` [correct] — asserts `torch.tensor` cannot handle
  NumPy arrays. `torch.tensor(np.array(...))` is in fact supported, it just
  copies. Several downstream claims ("does not support the same device
  placement as torch.tensor") are wrong too.
- `nn_module_modes_001` [correct] — describes train and eval mode with the
  same behaviour: "In training mode, dropout layers are disabled and batch
  normalization layers use their stored running statistics instead of
  computing statistics from the current batch." That is eval behaviour. Then
  the next sentence describes eval mode the same way. Scorer matched
  `dropout` and fuzzy-matched `batch normalization` and passed.
- `optim_training_loop_001` [correct] — "After calling zero_grad(),
  gradients from previous steps are added to current gradients, leading to
  incorrect updates." zero_grad clears, it does not add. Next paragraph says
  the correct version ("If you skip zero_grad(), gradients from previous
  steps will be added"). Self-contradicting within the same answer.

Also recorded as soft issues but not hard FPs:

- `shape_ops_003` (base) — says `stack` requires same shape "except the
  stacking dimension", which is a conceptual slip (stack has no "stacking
  dimension that can differ"; all shapes must match fully).
- `autograd_004` (base) — lists "store intermediate activations for logging"
  as a use case of `requires_grad=False`. That is `detach`'s job, not
  `requires_grad=False`'s.
- `nn_module_modes_002` (base) — says `model.eval()` is "important for
  preventing overfitting during training". eval is an inference-mode switch,
  not an overfitting control.
- `debugging_002` (base) — both worked examples are success cases, answer
  never shows the failure case the question asked about. Principle is
  correct.

### FP-adjusted comparison

If every hard FP is downgraded from `correct` to `partially_correct`:

| Run           | Original    | FP count | Adjusted      |
|---------------|-------------|----------|---------------|
| base + hybrid | `31/5/0`    | 4        | `27/9/0 = 0.750` |
| SFT  + hybrid | `32/4/0`    | 4        | `28/8/0 = 0.778` |

### What this really means

The FP sets in the two runs are almost disjoint:

- SFT **fixed** (were FPs in base, became correct in SFT):
  `tensor_creation_001`, `nn_module_modes_001`, `optim_training_loop_001`.
- SFT **broke** (were correct in base, became FPs in SFT):
  `debugging_002`, `debugging_004`, `shape_ops_003`.
- Shared across both: `autograd_001`.

So SFT is not a uniform improvement. It is a **swap**: it pulled some
previously confused answers into clean concept-level prose, but introduced
new mistakes in precise API behaviour (wrong worked examples, wrong scalar
vs per-dim output of argmax, wrong stricter-than direction for cat/stack).

The pattern of what SFT fixed (concept-level confusion) vs what it broke
(precise API behaviour detail) is consistent with a training slice whose 50
rows skewed toward conceptual explanation and away from hard shape /
behaviour specifics. With a 50-row cap over a 200-row source file and a
`seed=42` shuffle, we did not audit the category distribution of the actual
slice used.

## Next Step

**Do not scale to 200 rows yet.** The `+1 correct` signal is within FP noise
(`28 vs 27` after adjustment). Before any further SFT scaling, tighten the
measurement layer, then rerun:

1. Extend `src/v2/benchmark/scorer.py` (or equivalent) so a handful of known
   FP patterns are caught:
   - worked example consistency — when a row has `must_not_include` or
     shape literals, cross-check the example arithmetic against the stated
     principle.
   - self-contradiction check — flag answers where a phrase and its direct
     inverse both appear (e.g. "clears" and "added to" both applied to the
     same subject in one answer).
   - `dim=` grounded checks for argmax / max / sum / flatten — the answer
     must not describe per-dim reductions as "scalar" or "flattened".
2. Add the six FPs / FP-like failures above as **scorer unit tests** so we
   do not regress this again. Seeding the scorer with the actual strings
   from today's runs gives us a concrete target.
3. Audit the `autograd_001` RAG passage — same wrong "zero_grad fixes it"
   conclusion appears in base and SFT. This is not an SFT issue, it is a
   retrieval / template issue that should be fixed before any SFT
   comparison is trusted.
4. After the scorer is tighter and `autograd_001` is independently fixed,
   rerun **base + hybrid** on `benchmark_core` to re-establish the baseline
   under the new scorer. Keep that number as the real plateau.
5. Only then, rerun the SFT pilot (same config, same seed) under the new
   scorer. If the delta is still near zero or negative, the SFT-broke /
   SFT-fixed swap in this run is the true signal: concept-level gain, API
   precision loss, net near zero. In that case the next SFT experiment
   should change the **training data mix**, not the **scale**, to include
   more argument-level, shape, and debugging rows.

## Other Important Info

- Pilot data-slice infrastructure added this session:
  - `MAX_TRAIN_ROWS`, `MAX_VAL_ROWS` in `src/config.py` (both default to
    `None` for full dataset, currently `50 / 12` for the pilot).
  - `PILOT_SHUFFLE_SEED = 42` in `src/config.py`, threaded through
    `load_jsonl_data(file_path, max_rows, shuffle_seed)` so the slice is
    representative instead of taking the first N rows.
  - `src/training.py` prints `[data] training rows used: ... (cap: ...,
    seed: ...)` and `[data] validation rows used: ...` on each run for
    reproducibility.
- MPS slowdown fix: added `torch.mps.empty_cache()` after `optimizer.step()`
  guarded by `is_mps = device.type == "mps"` so CUDA and CPU runs are
  unchanged. Per-step timing print updated to `Step: X.XXs Cum: Y.YYs` for
  clearer diagnostics.
- Training entry point reminder: `python -m src.main` (v1 training loop).
  `src/v2/main.py` has `baseline / evaluate / rag_evaluate / hybrid /
  hybrid_with_base_model` modes but **no** `train` mode. Do not try to
  train from v2.
- Post-SFT eval was run with `MODE = MODE_HYBRID` and `BENCHMARK_NAME =
  "core"` (not smoke). The smoke / core distinction matters: smoke is a
  12-row subset and is not comparable to base+hybrid core numbers.
- Checkpoint path is determined at config load time from
  `MODEL_NAME / DATASET_NAME / BATCH_SIZE / LEARNING_RATE`, so the SFT
  checkpoint overwrote any previous same-config checkpoint in
  `data/checkpoints/`. Acceptable for a pilot; for future comparisons
  across SFT variants, move to timestamped checkpoint names.
- 50-row slice distribution was not explicitly audited post-hoc. One
  possible reason SFT broke `debugging_*` rows is that the slice may
  under-sample debugging / shape behaviour rows relative to comparison /
  definition rows. This would be worth checking before the next SFT run.
- This log closes the SFT pilot. The next round of changes is expected to
  target the scorer and `autograd_001` RAG bug, not the training loop.
