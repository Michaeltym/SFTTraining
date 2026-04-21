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
- Post-fix base+hybrid verification: [hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-171427.json](../eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-171427.json) (30/6/0 = 0.8333 raw, 29/7/0 FP-adjusted)
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

### Post-fix verification (Next Step #3 — done)

After the pilot, `autograd_001` was root-caused to a **symbol-index
pollution** bug, not a RAG passage bug as originally suspected. Body text
in several docs contains example code using object-prefix placeholders
(`loss.backward()`, `optimizer.zero_grad()`, `model.train()`). These were
being emitted as symbol-index keys during corpus build, each mapping back
to every doc whose body mentioned the placeholder. A query for
`loss.backward` was then exact-matching the polluted key and pulling 6
unrelated docs — the top doc was `api_docs_optimizer_zero_grad`, which
explains why both base and SFT confidently prescribed `zero_grad()` as
the fix.

Fix applied in `src/v2/corpus/build.py`: `OBJECT_EXAMPLE_PREFIXES =
{"loss", "optimizer", "model", "f"}` is now a **noise** set, not an
allowlist. `is_noise_symbol` returns `True` for any symbol whose first
segment is in that set, so these placeholders never reach the symbol
index. Canonical API names (`Tensor.backward`, `Optimizer.zero_grad`,
`Module.train`) remain indexed under their real names; queries written in
placeholder form fall through to them via the alias fallback on the short
form (`backward`, `zero_grad`, `train`).

Index rebuilt: `99 -> 87` entries, 12 polluted entries removed, canonical
entries intact.

Scope verification — smoke first, then core:

- **Smoke** (12 rows, regression harness): 10/12 byte-equal to pre-fix,
  only `autograd_001` and `nn_module_modes_001` changed. Exactly the two
  queries whose symbols were in the pollution set. Surgical.
- **Core** (36 rows): `30 byte-equal / 6 content-changed / 0
  retrieval-only`. The 6 touched rows are all in the
  `autograd / optim / nn.Module.modes` cluster, exactly where
  object-prefix pollution was concentrated. No unexpected ripples on the
  other 30.

Raw score drift: `31/5/0 -> 30/6/0`. The `-1 correct` is `autograd_001`
demoting from `correct` to `partially_correct`, which is the **desired**
outcome: the old answer was a scorer-labelled correct while containing a
dangerous wrong fix; the new answer is less helpful but not actively
misleading.

Per-row FP re-audit on the four base-run FPs identified earlier:

| Row                      | Pre-fix label | Post-fix label | FP still present? |
|--------------------------|---------------|----------------|-------------------|
| `autograd_001`           | correct       | partially_correct | No — label now captures the quality; new answer no longer prescribes `zero_grad()` |
| `nn_module_modes_001`    | correct       | correct        | No — new answer correctly says training mode applies dropout (old said disabled in both modes) |
| `optim_training_loop_001`| correct       | correct        | No — self-contradiction ("clears" vs "added to") removed, answer now clean on `.grad = None` |
| `tensor_creation_001`    | correct       | correct        | **Yes, unchanged** — unrelated to symbol-index fix, still wrong about `torch.tensor` not accepting numpy arrays |

Post-fix FP-adjusted score for base+hybrid: `29/7/0 = 0.8056`, up from
`27/9/0 = 0.750` before the fix. Quality gain of `+2 correct` on the
FP-adjusted scale even though raw count went `-1`. One remaining hard FP
(`tensor_creation_001`), which is a training-data / phrasing issue, not a
retrieval issue.

### Implication for the SFT comparison

The pre-fix comparison (`27/9/0` base vs `28/8/0` SFT, adjusted) was
based on a retrieval stack that was silently biasing both runs. The
symbol-index fix benefits base and any future SFT run equally, so the
apples-to-apples comparison still needs a **fresh SFT + hybrid run** on
the post-fix stack. That re-run is not required yet; it is gated on the
scorer work (Next Step #1 and #2).

### Scorer tightening (Next Step #1 — done)

Added a second forbidden-content layer so known FP patterns are caught
without relying on a re-run of inference.

Schema change in `src/v2/benchmark/types.py`:

- `BenchmarkItem` gains `must_not_include_regex: NotRequired[list[str]]`.
- `BenchmarkResultItem.notes` changed from `NotRequired[str]` to
  `NotRequired[BenchmarkLabelNotes]`, where `BenchmarkLabelNotes` is a
  structured `TypedDict` that mirrors the item's rule sections
  (`must_include`, `must_include_any_of`, `must_not_include`,
  `expected_symbols`). The structured notes record which patterns /
  phrases / symbols matched and which did not, so partially-correct
  reasons are now visible in the output JSON instead of being a freeform
  string.

Scorer change in `src/v2/benchmark/label.py`:

- New helper `get_matched_forbidden_regex(patterns, normalized_answer)`
  runs `re.search(..., re.IGNORECASE)` against the normalized answer.
  Its docstring spells out the normalization contract (lowercased +
  whitespace collapsed to single space + strip, punctuation preserved)
  so pattern authors know what shape of text their regex will see.
- Two-tier forbidden semantics (option 丙 — explicitly chosen over the
  uniform "both soft" alternative):
  - `must_not_include` (keyword phrase): **hard rule**, hit forces
    `incorrect`. Preserves the historical keyword-layer semantics so
    older result files remain directly comparable on this axis.
  - `must_not_include_regex` (per-row regex): **soft rule**, hit caps
    the label at `partially_correct`. Acts as an additive experimental
    signal we can observe before deciding to promote any pattern to a
    hard rule.
- Historical impact of the two-tier choice on the three runs analysed
  in this log: zero. `must_not_include` (keyword) fired 0 times across
  all 36 rows of all three runs, so the soft-vs-hard split only affects
  future authoring.

Seven FP patterns added to `data/eval/benchmark_core_pytorch.jsonl`
(one `must_not_include_regex` entry per FP-ed row, two patterns on
`tensor_creation_001`):

| Row                        | Pattern                                                |
|----------------------------|--------------------------------------------------------|
| `tensor_creation_001`      | `torch\.tensor cannot`                                 |
| `tensor_creation_001`      | `safer to modify[^.]{0,40}without affecting`           |
| `shape_ops_003`            | `cat is stricter than`                                 |
| `autograd_001`             | `optimizer\.zero_grad`                                 |
| `nn_module_modes_001`      | `training mode[^.]{0,60}(disabled|off|skip)`           |
| `optim_training_loop_001`  | `added to (current|previous|the) gradients`            |
| `debugging_002`            | `\[4, ?3\][\s\S]{0,150}(raise|fail|error)`             |
| `debugging_004`            | `flattened tensor as a scalar`                         |

Each pattern was verified against the three historical runs
(`101108` base mainline, `171427` post-fix base, `111202` SFT) by
walking both `normalize_text(answer)` and raw `answer` and confirming
identical hit behaviour — i.e. none of the seven patterns smuggle in a
hidden dependency on raw whitespace or case. Same-row cross-run checks
confirm each pattern fires exactly on the intended FP answers and not
on same-row answers that are substantively correct.

### Rescore across the three historical runs (Next Step #4 — done)

New script: `scripts/rescore_benchmark.py`. Loads a previous benchmark
result JSON, re-labels every row using the current
`get_benchmark_label`, recomputes the summary, and writes a copy to
`experiments/eval_results/benchmark/rescored/<stem>-rescored.json`.
Inference-time fields (`answer`, `citations`, `used_symbols`,
`abstained`, `confidence_band`, `retrieval_debug`) are preserved
untouched. Two new top-level fields (`rescored_at`, `rescored_from`)
record the rescoring provenance.

Rescored outputs:

| Run                          | Before    | After (new scorer) | Label changes |
|------------------------------|-----------|--------------------|---------------|
| `171427` post-fix base       | `30/6/0 = 0.833` | `29/7/0 = 0.806` | 1 (`tensor_creation_001`) |
| `101108` base mainline       | `31/5/0 = 0.861` | `27/9/0 = 0.750` | 4 (`tensor_creation_001`, `autograd_001`, `nn_module_modes_001`, `optim_training_loop_001`) |
| `111202` SFT + hybrid        | `32/4/0 = 0.889` | `28/8/0 = 0.778` | 4 (`shape_ops_003`, `autograd_001`, `debugging_002`, `debugging_004`) |

Numbers match the FP-adjusted column in the earlier table, confirming
the scorer changes simply automate what was previously a by-hand
audit. Zero rows moved between `partially_correct` and `incorrect`
because no `must_not_include` keyword hits were observed anywhere.

Reference paths for the rescored JSONs:

- [post-fix base rescored](../eval_results/benchmark/rescored/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-171427-rescored.json)
- [base mainline rescored](../eval_results/benchmark/rescored/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-101108-rescored.json)
- [SFT + hybrid rescored](../eval_results/benchmark/rescored/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-21-111202-rescored.json)

### Scorer-adjusted SFT comparison

With all three runs rescored under the new scorer, the current
apples-to-apples numbers are:

| Run                      | correct | partially | incorrect | accuracy |
|--------------------------|---------|-----------|-----------|----------|
| `101108` base, pre-fix   | 27      | 9         | 0         | `0.750`  |
| `171427` base, post-fix  | 29      | 7         | 0         | `0.806`  |
| `111202` SFT, pre-fix    | 28      | 8         | 0         | `0.778`  |

The SFT pilot (`111202`) was run on **pre-fix retrieval**. The
appropriate baseline for it is `101108` (also pre-fix), not `171427`.
Under that comparison the SFT pilot is `+1 correct, -1 partially`
relative to its matched baseline, net `+0.028` accuracy — still within
FP-audit noise, and the retrieval fix alone (`101108 -> 171427`) moves
the baseline by more than the SFT did (`+2 correct` vs `+1 correct`).
A fresh SFT + hybrid run on the post-fix retrieval stack is still
required to get a clean read on the SFT contribution.

## Next Step

**Do not scale to 200 rows yet.** The `+1 correct` signal is within FP noise
(`28 vs 27` after adjustment). Before any further SFT scaling, tighten the
measurement layer, then rerun:

1. ~~Extend `src/v2/benchmark/scorer.py` (or equivalent) so a handful of
   known FP patterns are caught~~ — **done**. Shipped as
   `must_not_include_regex` (soft forbidden layer) in
   `src/v2/benchmark/types.py` and `src/v2/benchmark/label.py`. Seven
   patterns added to `data/eval/benchmark_core_pytorch.jsonl`. See
   "Scorer tightening" above for the schema change and the two-tier
   (hard keyword vs soft regex) semantics.
2. ~~Add the six FPs / FP-like failures above as **scorer unit tests**~~
   — **skipped for now**. No test framework in the repo, scorer is a
   small pure function, and the seven regex patterns have already been
   empirically verified across all three historical runs (see "Scorer
   tightening" above). The rescore script serves as the regression
   check. Revisit if/when the scorer accumulates more rules or we add
   `tests/` for other reasons.
3. ~~Audit the `autograd_001` RAG passage~~ — **done** (see "Post-fix
   verification" above). Root cause was symbol-index pollution from
   object-prefix placeholders, not the RAG passage itself. Fix landed in
   `src/v2/corpus/build.py`, verified with smoke + core re-run. Three of
   the four base-run FPs are eliminated, one remains
   (`tensor_creation_001`, unrelated to retrieval).
4. ~~After the scorer is tighter, rerun **base + hybrid** on
   `benchmark_core` to re-establish the baseline under the new
   scorer~~ — **done via rescore** (see "Rescore across the three
   historical runs" above). Baseline under the new scorer is
   `29/7/0 = 0.806` (post-fix `171427`). The pre-fix baseline
   `101108` rescores to `27/9/0 = 0.750`.
5. **Next: fresh SFT + hybrid run on the post-fix retrieval stack, same
   config and seed, new scorer.** Compare against `171427` (`29/7/0 =
   0.806`), not the pre-fix `101108`. If the delta is still near zero
   or negative, the SFT-broke / SFT-fixed swap in this run is the true
   signal: concept-level gain, API precision loss, net near zero. In
   that case the next SFT experiment should change the **training data
   mix**, not the **scale**, to include more argument-level, shape,
   and debugging rows.

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
- **Update after post-fix verification:** the `autograd_001` fix landed
  as a symbol-index build change (not a RAG passage change). Remaining
  next-round work is the scorer. All retrieval-side changes are in
  `src/v2/corpus/build.py` under `OBJECT_EXAMPLE_PREFIXES` and
  `NOISE_OBJECT_PREFIXES` with inline comments explaining each rule.
- **Update after scorer tightening:** scorer work is now complete for
  this pilot. Two-tier forbidden layer (keyword = hard, regex = soft)
  is documented in `get_matched_forbidden_regex`'s docstring, and the
  seven regex patterns live alongside the per-row rules in
  `benchmark_core_pytorch.jsonl`. The `scripts/rescore_benchmark.py`
  tool is available for future scorer/rule changes — run it against
  the three archived JSONs listed above (or any other result JSON) to
  re-label without re-running inference.
