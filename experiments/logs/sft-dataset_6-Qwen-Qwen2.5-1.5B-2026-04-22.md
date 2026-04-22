## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_6` (= `dataset_5` mandatory-row scaffold + 6 direct-negation anti-pattern rows for dropout-vs-BN and argmax-returns-scalar)
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT` (same recipe as ds4/ds5 pilot)
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1`
- Train rows: `50`; Val rows: `12`
- LoRA Config (unchanged): `r=16, alpha=32, dropout=0.05, target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`
- Trainable params: `18,464,768 / 1,562,179,072` (`1.1820%`)
- Device: `mps` (Apple Silicon M1 Pro, 16 GB)
- Post-SFT checkpoint: [Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt](../../data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt)
- Benchmark result (hybrid mode): [hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-104632.json](../eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-104632.json)
- Baseline reference: [rescored `171427`](../eval_results/benchmark/rescored/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-21-171427-rescored.json) `29/7/0 = 0.806` under patched scorer
- Timestamp: `2026-04-22`

## Goal

Test whether **direct-negation anti-pattern rows** — training rows whose `input` asks the exact wrong belief and whose `output` opens with "No, …" followed by the correct mechanism — close the two recurring false positives that the softer `dataset_5` intervention failed to move:

1. `nn_module_modes_003` (dropout after `model.eval()` → model produced "dropout layers to use their stored running statistics", conflating with batch norm).
2. `debugging_004` (`torch.argmax(x, dim=1)` shape → model produced "flattened tensor as a scalar", collapsing the `dim=1` argument).

Hypothesis (from ds5 debrief): 2–3 generic explainer rows scattered in a 50-row SFT slice are not enough signal to overwrite a pretraining-era bias in 1 epoch of LoRA at `r=16`. Reframing the same counter-evidence as direct negation (`Q: "Does dropout use running stats?" → A: "No, dropout has no running statistics at all. …"`) is a stronger per-row signal because the refutation is the topic of the row rather than a side remark.

Dataset size, val size, LoRA config, retrieval stack, and mandatory-row scaffold all held fixed — the only changing variable is the presence of 6 direct-negation rows in the training pool.

## Findings

### Data

- `data/raw/training/dataset_6_addendum.jsonl` — 6 new rows: 3 dropout-vs-BN refutations, 3 argmax-shape refutations. Each row pairs a wrong-belief question with a "No, …" / "It does not." / "That framing is wrong …" opener.
- `scripts/build_dataset_6.py` — forked from `build_dataset_5.py`. Reads three pools (`dataset_3.jsonl`, `dataset_3_addendum.jsonl`, `dataset_6_addendum.jsonl`) with `(input, output)` dedup. `MANDATORY_INPUT_SUBSTRINGS` extended from 4 to 10 substrings (4 from ds5 kept + 6 new), each verified unique against the pool so no mandatory pre-fill silently duplicates.
- `data/raw/training/dataset_6.jsonl` (50 rows): 11 mandatory pre-fills (6 `shape_ops` + 5 `nn_module_modes`) + 39 random-sampled. 39 rows are shared with `dataset_5`; the other 11 differ (6 new anti-pattern rows, 5 RNG drift from the larger mandatory set).
- `data/raw/validation/dataset_6.jsonl` (12 rows): same validation pool as `dataset_5` — validation drift is not the variable under test.

### Training

Run completed cleanly after a host-side restart (swap had been exhausted on prior attempts). Step sequence after MPS cold start was `27.35s, 29.81s, 12.94s, 15.37s, 6.05s, …` — normal warmup profile for this config. Loss sequence `2.40 → 2.30 → 2.22 → 2.21 → 2.09`, monotone decrease, no instability.

### Benchmark — all runs rescored under the patched scorer

| Run                    | c / p / i | accuracy | vs baseline |
|------------------------|-----------|----------|-------------|
| baseline `171427`      | 29 / 7 / 0 | 0.806 | — |
| `dataset_4` `205027`   | 28 / 6 / 2 | 0.778 | -0.028 |
| `dataset_5` `212112`   | 28 / 6 / 2 | 0.778 | -0.028 |
| **`dataset_6` `104632`** | **27 / 7 / 2** | **0.750** | **-0.056** |

Raw scorer shows ds6 is the worst SFT run so far. The per-item diff is what matters, not the aggregate.

### Per-item diff: `dataset_5` → `dataset_6` (patched scorer, both runs)

**Improvements (2):**

| id | ds5 | ds6 | notes |
|----|-----|-----|-------|
| `nn_module_modes_003` | incorrect | **correct** | **Target FP #1 closed.** ds5 answer contained "dropout layers to use their stored running statistics"; ds6 answer drops that sentence entirely and just says "dropout layers are disabled and pass all inputs through without zeroing any elements". Direct-negation rows worked on this one. |
| `data_loading_002`    | partial   | correct   | Incidental gain. ds5 lost the `must_include` "variable length" exact match; ds6 restored it. |

**Regressions (3):**

| id | ds5 | ds6 | notes |
|----|-----|-----|-------|
| `autograd_001`            | correct   | partial   | **Real semantic regression.** ds5 correctly explained `requires_grad=False → autograd does not track`; ds6 drifted to an irrelevant "computation graph is freed after backward" explanation and lost both required symbols (`requires_grad`, `autograd`). Unrelated to the ds6 intervention — looks like LoRA-induced style shift. |
| `nn_module_modes_004`     | correct   | partial   | Scorer-sensitive. Both answers say "running mean and running variance" and "running statistics"; the `must_include` "running statistics" fuzzy-matches in both. ds6 is demoted because it dropped the `model.train()` / `model.eval()` tokens that matched `Module.train` / `Module.eval` expected symbols. Substantively still correct on the BN semantics, but lost the concrete API anchor. |
| `optim_training_loop_003` | correct   | incorrect | **Scorer false negative.** ds6 answer says "model state_dict()" / "optimizer state_dict()" without the dot, but adds `torch.save()` which ds5 did not. The `must_include` `model.state_dict` / `optimizer.state_dict` requires the dot, so the scorer reports no match. Semantically ds6 is at least as correct as ds5, arguably better. |

**Target FP #2 status — `debugging_004` — still `incorrect` in ds6.** Both ds5 and ds6 produce "returns the index of the overall maximum in the flattened tensor as a scalar tensor". The 3 argmax direct-negation rows did not dent this belief.

### Hand-regrade

- `optim_training_loop_003`: scorer labels `incorrect`, true label `correct` (torch.save + both state_dicts mentioned, just missing the dot the scorer looks for). Net +1 correct, -1 incorrect.
- `nn_module_modes_004`: scorer labels `partial`, true label borderline. I leave it at `partial` because the answer really did drop the `Module.train` / `Module.eval` anchor — that is a meaningful loss of API grounding for a question whose category is `nn_module_modes`.
- All other items: scorer label matches hand-regrade.

Hand-regrade: **28 correct / 6 partial / 2 incorrect = 0.778**, i.e. same as ds4/ds5. The raw 0.750 number overstates the regression; the true story is "closed one FP, opened one new regression on an unrelated row, net flat."

### What this tells us about direct-negation anti-pattern rows

- **They work when the wrong belief is a phrase-level confusion.** `nn_module_modes_003` flipped because the wrong belief is a concrete phrase ("dropout uses running statistics") and the direct-negation rows repeatedly say "No, dropout has no running statistics at all". The model learned to not produce that phrase at all after SFT; instead of substituting the right phrase, it simply stopped emitting the whole sentence that was hosting the wrong one.
- **They do not work when the wrong belief is a structural/shape claim that rides on the argument `dim=1`.** `debugging_004` did not flip. The argmax answer is built from three habitual fragments — "returns the index of the overall maximum", "in the flattened tensor", "as a scalar tensor" — and the 3 direct-negation rows only push back on the whole compound. The model is happy to reproduce the compound because each of its fragments is individually a true statement about some form of argmax (just not `argmax(x, dim=1)`). Direct-negation at 3 rows / 1 epoch is not enough signal to get the model to condition its output on the `dim=1` argument.
- **SFT floor is unchanged at the overall accuracy level.** ds4 / ds5 / ds6 all land at hand-regrade 0.778 on this benchmark against a base-model baseline of 0.806. No additive-per-intervention gain; we are trading one FP for another regression. The 1-epoch LoRA regime at `r=16, lr=1e-4` is not pulling its weight on this dataset.

## Next Step (original plan before 3-epoch run)

The scorecard is now clear: at 50 rows / 1 epoch / `r=16`, SFT on this dataset is a wash. To move off the 0.778 floor, pick one of:

1. **More epochs** (2–3 epochs on the same ds6 data). The `nn_module_modes_003` win shows direct-negation has signal; more passes over the same 6 rows may compound the signal enough to also close `debugging_004`. Cheap to try — same data, just change `EPOCHS` in `src/config.py`. If this moves the argmax FP, the answer is "1 epoch was the bottleneck"; if not, the bottleneck is row count or phrasing.
2. **Higher LoRA capacity** (`r=32, alpha=64`). Same data, larger adapter. Moves the cost/benefit away from "tiny adapter can't absorb new signal" if that is the bottleneck.
3. **Scale ds6 to 200 rows + 24 val** (the CLAUDE.md scale-up default). Keeps all 11 mandatory rows and grows the random-sampled tail 4×. Dilutes the per-row signal ratio but gives each row more neighbors of the same semantic shape. Paired with 1 epoch this is cheap; paired with 2 epochs it is the natural next rung.
4. **Stop using SFT to fix FPs. Fix the knowledge store instead.** Both target FPs are retrieval-adjacent: the dropout question's correct answer should have been served by a `Dropout` knowledge chunk, not synthesized by the model. An audit of `data/knowledge/pytorch_docs` for `Dropout`-vs-`BatchNorm` separation, and for `torch.argmax` argument semantics, may be the higher-ROI lever than any further SFT tuning on 50-row slices.

Recommended order: **(1) first** — it is the smallest variable change and directly tests whether the 1-epoch ceiling is the blocker for `debugging_004`. If (1) does not move the argmax FP, go to **(4)** rather than (2) or (3).

Also worth investigating before the next training run: **why `autograd_001` regressed in ds6** despite no `autograd` rows being changed between ds4/ds5/ds6. The ds4 slice had zero autograd rows; ds5 slice had zero autograd rows; ds6 slice has zero autograd rows. Yet ds6's answer drifted from correct to partial. Suspicion: the 6 new anti-pattern rows share a "No, X is actually Y because Z …" template that may be biasing LoRA toward a new-style answer template for unrelated questions. Check if ds6's training loss curve differs materially from ds5's, and consider softening the direct-negation opener so it does not flood the training signal with a single answer shape.

## 3-Epoch Resume Findings

The "more epochs" option was run next: `MODE = MODE_RESUME`, `EPOCHS = 3`. The resume loop in `src/resume.py` picks up from the 1-epoch ds6 checkpoint, runs epochs 2 and 3, and saves a new adapter if validation loss improves. Both epochs improved val loss, so the adapter on disk at `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt` is now the **epoch-3** version.

### Training loss curve

| Epoch | Train loss (avg) | Val loss | Adapter saved? |
|------:|-----------------:|---------:|:--------------:|
| 1 (original ds6 run) | 2.0923 (last step) | — | ✓ |
| 2 (resume) | 1.5943 | 1.4165 | ✓ |
| 3 (resume) | 1.2305 | 1.3897 | ✓ |

Val loss kept descending (1.42 → 1.39), which *looks* like healthy learning. But the train-val gap inverted between epoch 2 (train 1.59 < val 1.42, still underfit) and epoch 3 (train 1.23 < val 1.39, early overfit). On a 50-row training set with 7 steps per epoch this gap shift is the canonical overfit tell, and the benchmark confirms it.

### Benchmark — 3-epoch run `113856` under patched scorer

| Run                    | c / p / i | accuracy | vs baseline | vs 1-ep ds6 |
|------------------------|-----------|---------:|------------:|------------:|
| baseline `171427`      | 29 / 7 / 0 | 0.806 | — | +0.056 |
| ds5 `212112`           | 28 / 6 / 2 | 0.778 | -0.028 | +0.028 |
| ds6 1-ep `104632`      | 27 / 7 / 2 | 0.750 | -0.056 | — |
| **ds6 3-ep `113856`**  | **24 / 8 / 4** | **0.667** | **-0.139** | **-0.083** |

Raw regression is -0.083 from 1-ep and -0.139 from baseline. **Worst of all four SFT runs.** This is the key point: val loss was still decreasing, yet benchmark collapsed. On a 50-row training set at 3 epochs, val loss and downstream benchmark have fully decoupled.

### Per-item diff: ds6 1-epoch → ds6 3-epoch

**Improvements (3) — the intervention worked on:**

| id | 1-ep | 3-ep | notes |
|----|-----|-----|-------|
| **`debugging_004`** | incorrect | **correct** | **Target FP #2 closed.** 1-ep answer started with "returns the index of the overall maximum in the flattened tensor as a scalar tensor"; 3-ep answer opens with "returns indices of the maximum values along dimension 1". The extra two epochs on the 3 argmax direct-negation rows were enough to overwrite the flatten-to-scalar compound. |
| `autograd_001` | partial | correct | 1-ep's drift to "computation graph is freed" recovered; 3-ep correctly mentions `requires_grad` and autograd tracking. |
| `optim_training_loop_003` | incorrect | correct | The "model state_dict()" dotless phrasing flipped back to "model.state_dict()" dotted form, clearing the `must_include` strict match. |

And `nn_module_modes_003` (target FP #1) remained `correct` — the 3-ep run preserved the 1-ep win without degradation.

**Regressions (7) — overfit damage, mostly substantive:**

| id | 1-ep | 3-ep | category | nature |
|----|-----|-----|----------|--------|
| **`shape_ops_002`** | correct `[2,1,3,4]` | **incorrect `[2,3,1,4]`** | shape_ops | **Real factual error.** Classic overfit symptom — model lost the correct `unsqueeze(1)` semantic. |
| **`shape_ops_004`** | correct `[3,2,20]` | **incorrect `[3,8,5]`** | shape_ops | **Real factual error.** `flatten(start_dim=2)` should merge the last two dims of `[3,2,4,5]`; 3-ep merged the middle two instead. |
| `autograd_003` | correct | partial | autograd | **Real hallucination.** 3-ep claims `detach()` "keeps the tensor's requires_grad flag and gradient tracking" — exactly backwards. |
| `data_loading_003` | partial | incorrect | data_loading | **Real fabrication.** 3-ep claims `pin_memory` is "not a standard PyTorch API" — it is literally a standard DataLoader flag. |
| `hallucination_refusal_003` | correct | incorrect | hallucination_refusal | Refusal weakened. 3-ep dropped the phrase "not a valid" that the scorer requires for refusal items. Answer is still directionally correct but the scorer cannot tell. |
| `optim_training_loop_001` | correct | partial | optim_training_loop | Softer regression. 3-ep answer now contains the forbidden phrase "added to the current gradients", triggering the soft forbid. |
| `data_loading_004` | correct | partial | data_loading | Scorer FN. 3-ep says "Use shuffle=False for validation" which is substantively correct; scorer needs the specific word "unshuffled" which 3-ep dropped. |

Hand-regrade (being generous and recovering `data_loading_004` as a true FN): **~25 correct / 7 partial / 4 incorrect = 0.694**. Still well below baseline 0.806 and 1-ep 0.750. The two shape_ops errors and the autograd_003 + data_loading_003 hallucinations are not scorer noise — they are real factual corruption of knowledge the base model had correct.

### What 3-epoch resume tells us

- **More epochs did close `debugging_004`.** The direct-negation intervention was strong enough per row; what was missing was gradient-update volume. Two more passes over the same 6 anti-pattern rows was the threshold that got the model to stop producing the flatten-to-scalar compound.
- **But 3 epochs on 50 rows overfits the LoRA adapter.** The benchmark regressions are concentrated in categories where 1-epoch ds6 was already doing well (`shape_ops`, `data_loading`, `autograd`). The 3-epoch adapter has started to overwrite base-model knowledge with whatever happens to be in the 50-row slice, and because the slice is dominated by `shape_ops` (12 rows) and `debugging` (6 rows), categories adjacent to those (other `shape_ops` items, some hallucination refusal cases) take collateral damage.
- **Val loss is not a safe stop criterion at this scale.** Epoch 3 val loss (1.39) was the best of the three, but the benchmark said epoch 3 is the worst. With 12 validation rows and 50 training rows, validation loss is tracking memorization of the (very similar) validation distribution, not generalization to the held-out benchmark. Using val loss to decide "keep training" produced a clearly worse artifact. For this dataset size, the benchmark is the only trustworthy stop signal.
- **The 1-epoch / 3-epoch trade-off is a real Pareto frontier.** 1-ep kept `shape_ops` / `data_loading` clean and left `debugging_004` broken. 3-ep closed `debugging_004` and broke `shape_ops`. There is no dataset_6 epoch count that wins both. The interventions are not additive; they are trading against each other.
- **Confirms the `autograd_001` suspicion from the 1-epoch write-up.** 1-ep had drifted `autograd_001` to a generic "computation graph is freed" style. 3-ep recovered it back to a correct answer referencing `requires_grad` and autograd. That suggests the drift was caused by *insufficient* updates to dislodge the drifted template, not by the 6 anti-pattern rows flooding the style. More epochs helped on that front, even as they hurt elsewhere.

## Final Next Step

The scorecard for the full SFT arc on this benchmark:

| Run | baseline | ds3 (pilot) | ds4 | ds5 | ds6 1-ep | ds6 3-ep |
|-----|---------:|------------:|----:|----:|---------:|---------:|
| accuracy | **0.806** | 0.778 | 0.778 | 0.778 | 0.750 | 0.667 |

Four SFT experiments, each varying one thing at a time. **Zero of them beat the post-fix baseline.** The closest was the 0.778 floor; everything past that has been losing on one axis while winning on another. The 3-epoch ds6 confirmed the ceiling: even when SFT successfully closes a target FP, the overfit penalty on adjacent categories outweighs the win.

**Pivot.** Stop using SFT to patch benchmark FPs at this scale. Two concrete follow-ups:

1. **Audit `data/knowledge/pytorch_docs` for the two target FP areas.** Both `nn_module_modes_003` and `debugging_004` are retrieval-adjacent — the baseline run (no SFT at all) answered them correctly. The failure mode for SFT runs is that the model *ignored* clean retrieval context and produced a confabulation from base-model priors. If the knowledge store has a clean `nn.Dropout` chunk that doesn't co-mention BatchNorm, and a `torch.argmax` chunk that clearly states the `dim` argument's shape semantics, then the retrieval+base-model pipeline can answer both questions without SFT. Per CLAUDE.md *RAG Knowledge Rules*, narrowing and disambiguating knowledge chunks is a sanctioned direct change.
2. **Before any further SFT, freeze `data/eval/benchmark_core_pytorch.jsonl`.** The scorer patches applied during ds5 debugging have matured to the point where the scorer catches both target FPs. Any future SFT experiment should be measured against this frozen benchmark so the comparisons stay apples-to-apples without per-run rescoring.

If after knowledge-store cleanup we still want to rerun SFT, the recommended configuration based on everything learned so far is:

- **Dataset size**: scale to 200 rows + 24 val (CLAUDE.md scale-up default). At 50 rows, any category with 6+ rows (which includes `shape_ops`, `debugging`, `nn_module_modes`) is at risk of overfitting after 2 epochs.
- **Epochs**: 2, not 1 and not 3. 1 was not enough to close `debugging_004`; 3 was too much to keep `shape_ops` clean. The 2-epoch number is extrapolated from the val-loss curve and is the obvious next rung.
- **Direct-negation rows**: keep the dropout rows (they closed `nn_module_modes_003` cleanly at 1 epoch). The argmax rows may need more variety in phrasing — the 3 current rows all share the pattern "No. [explanation] A scalar 0-dim tensor only comes out of torch.argmax(x) with no dim argument" which may have over-specialized the model's argmax response template.

## Knowledge-Cleanup Findings (v2 source)

Followed up on the "audit knowledge store" item from Final Next Step above. Discovered a significant process error, then executed and measured the first cleanup pass.

### Process error: two knowledge stores exist, I edited the wrong one first

- `src/rag/` (v1) reads `data/knowledge/pytorch_docs/*.md` at runtime via `load_knowledge()`.
- `src/v2/` (hybrid benchmark) reads `data/source/pytorch_docs/*.jsonl`, chunked and cached via `src/v2/corpus/build.py` into `data/output/cache/pytorch_corpus.jsonl` + `pytorch_symbol_index.json`.
- The `hybrid-hybrid-*` and `hybrid_with_base_model-hybrid_with_base_model-*` benchmark files are all v2; the `.md` files had no effect on any benchmark we have been comparing against. I audited and edited 5 `.md` files (3 new: `dropout_basics`, `batchnorm_basics`, `argmax_dim_semantics`; 2 revised: `train_vs_eval`, `eval_vs_no_grad`; plus README). Those edits are retained as-is for v1 but are not the experiment path.
- The real experiment path was to edit `data/source/pytorch_docs/nn_module_modes.jsonl` (rows `api_docs_module_train` and `api_docs_module_eval`), rebuild the corpus, and rerun the benchmark on the base model with `MODE_HYBRID_WITH_BASE_MODEL`.

### First source-level cleanup: structural separation

Edited both `api_docs_module_train` and `api_docs_module_eval` to:

- Describe dropout and BN mechanisms in separate sentences (no more "dropout does X, and BN does Y with running statistics" single-clause structure).
- Add explicit "two independent mechanisms that share the same switch" framing.
- Point at `torch.nn.Dropout` and `torch.nn.BatchNorm2d` for individual mechanism.
- In the `eval` doc, keep "running mean / running variance" language but only inside the BN-specific sentence (154 chars away from any mention of `dropout`).

Rebuilt corpus: `python -m src.v2.corpus.build` → 41 chunks, 88 symbol-index keys, all verified.

### Result: benchmark 121402, base model + cleaned source

Scorer: 26 / 10 / 0 = **0.722**. Run file: `experiments/eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-22-121402.json`.

Post-rescore diff against baseline (hand-regraded 27/7/2 = 0.750):

| Item | Baseline | New cleanup | Diagnosis |
|------|----------|-------------|-----------|
| `nn_module_modes_001` | correct | partial | **scorer FN** — new answer uses `nn.Dropout` / `nn.BatchNorm`; benchmark `must_include: ["batch normalization"]` does literal substring match and misses the `nn.BatchNorm` form. Content is correct. |
| `nn_module_modes_002` | correct | partial | **scorer FN** — my rewrite replaced the literal phrase "changes module behavior" with "changes the behavior of stateful layers that read the training flag". Benchmark `must_include: ["module behavior"]` no longer matches. Content is correct. |
| `nn_module_modes_003` | correct | partial (hand-regrade: incorrect) | **real regression, the target FP reappeared**. New answer: "dropout layers in the model will switch to using their stored running statistics for normalization, instead of the current batch statistics." Exactly the dropout-has-running-statistics confabulation the cleanup was supposed to prevent. |

Hand-regrade:
- `nn_module_modes_001`: partial → correct (phrasing only)
- `nn_module_modes_002`: partial → correct (phrasing only)
- `nn_module_modes_003`: partial → incorrect (content wrong)

**Hand-regrade score: 28 / 7 / 1 = 0.778.** Marginal net improvement over baseline's 0.750, but the most important target FP got worse, not better.

### Why the target FP got worse — subject-swap on parallel structure

Original `api_docs_module_eval` had:

> dropout layers are disabled and pass all inputs through without zeroing any elements, **and** batch normalization layers use their stored running statistics

The dropout clause and the BN clause were joined but structurally *asymmetric* (one talks about disabling, the other about running statistics). The baseline and both ds6 adapters all learned to reproduce the dropout clause verbatim.

My rewrite replaced that with two structurally *parallel* sentences:

> nn.Dropout in eval mode becomes an identity function: all input elements pass through unchanged, with no zeroing and no scaling.
> nn.BatchNorm in eval mode switches to using its stored running statistics ... for normalization, instead of the current batch statistics.

Two sentences, same template ("X in eval mode [verb] ... : ..."). When the retriever returned this chunk and the query asked about dropout in eval mode, the base model subject-swapped "nn.BatchNorm" with "dropout" in the BN sentence — producing a fluent but completely false statement about dropout using running statistics for normalization. The new answer is almost verbatim the BN sentence with the subject replaced.

**Takeaway: "separate the two mechanisms" is not enough. The rewrite must also break the structural symmetry that enables subject-swap.**

### Cross-run target-FP matrix (all four runs)

| Item | baseline | ds6 1-ep | ds6 3-ep | new cleanup |
|------|----------|----------|----------|-------------|
| `nn_module_modes_003` (dropout-FP target) | correct | correct | correct | **incorrect** (hand) |
| `debugging_004` (argmax-FP target) | correct | incorrect | correct | correct |
| `shape_ops_002` | correct | correct | incorrect | correct |
| `shape_ops_004` | correct | correct | incorrect | correct |
| `autograd_003` | correct | correct | partial | correct |
| `data_loading_003` | partial | partial | incorrect | partial |

The new cleanup restored everything that 3-epoch ds6 broke except for the dropout FP, which it made worse. argmax is fine (never was the retrieval problem; source `api_docs_torch_argmax` was already correct).

### Follow-up edits (this session)

1. **Rewrite `api_docs_module_train` and `api_docs_module_eval` again** to break the parallel "X in eval mode [verb]" template. Dropout sentence becomes short with a simple predicate; BN sentence stays longer and uses a different verb structure. Also add an explicit defensive sentence: "Dropout does not maintain running statistics of any kind; the running-statistics branch is specific to batch normalization."
2. **Tighten `nn_module_modes_003` scorer rules** (`data/eval/benchmark_core_pytorch.jsonl`). Added 5 new hard `must_not_include` literals covering the observed FP family (`"switch to using their stored running statistics"`, `"switch to using running statistics"`, `"switch to using the running statistics"`, `"will switch to using their stored running statistics"`, `"dropout layers in the model will switch to using"`) — hard hits force `incorrect`. Replaced the single narrow regex with two broader ones: (a) `dropout[^.]{0,100}(uses?|using|switch\w+|rely\s+on|relies\s+on)[^.]{0,40}running\s+(statistics|stats|mean|variance)` to catch dropout + running-stats confabulation within a single sentence regardless of intervening words; (b) `dropout[^.]{0,100}normali[sz]e[sd]?[^.]{0,40}running\s+(statistics|stats|mean|variance)` to catch the "dropout ... normalized ... running ..." variant. Regex hits remain soft (demote to `partial`) per scorer semantics. Verified: the observed FP answer hits 3 hard literals and 1 regex; a correct-style "dropout is disabled" answer hits none. Total row count unchanged (36), no duplicate ids.

### Second-pass result: benchmark 125849

Scorer: 27 / 8 / 1 = **0.750**. Run file: `experiments/eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-22-125849.json`.

Items that moved 121402 → 125849:

| Item | 121402 | 125849 | Diagnosis |
|---|---|---|---|
| `nn_module_modes_003` | partial (hand: incorrect) | **correct** | Target FP genuinely fixed. Answer: `"dropout layers are disabled ... passes all inputs through unchanged"`. Zero confabulation. The structural rewrite (non-parallel sentences + defensive negation sentence) is what did the work. |
| `nn_module_modes_001` | partial | incorrect | **Not a real regress.** 121402 answer contained `"switch to using their stored statistics for normalization"` — a confabulation the scorer missed and counted as partial (via `must_include: dropout` match). 125849 answer no longer confabulates at all — it's correct-but-vague, just no longer mentions `nn.Dropout` or `nn.BatchNorm`, so both `must_include` entries miss. Content improved; scorer label worsened. |

FP/FN audit of 125849 surfaced 5 pre-existing partial-labeled items whose answers are actually incorrect, untouched by this cleanup: `autograd_001` (off-topic), `nn_module_modes_002` (says eval disables gradient tracking), `nn_module_modes_004` (says "training mode sets the module to evaluation mode"), `optim_training_loop_004` (says "update once per epoch"), `data_loading_003` (says `pin_memory` pins GPU memory). These are scorer blind spots, not caused by the source rewrite.

### Third-pass source edit: named-layer appositive

To recover the concrete `nn.Dropout` / `nn.BatchNorm` mention that 125849 lost on `nn_module_modes_001`, insert a named-layer appositive into both mode sentences without re-introducing the `"in X mode, A does a, B does b"` parallel predicate template.

- `api_docs_module_train`: `"Only a few stateful layers read the training flag."` → `"Only a few stateful layers read the training flag — nn.Dropout and nn.BatchNorm are the main ones, and each interprets it in its own independent way."`
- `api_docs_module_eval`: `"Evaluation mode is a flag on the module that a few stateful layers read individually."` → `"Evaluation mode is a flag on the module that a few stateful layers — nn.Dropout and nn.BatchNorm are the main ones — read individually, each in its own way."`

Structure: appositive lists both layers as a group; predicates are shared (`"read individually"`, `"each interprets it in its own way"`) not parallel-split, so subject-swap incentive is absent. Defensive negation sentence (`"Dropout does not maintain running statistics of any kind ..."`) left in place.

### Result: benchmark 131502

Scorer: 28 / 7 / 1 = **0.778**. Run file: `experiments/eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-22-131502.json`. `nn_module_modes` category up from 0.25 → 0.50.

Items that moved 125849 → 131502:

| Item | 125849 | 131502 | Diagnosis |
|---|---|---|---|
| `nn_module_modes_004` | partial (hand: incorrect) | **correct** | Unexpected bonus win. Previous answer had the nonsense opener `"training mode sets the module to evaluation mode"`. New answer: `"In train mode, it computes the mean and variance from the current mini-batch, while in eval mode, it uses the previously accumulated running statistics."` Clean BN-only description. The restructured `api_docs_module_train` — particularly `"For nn.BatchNorm, training mode has a different effect entirely: the layer normalizes using statistics computed from the current mini-batch ..."` — gave the generator a cleaner template than the previous intermixed version. |

`nn_module_modes_003` stayed correct across 125849 → 131502 (stable fix). `nn_module_modes_001` stayed incorrect — the appositive did not flip it; the base model's generation path for `"difference between X and Y"` questions resolves to an abstract property enumeration without naming layers. Fixing this is likely SFT territory (direct-negation or difference-style rows), not source-content territory.

### Full trajectory table

| Run | Scorer | `nn_module_modes` category | Note |
|---|---|---|---|
| baseline (rescored 171427) | 29 / 7 / 0 = **0.806** | — | base model + untouched v2 source |
| 121402 (first cleanup) | 26 / 10 / 0 = 0.722 | 0 / 4 | parallel template introduced subject-swap |
| 125849 (template broken) | 27 / 8 / 1 = 0.750 | 1 / 4 | target FP fixed; 001 lost its layer mentions |
| **131502 (+ appositive)** | **28 / 7 / 1 = 0.778** | **2 / 4** | 004 recovered as bonus; 001 still stuck |

Residual gap to baseline: 1 item (`nn_module_modes_001`). The three long-standing FPs inside `partial` (autograd_001, nn_module_modes_002, optim_training_loop_004, data_loading_003) are orthogonal to this cleanup — they need scorer-rule tightening, not source-content fixes.

### Fourth-pass source edit: re-order `api_docs_module_eval`

Target: `nn_module_modes_002` (`"Why is model.eval() not the same as torch.no_grad()?"`). The fact `"model.eval() does not disable gradient tracking"` is already in `api_docs_module_eval` but sits near the end of the chunk; the base model's generation consistently picks the early `"Module.eval() sets the module to evaluation mode"` framing and never reaches the distinction. Move the eval-vs-no_grad distinction into the second sentence slot so it is inside the generator's high-probability window.

### Result: benchmark 132507 — net regression, reverted

Scorer: 27 / 8 / 1 = **0.750** (down from 0.778). Run file: `experiments/eval_results/benchmark/hybrid_with_base_model-hybrid_with_base_model-Qwen-Qwen2.5-1.5B-core-2026-04-22-132507.json`.

Per-item effect of the reorder:

| Item | 131502 | 132507 | Effect |
|---|---|---|---|
| `nn_module_modes_002` (target) | partial (content: confabulates on "both disable gradient tracking") | partial (**content worse**) | The generator produced a self-contradictory answer `"This mode disables gradient tracking and reduces memory usage, but it does not affect autograd or gradient tracking"` and also confused `torch.no_grad()` with `detach()` (`"creates a new tensor object that is separated from autograd tracking while keeping the data"`). Front-loading four consecutive `not`/`does not` clauses overloads the small base model's negation-binding: it copies the surface tokens but flips the scope. |
| `nn_module_modes_004` | correct | partial | Generator wandered into tangential buffer storage material (`running_mean`/`running_var buffers`, `"inaccurate running statistics"`) and stopped using `nn.BatchNorm` / `Module.train` / `Module.eval` symbols. `expected_symbols` 0/3. |
| `nn_module_modes_001` | incorrect | incorrect | Label unchanged. Content shifted away from the training-flag loop and now touches the autograd distinction (`"does not affect autograd or gradient tracking"`), which is actually closer to the item's gold content, but still no `dropout` / `batch normalization` substring, so `must_include` still misses. |
| `nn_module_modes_003` | correct | correct | Stable. Shorter and cleaner than 131502, zero confabulation. |

Root cause: the eval chunk after the reorder front-loaded four stacked negations (`"is not the same"`, `"is not a replacement"`, `"does not disable gradient tracking"`, `"does not reduce memory usage"`). This increases the density of the *exact phrases* the target FP uses, which helps a large model but hurts a 1.5B base model, whose local copy-paraphrase pass mis-binds the `not` scopes and emits affirmative versions of the same phrases. Signal density can be net-negative below a model-size threshold.

Reverted `api_docs_module_eval` to the 131502 form: eval-vs-no_grad distinction back at ~71% of the text (after the layer-mechanics body), named-layer appositive and defensive negation sentences preserved. Corpus rebuilt (41 chunks).

**Decision**: source-content work is ceilinged at ~0.778 for this model size. The remaining `nn_module_modes_002` FP is a base-model semantic-association problem (model.eval ↔ torch.no_grad are too tightly bound in pretraining data) that needs SFT-level intervention: add direct-negation rows to a `dataset_7` pool, e.g. `Q: "Does model.eval() disable gradient tracking?"` → `A: "No, model.eval() only flips the training flag. ..."`.

### Final trajectory

| Run | Scorer | `nn_module_modes` | Note |
|---|---|---|---|
| baseline (171427, rescored) | 29 / 7 / 0 = 0.806 | — | base model + untouched v2 source |
| 121402 (first cleanup, parallel template) | 26 / 10 / 0 = 0.722 | 0 / 4 | subject-swap confabulation introduced |
| 125849 (parallel broken + defensive negation) | 27 / 8 / 1 = 0.750 | 1 / 4 | target FP fixed |
| 131502 (+ named-layer appositive) | 28 / 7 / 1 = **0.778** | 2 / 4 | bonus win on 004; **kept as final source** |
| 132507 (eval reorder) | 27 / 8 / 1 = 0.750 | 1 / 4 | negation pile-up hurt small model; **reverted** |

Source ceiling for 1.5B base model + hybrid retrieval: 0.778 (1 item below baseline 0.806). The missing item is `nn_module_modes_001`, a vagueness issue that source-content tweaks cannot fix reliably. Next lever is training-side (`dataset_7` direct-negation pool), not source-side.

## Other Important Info

- Same `MODEL_NAME / BATCH_SIZE / LEARNING_RATE / seed=42` as ds4 / ds5 — checkpoints are discriminated only by `DATASET_NAME`. `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt` now exists; ds4 and ds5 checkpoints untouched.
- Training run was blocked for ~40 minutes on swap exhaustion (host RAM + VS Code Pylance extension host + other long-running processes pushed Mac into SSD thrashing). Fixed by restart + running with only terminal open. Documented in CLAUDE.md pre-train check flow. Not a model / data / config issue.
- Rescoring of ds4 / ds5 benchmark JSONs was done in-session via `scripts/rescore_minimal.py`; original files on disk still carry their pre-patch labels. `experiments/eval_results/benchmark/rescored/` only contains the baseline (`171427`) and two earlier runs — ds4 / ds5 rescored numbers in the table above are recomputed inline each time a comparison is needed.
- Key file deltas from this log:
  - Added: `scripts/build_dataset_6.py`
  - Added: `data/raw/training/dataset_6.jsonl`, `data/raw/validation/dataset_6.jsonl`
  - Added: `data/raw/training/dataset_6_addendum.jsonl`
  - Added: `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt`
  - Added: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-104632.json` (1-epoch ds6 benchmark)
  - Added: `experiments/eval_results/benchmark/hybrid-hybrid-Qwen-Qwen2.5-1.5B-core-2026-04-22-113856.json` (3-epoch ds6 benchmark)
  - Modified (by resume run): `data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt` — now the 3-epoch checkpoint. The 1-epoch version is preserved at `Qwen-Qwen2.5-1.5B-dataset_6-8-0.0001.pt.1epoch.bak` (if the pre-resume backup was taken).
