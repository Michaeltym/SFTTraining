# PyTorch API Assistant — v2

A learning project exploring end-to-end SFT + hybrid RAG on top of a
small open-source base model, scoped to PyTorch API question answering.

**Status (2026-04-22):** v2 reached its practical ceiling. The SFT recipe
plateaued, benchmark audits showed most of the SFT "gain" is trained-domain
only, and we decided to stop before turning it into a real product would
require either a much larger base model, a much larger curated corpus, or
both.

This README covers the v2 arc: why we did it, what we built, what went
wrong, what the numbers said, and why we stopped here.

---

## TL;DR

| | Best hand-regraded accuracy | Notes |
|---|---|---|
| v1: `Qwen2.5-0.5B + LoRA + top_k=1 lexical` | ~0.47 (18/36) | structural ceiling, motivated v2 |
| v2 baseline: `Qwen2.5-1.5B base + hybrid RAG` | 0.778 (36-item) / 0.717 (60-item) | simplest thing that works |
| v2 best SFT: `ds8 2-ep + hybrid RAG` | 0.833 (36-item) / 0.733 (60-item) | ceiling of the recipe |
| v2 target | `≥ 0.90` | not reached |

Headline finding: on the original 36 items (trained domain) SFT gives a
real +0.056 lift. On the 24 newer items (unseen phrasings in the same
task family) SFT actually **loses** to the base model (0.583 vs 0.625
hand-regraded). The net scorer delta of +0.084 is ~80% scorer artifact
and ~20% real.

---

## Why v2

v1 was a small exploratory prototype:

- `Qwen2.5-0.5B` base
- LoRA SFT on a tiny hand-written dataset
- `top_k = 1` lexical retrieval against a handful of markdown chunks
- no formal benchmark — "it feels smarter" was the only success criterion

That combination had a very low ceiling. On the first real benchmark the
v1 stack hit **0.472** and stopped responding to further data/prompt
tweaks. The failure mode was structural, not fixable by more rows:

- the 0.5B answerer got behavior and shape reasoning wrong even when
  the right doc was retrieved
- `top_k = 1` lexical retrieval missed comparison-style questions
  (e.g. `view` vs `reshape`) because the two candidates never ended
  up in context together
- the hand-written knowledge base had huge gaps and no symbol index
- "correct" was defined by eyeballing — no per-category signal, no
  refusal metric, no citation metric

So v2 was written as a deliberate re-architecture, not an incremental
patch. The goal was stated explicitly in `plan-v2.md`:

> Success is not "it feels smarter"; success is "it reaches at least
> 90% acceptable-answer rate on a fixed core PyTorch benchmark."

Scope was narrowed to `Core PyTorch dev QA`:
`tensor / shape / dtype / device / autograd / nn / optim / data /
training-vs-eval / debugging`, with a separate hallucination-refusal
slice for fake-API questions.

---

## What changed in v2

v2 is organised as five layers, each implemented as an isolated module
under `src/v2/`:

```
src/v2/
├── corpus/       # source docs → structured chunks → symbol index
├── retrieval/    # symbol match + BM25 lexical + hybrid merge
├── prompts/      # exact / reasoning / refusal templates + router
├── benchmark/    # data loader, scorer, per-category summary, saver
├── answer_fns.py # pluggable answerer (base / SFT / hybrid)
└── main.py       # entry point
```

Concretely, v2 added or replaced the following:

**Corpus layer.** Markdown source files under
`data/source/pytorch_docs/` are chunked into
`data/output/cache/pytorch_corpus.jsonl`, and a symbol index is written
to `data/output/cache/pytorch_symbol_index.json`. Chunks are curated by
failure slice (`tensor_creation`, `shape_ops`, `dtype_device`,
`autograd`, `nn_module_modes`, `debugging`, etc.) with fake-API
refusal rules as a dedicated slice.

**Retrieval layer.** Three retrievers that run in parallel and merge:

- `symbol` — exact PyTorch symbol match on the query against the
  symbol index (e.g. `torch.cat`, `Tensor.view`, `nn.BatchNorm2d`)
- `lexical` — BM25-style token scoring, with title/tag/text weights
  configurable in `src/config.py` (`RAG_RETRIEVAL_*_WEIGHT`,
  `_STEP`, `_BASE` constants)
- `hybrid` — merge of the above two, capped at `top_k=3`, with a
  comparison-question bonus so questions like "view vs reshape" pull
  both chunks into context

**Prompt layer with routing.** Three prompt templates plus a keyword
router:

- `exact.py` — short, grounded, single-symbol answers (used for
  comparison and share-memory style questions where the model tends
  to hallucinate if given more freedom)
- `reasoning.py` — longer reasoning, used for debugging and
  shape-analysis questions
- `refusal.py` — hallucination-resistance template for fake-API
  queries
- `router.py` — keyword-based route (`EXACT_PROMPT_QUERY_KEYWORDS`
  drives which template a question lands in)

**Benchmark + scorer.** A formal benchmark at
`data/eval/benchmark_core_pytorch.jsonl` with strict item schema:

```python
class BenchmarkItem(TypedDict):
    id: str
    question: str
    category: BenchmarkCategory
    gold_type: BenchmarkGoldType
    expected_symbols: list[str]
    must_include: list[str]
    must_include_any_of: NotRequired[list[list[str]]]
    must_not_include: list[str]
    must_not_include_regex: NotRequired[list[str]]
    requires_citation: bool
    difficulty: BenchmarkDifficulty
```

Labels are a fixed three-way scale `correct / partially_correct /
incorrect`. The summary reports overall accuracy, per-category
accuracy, citation support rate, and hallucination refusal accuracy.

**Answer-fn plug-in.** `answer_fns.py` lets the same retrieval + scorer
stack run against different answerers:

- `base` — no SFT, just Qwen2.5-1.5B + hybrid context
- `sft` — LoRA adapter on top of the base model
- `hybrid_with_base_model` — explicit base + hybrid mode added so we
  can measure hybrid RAG without any SFT dependency

**Data pipeline for SFT.** `data/raw/training/dataset_{1..10}.jsonl`
and matching validation files, each with a changelog in
`experiments/logs/sft-dataset_*`. Ten dataset iterations over the
course of v2, ranging from 24-row pilots to 200-row scale-ups,
targeting specific benchmark failure slices after each audit.

**Operational tooling.**

- `scripts/expand_benchmark.py` — schema-validated benchmark expander;
  one-shot appender + softener with a strict validator
- `scripts/rescore_benchmark.py` — re-score historical result JSONs
  under the current scorer, with before/after deltas
- `scripts/_rescore_shim.py` — Python 3.10 `typing.NotRequired`
  compat shim for the sandboxed run environment
- `src/training.py` — LoRA SFT loop with a
  `torch.mps.empty_cache()` fix that was required to stop
  M1 Pro step times from drifting `8s → 45s → 444s`

**Dropped from the original v2 plan.** Two items from `plan-v2.md`
never landed:

- dense retrieval / reranker. Lexical + symbol was strong enough that
  we never needed it, but the coverage tail for paraphrased questions
  is still lexical-blind.
- full official PyTorch corpus ingestion. We stayed on a curated
  hand-written knowledge base (~20 chunks). The plan was to pull
  official docs/tutorials/recipes; that slice of work was deferred.

---

## What we learned while building v2

### 1. The benchmark scorer is a dashboard, not ground truth

The keyword-based scorer (`must_include` / `must_not_include` /
`must_include_any_of` / `must_not_include_regex`) has two intrinsic
failure modes:

- **false positives** — a wrong answer that happens to contain the
  right keywords. `cat` vs `stack` inversions got labeled `correct`
  because the answer mentioned both symbols.
- **false negatives** — a correct answer phrased with a valid synonym
  the rule didn't list. "detached from the computation graph" was
  labeled `partially_correct` because the rule only listed
  "shares storage".

CLAUDE.md now requires a manual FP/FN audit after every benchmark run.
On the 60-item run we found 3 FPs and 4 FNs in the ds8 2-ep result,
and 2 FPs and 5 FNs in the baseline result. Without the audit the
headline `+0.084` scorer delta would have been mis-reported as a real
SFT win; under hand regrading the delta collapses to `+0.017`.

### 2. 2-epoch SFT at 1.5B produces confident factual inversions

The single most consistent failure signature of over-applied SFT on
this base model is **confidently wrong** answers on unseen items —
the model has picked up the benchmark's answer shape but lost a real
semantic fact. Hand audits on the new-24 items caught:

- `shape_ops_007`: "cat is stricter than stack" — inverted (stack is
  stricter because it requires identical shapes)
- `nn_module_modes_008`: "Dropout is active during inference" — inverted
- `autograd_006`: `requires_grad=True` credited for *disabling* autograd

All three are cases where the base model hedges and gets `P`, but the
SFT model commits and gets labeled `C` despite being wrong. The scorer
doesn't catch this unless a specific `must_not_include` rule exists.

### 3. "More data" stops helping early on 1.5B + LoRA

Dataset sizes went `24 → 50 → 50 → 50 → 50 → 50 → 50 → 200` across
ds3 through ds10. The best result was ds8 (50 rows, 2 epochs, targeted
recovery rows after the ds6/ds7 audits). ds10 scaled training from
50 to 200 rows on the same recipe and **regressed** on the benchmark.
Data quantity was not the bottleneck; the recipe itself was.

### 4. Knowledge chunks cross-contaminate when written as tutorials

Early knowledge files like `train_vs_eval.md` mixed BN, Dropout,
`model.eval()`, and `no_grad()` in one chunk. Retrieval would pull the
chunk correctly but the answerer would fuse two nearby-but-different
APIs into one paragraph. Splitting into single-purpose chunks
(`batchnorm_basics.md`, `dropout_basics.md`, `eval_vs_no_grad.md`)
fixed several confident-but-wrong answers. CLAUDE.md now has explicit
rules about keeping chunks narrow and single-purpose.

### 5. Small operational landmines

- **MPS step-time drift on M1 Pro**: without
  `torch.mps.empty_cache()` after `optimizer.step()`, step times
  grew exponentially — one early run went `8s → 53s → 497s` before
  being killed. Fix is a one-liner; finding it took a morning.
- **Python version drift**: `typing.NotRequired` is 3.11+; the
  sandbox is 3.10, the project is 3.12. The shim script
  `scripts/_rescore_shim.py` patches `typing` before importing the
  real entry point.
- **Refusal prompt interference**: an early attempt to add
  refusal-template routing degraded several normal questions because
  the trigger rules were too broad. Keeping the refusal trigger
  strictly on fake-API symbols (and nothing else) was the fix.
- **Retrieval is not the latency bottleneck**: generation dominates;
  retrieval is under 5% of per-question wall time on MPS. Early
  optimisation on retrieval speed was wasted effort.

---

## Results

All numbers below are on the same base model (`Qwen2.5-1.5B`) and the
same hybrid retrieval stack, varying only the answerer (base vs LoRA
checkpoint) and the scorer/benchmark version. Each run is paired with
its result JSON in `experiments/eval_results/benchmark/`.

### 36-item core benchmark (v1 scorer, historical)

| Run | C/P/I | acc | notes |
|---|---|---|---|
| v1: 0.5B + SFT + top_k=1 lexical | 17/7/12 | 0.472 | starting point; v1 ceiling |
| v2: 0.5B + SFT + hybrid | 17/7/12 | 0.472 | retrieval alone didn't help 0.5B |
| v2: 1.5B base + hybrid | 21/5/10 | 0.583 | +0.11 from model capacity only |

### 36-item core benchmark (post-polish scorer, 2026-04-21)

After corpus expansion, three scorer `any_of` migrations, and a
router fix for the `from_numpy` comparison case:

| Run | C/P/I | acc |
|---|---|---|
| 1.5B base + hybrid (mainline) | 31/5/0 | **0.861** |
| ds3 pilot SFT + hybrid (1 ep, 50 rows) | 32/4/0 | 0.889 |
| post-fix base + hybrid (after FP patches) | 30/6/0 | 0.833 raw, 29/7/0 FP-adjusted |
| ds3 pilot rescored under tightened scorer | 27/9/0 | 0.750 |

### 60-item core benchmark (final, 2026-04-22)

The benchmark was expanded from 36 → 60 items: 4 softens on chronic
synonym-rigidity rows, plus 24 new items targeting wobbly categories
(`shape_ops`, `nn_module_modes`, `optim_training_loop`, `autograd`,
`debugging`). This is the grading used for the final stop decision.

| Run | C/P/I | scorer acc | hand acc |
|---|---|---|---|
| baseline: 1.5B base + hybrid | 38/16/6 | 0.633 | **0.717** |
| ds8 2-ep + hybrid (ceiling SFT) | 43/11/6 | **0.717** | **0.733** |
| scorer Δ | | +0.084 | |
| hand Δ | | | +0.017 |

Split by subset, hand-regraded:

| Subset | baseline | ds8 2-ep | Δ |
|---|---|---|---|
| Old 36 (trained domain) | 0.778 | 0.833 | +0.056 |
| New 24 (unseen phrasings) | 0.625 | 0.583 | **-0.042** |

### Dataset iteration ROI

Ten SFT dataset iterations (ds3 → ds10), none of which beat ds8 2-ep:

| Dataset | Size | Epochs | Best hand acc (36-item) |
|---|---|---|---|
| ds3 pilot | 50 | 1 | 0.833 |
| ds4, ds5 | 50 | 1 | regressed on target FPs, documented FPs in log |
| ds6, ds6+add | 50 | 1 / 3 | 3-ep over-trained, 1-ep marginal |
| ds7 | 50 | 1 | parity with ds6 |
| **ds8** | **50** | **2** | **0.833 ← best** |
| ds9, ds9+add | 50 | 1 / 2 | recovery attempt; did not beat ds8 |
| ds10, ds10+add | 200 | 1 / 2 | regressed vs ds8 — scale-up hurt |

The full audit for each run is in `experiments/logs/sft-dataset_*`.

---

## The good and the bad

### What worked well

- **Five-layer separation.** The `corpus / retrieval / prompts /
  benchmark / answer_fns` split held up through all 10 dataset
  iterations without needing a refactor. Changing one layer never
  required touching another.
- **Formal benchmark + forced FP/FN audit.** Catching that the
  `+0.084` scorer delta was mostly artifact is only possible because
  the audit discipline was baked into the workflow. Several
  published claims about SFT gains in small-model experiments would
  not survive the same audit.
- **Hybrid retrieval.** Symbol + lexical merge handled comparison
  and single-symbol questions cleanly without needing dense
  embeddings or a reranker. This was the best cost/value trade in v2.
- **Rescore + softener tooling.** Being able to re-grade any
  historical run under a new scorer without re-running inference
  (`scripts/rescore_benchmark.py`) made scorer iteration cheap.
  Without it we would have either frozen the scorer early (bad) or
  spent hours re-running benchmarks (expensive).
- **Capacity upgrade paid for itself.** Moving 0.5B → 1.5B gave a
  `+0.11` lift that no amount of SFT-on-0.5B recovered. The
  structural decision was correct.

### What didn't work or is still weak

- **Never hit the 90% target.** Best hand-regraded score is 0.833
  on the original 36, 0.733 on the 60-item expansion. The gap to
  0.90 is not bridgable by incremental patches on this recipe.
- **SFT generalisation is poor.** On items written after the SFT
  dataset was built, the SFT model actually loses to the base model
  (0.583 vs 0.625 hand-regraded on the new 24). The ds8 recipe
  memorises the benchmark's answer shape.
- **Confident inversion risk scales with 2 epochs.** ds10 2-ep
  produced 3 confident factual inversions on the new 24 that
  baseline doesn't produce. Running more epochs is a real downside,
  not just a diminishing return.
- **Knowledge base is tiny.** ~20 curated markdown chunks.
  `plan-v2.md`'s official-corpus ingestion never happened, so the
  retrieval side hasn't been stress-tested at scale.
- **Scorer has an irreducible FP/FN tail.** Every benchmark run
  surfaces new phrasings the keyword rules don't capture. A
  scorer-v2 patch list is outstanding (see the end of
  `experiments/logs/sft-dataset_10-Qwen-Qwen2.5-1.5B-2026-04-22.md`).
- **No real abstention layer.** The model is willing to answer
  confidently even when retrieval returns weak chunks. A proper
  "refuse when unsure" policy would prevent most of the new-24 FPs.

---

## Why we stopped here

Three reasons, in order of weight:

**1. The recipe has plateaued.** ds8 2-ep is the ceiling.
ds9 and ds10 did not beat it, scaling to 200 rows regressed, and the
60-item hand-regrade showed the remaining `+0.017` delta over baseline
is within noise and concentrated on trained-domain items. Another
dataset iteration has ~0 expected ROI.

**2. SFT is the wrong tool for the remaining gap.** The outstanding
failures on this stack break down into:

- knowledge gaps (retrieval returns a weak chunk because the fact
  isn't in the corpus) — this is a corpus problem, not a weights
  problem
- phrasing rigidity in the scorer — this is a benchmark problem, not
  a model problem
- confident hallucination on out-of-distribution queries — this
  wants a proper abstention policy, which is a prompt + routing
  problem

None of these is what LoRA SFT on 50 rows fixes. Continuing to build
more datasets would be optimising the wrong variable.

**3. The learning goal is met.** CLAUDE.md framed this repo as a
learning project to understand the full SFT post-training workflow.
Over v2 we ran 10 dataset iterations, debugged MPS perf issues, built
a benchmark + scorer + audit loop, and caught a real over-fit signal
on unseen items. That is the end-to-end picture we came for.

The sensible next moves — product-oriented or research-oriented —
are all outside the current recipe:

- **product pivot:** narrow the scope to a single high-leverage
  scenario (e.g. "PyTorch shape/device error fixer"), drop SFT for
  v1, and invest the effort in UX + curated knowledge + strict
  abstention. Estimate: 3–4 weeks to a deployable MVP.
- **research pivot:** keep the same eval loop but test it against a
  larger base model (3B or 7B). The question becomes "does the
  confident-inversion signature persist at 3B?" rather than "can
  another 50 rows save 1.5B?".
- **corpus pivot:** ingest the official PyTorch docs/tutorials as
  originally scoped in `plan-v2.md`, and re-measure base + hybrid.
  If base + a bigger corpus reaches 0.90, the whole SFT branch was
  avoidable for this task.

This repo is left in a committed, reproducible state as a reference
point for any of those pivots.

---

## Repo layout at stop

```
.
├── CLAUDE.md                          # project rules + dataset/scorer conventions
├── plan-v2.md                         # the original v2 architecture plan
├── requirements.txt
├── data/
│   ├── eval/
│   │   ├── benchmark_core_pytorch.jsonl       # 60 items (final)
│   │   └── benchmark_core_pytorch_v1.jsonl    # 36 items (backup)
│   ├── knowledge/pytorch_docs/*.md            # curated chunks (~20)
│   ├── source/pytorch_docs/*.jsonl            # source material for the corpus builder
│   ├── output/cache/
│   │   ├── pytorch_corpus.jsonl               # built corpus
│   │   └── pytorch_symbol_index.json          # symbol index
│   ├── raw/training/dataset_{1..10}.jsonl
│   ├── raw/validation/dataset_{1..10}.jsonl
│   ├── checkpoints/                           # LoRA adapter checkpoints
│   └── adapters/
├── src/
│   ├── config.py                      # central config: model, paths, LoRA, RAG weights
│   ├── training.py                    # LoRA SFT loop with MPS cache fix
│   ├── baseline.py / evaluate.py / resume.py / runtime.py
│   ├── dataset.py / dataloader.py / data.py / tokenizer.py / model.py / adapter.py
│   ├── rag/                           # v1 RAG (kept for baseline)
│   └── v2/
│       ├── corpus/build.py + types.py
│       ├── retrieval/{hybrid,lexical,symbol,load,types}.py
│       ├── prompts/{exact,reasoning,refusal,router,types}.py
│       ├── benchmark/{data,label,run,save,summary,types}.py
│       ├── answer_fns.py
│       └── main.py
├── scripts/
│   ├── expand_benchmark.py            # schema-validated benchmark expander
│   ├── rescore_benchmark.py           # re-grade historical runs
│   └── _rescore_shim.py               # Py 3.10 typing compat
└── experiments/
    ├── logs/                          # per-run markdown logs with FP/FN audit
    └── eval_results/benchmark/        # all benchmark result JSONs
```

---

## Reproducing key runs

Config is centralised in `src/config.py`. The two switches that matter
for a benchmark run are `MODE` (which code path `main.py` takes) and
the LoRA `CHECKPOINT_PATH` (derived from `MODEL_NAME`, `DATASET_NAME`,
`BATCH_SIZE`, `LEARNING_RATE`).

**Baseline (no SFT, hybrid RAG):**
```
# src/config.py
MODE = MODE_HYBRID_WITH_BASE_MODEL
python -m src.main
```

**Best SFT (ds8 2-ep, hybrid RAG):**
```
# src/config.py
DATASET_NAME = "dataset_8"
MODE = MODE_HYBRID
# expects data/checkpoints/Qwen-Qwen2.5-1.5B-dataset_8-8-0.0001.pt
python -m src.main
```

**Re-score an older result under the current scorer:**
```
python scripts/rescore_benchmark.py \
  experiments/eval_results/benchmark/<run>.json
```

**Expand the benchmark:**
```
python scripts/expand_benchmark.py
# validates schema, backs up v1, writes 60-item JSONL
```

All runs append a result JSON under
`experiments/eval_results/benchmark/` with naming convention
`<answer_fn>-<answer_fn>-<model_name>-core-<timestamp>.json`, and the
matching markdown log lives under `experiments/logs/`.

---

## Related docs in the repo

- `CLAUDE.md` — hard rules on model usage, dataset conventions,
  scorer FP/FN audit discipline, log format
- `plan-v2.md` — original v2 architecture plan (what was promised,
  what landed, what was deferred — see the "Dropped from v2 plan"
  section above for the gap)
- `experiments/logs/hybrid-v2-*.md` — retrieval + prompt iteration
  logs (corpus expansion, router fixes, single-variable sessions)
- `experiments/logs/sft-dataset_*.md` — per-dataset SFT logs, each
  including run config, findings, FP/FN audit, and decision

If you are picking this up later, start with `plan-v2.md` (what we
set out to do), then this README (what actually happened), then
`experiments/logs/sft-dataset_10-*.md` (the last run, which also
records the stop decision and the outstanding scorer-v2 patch list).
