# AGENTS.md

## Purpose

This repository is for learning and experimenting with supervised fine-tuning (SFT) on top of pretrained language models.

The goal is to understand the full post-training workflow end to end:

- loading a pretrained base model
- loading the matching tokenizer
- preparing instruction-style datasets
- running supervised fine-tuning
- comparing base vs SFT behavior
- saving checkpoints and experiment logs

This repository is intentionally separate from the from-scratch toy LLM project.

## Code Review Standard

When reviewing code, always focus on best practices, especially:

- correctness
- error handling
- reproducibility
- training stability
- dataset quality and formatting
- checkpoint safety
- avoiding accidental base/instruct model mix-ups

## Project Direction

This project is expected to use:

- a pretrained base model with language ability
- no instruction tuning at the start
- a small, explicit SFT dataset for PyTorch API question answering
- clear before/after evaluation

The main educational goal is to learn the full process:

- base model inference
- instruction dataset preparation
- SFT training
- post-SFT evaluation

## Recommended Scope

Keep the first version small and focused.

Recommended first milestone:

- choose one pretrained base model
- run plain inference on a fixed PyTorch API prompt set
- prepare a small PyTorch API dataset
- run one small SFT experiment
- compare before vs after outputs

Do not mix too many goals into the first implementation.

## Suggested Structure

A good initial structure for this repository is:

- `src/`
- `data/`
- `experiments/`
- `README.md`
- `requirements.txt`

Likely future modules:

- `src/config.py`
- `src/inference.py`
- `src/train_sft.py`
- `src/dataset.py`
- `src/prompts.py`
- `src/evaluate.py`

## Model Rules

- Prefer starting from a pretrained **base** model, not an instruct/chat model, if the goal is to learn SFT properly.
- Keep the base model name explicit in configs and logs.
- Do not silently switch between base and instruct variants.
- Always keep tokenizer and model paired correctly.

## Dataset Rules

For the current PyTorch API assistant direction, prefer the simpler schema:

- `input`
- `output`

Do not repeat the same instruction text in every row if the training template already provides the shared behavior framing.

Be careful about:

- inconsistent formatting
- empty targets
- duplicated samples
- mixed task styles in the first experiment

For early runs, use a small dataset and keep the task narrow.

### PyTorch API Dataset Rules

When creating PyTorch API datasets, optimize for API understanding rather than generic tutorial tone.

Each dataset should favor:

- argument-level detail
- return value and shape detail
- behavior nuance
- debugging and edge cases
- API comparison questions

Dataset budget should increase real coverage, not just apparent size.

Prefer:

- one strong row per semantic fact
- one phrasing per shape/debugging fact unless the paraphrase adds clear new value
- more symbol coverage over more near-duplicate wording

Avoid inflating dataset size with rows that only change:

- `If x has shape ...` to `Suppose x has shape ...`
- `Why can ...` to `Why might ...` or `Why would ...`
- fake API names paired with almost identical refusal answers

When expanding a PyTorch API dataset:

- deduplicate aggressively
- compress repeated refusal patterns
- keep paraphrases only when they improve generalization meaningfully
- vary answer wording enough to avoid robotic output templates

Do not over-concentrate on only:

- `What does X do?`
- `How do I use X?`

Mix in more realistic question styles such as:

- why does this fail
- when should I use A vs B
- what shape does this return
- why is this tensor on the wrong device
- how do I fix this error

For PyTorch API answers, prefer including:

- what the API does
- key arguments
- return type or output shape when relevant
- common pitfalls
- a short code example when the API is usage-oriented

High-value PyTorch API dataset items include:

- `view` vs `reshape`
- `torch.tensor` vs `torch.as_tensor`
- `torch.from_numpy`
- `cat` vs `stack`
- `mm` vs `matmul`
- `model.eval()` vs `torch.no_grad()`
- `loss.backward()`
- `optimizer.zero_grad()`
- device mismatch
- dtype mismatch
- broadcasting mismatch
- fake API refusal / hallucination resistance

For fake API refusal specifically:

- keep a diverse but bounded block, usually around `10` to `20` rows
- vary namespaces such as `torch.*`, `nn.*`, `optim.*`, `Tensor.*`, and `Dataset`-style names
- vary refusal wording slightly
- do not spend large portions of the dataset on dozens of near-identical fake API rows

When writing answers:

- avoid vague wording when a precise API statement is possible
- prefer symbol-grounded explanations over generic training advice
- mention shared memory explicitly for APIs like `torch.from_numpy`
- mention gradient accumulation explicitly for `loss.backward()`
- be careful not to hardcode environment-specific examples like `device='cuda'` unless device semantics are the point
- prefer precise API behavior over hedged wording like `may raise depending on context` when the standard behavior is known
- vary opening sentence patterns so answers do not all sound like the same template

### Validation Set Rules

Validation data should stay in the same task domain, but it should not be a trivial rephrasing of training rows.

Validation sets should include:

- unseen phrasings
- nearby or unseen symbols
- shape reasoning
- debugging cases
- behavior / mode semantics

For PyTorch API validation specifically:

- include some symbols not directly trained, but close to the trained API surface
- use validation to test generalization, not just memorization

### Dataset Comparison Size Rule

Dataset comparison runs should keep dataset size fixed by default.

Recommended pilot default:

- training set: `50`
- validation set: `12`

Use the same train/validation size across comparison datasets unless there is a clear reason to change it and that change is recorded in the experiment log.

When the problem is already clear and the next run is a targeted scale-up rather than a small pilot comparison, use:

- training set: `200`
- validation set: `24`

Use larger runs only after the short pilot has already identified which capability gap to target.

## Evaluation Rules

Always compare:

- base model output
- SFT model output

Use the same fixed prompts before and after training.

Evaluation should include both:

- qualitative samples
- training/eval loss when applicable

If possible, keep a fixed prompt set under version control.

## Output Layout

Recommended output areas:

- model checkpoints: `data/output/checkpoints/`
- cached datasets or processed data: `data/output/cache/`
- evaluation results: `experiments/eval_results/`
- experiment logs: `experiments/logs/`

## Logging and Documentation

For each SFT run, log:

- base model used
- dataset used
- task type
- training settings
- evaluation prompts
- before/after outputs
- decision on whether the run helped

Keep logs:

- concise
- reproducible
- explicit about what changed and what stayed fixed

### Experiment Log Rules

When writing an experiment log under `experiments/logs/`, include at least:

- model name
- dataset name
- learning rate
- batch size
- timestamp
- goal of the run
- findings
- next step
- other important notes needed for reproducibility

Preferred log sections:

- `Run Info`
- `Goal`
- `Findings`
- `Next Step`
- `Other Important Info`

Log filenames should include at least:

- dataset name
- model name
- timestamp

Example:

- `dataset_1-Qwen-Qwen2.5-0.5B-2026-04-11.md`

If a later run uses `dataset_2`, the log filename should make that distinction explicit.

## What to Be Careful About

- Do not confuse pretrained base checkpoints with SFT checkpoints.
- Do not overwrite the original base model reference in logs.
- Do not run SFT without first checking base-model behavior.
- Do not change multiple major variables at once in early experiments.
- Do not assume a stronger model automatically means a better learning setup.

## Current Intended Workflow

A sensible first path for this repository is:

1. choose a pretrained base model
2. run baseline inference on PyTorch API prompts
3. prepare a small PyTorch API SFT dataset
4. run one small SFT experiment
5. compare outputs before and after
6. record results in `experiments/logs/`

## NOTE

Reply in Chinese even user is asking in English
