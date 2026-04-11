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
- a small, explicit SFT dataset
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
- run plain inference on a fixed prompt set
- prepare a small instruction dataset
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

Instruction data should be explicit and structured.

Prefer a stable schema such as:

- `instruction`
- `input`
- `output`

or a chat-style message list if the chosen model format requires it.

Be careful about:

- inconsistent formatting
- empty targets
- duplicated samples
- mixed task styles in the first experiment

For early runs, use a small dataset and keep the task narrow.

### Dataset Comparison Size Rule

From `dataset_3` onward, dataset comparison runs should keep dataset size fixed by default.

Recommended default:

- training set: `50`
- validation set: `12`

Use the same train/validation size across comparison datasets unless there is a clear reason to change it and that change is recorded in the experiment log.

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
2. run baseline inference
3. prepare a small SFT dataset
4. run one small SFT experiment
5. compare outputs before and after
6. record results in `experiments/logs/`

## NOTE

Reply in Chinese even user is asking in English
