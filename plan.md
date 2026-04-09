# SFT Project Plan

## Goal

Build a small learning project for supervised fine-tuning (SFT) on top of a pretrained base language model.

This project should help answer these questions clearly:

1. What does a pretrained base model look like before instruction tuning?
2. How should instruction-style data be prepared?
3. How does supervised fine-tuning change model behavior?
4. How should before/after results be evaluated and logged?

## Scope

This project is not for building a production assistant.
The first goal is to learn the full workflow end to end with a small, controlled setup.

The first version should stay small and simple.

## Phase 1: Project Setup

### Objective

Create a minimal but clean codebase for base-model inference and future SFT work.

### Tasks

- create project structure
- add `README.md`
- add `requirements.txt`
- add `src/`
- add `data/`
- add `experiments/`
- define config structure

### Expected Output

A runnable project skeleton with a clear separation between:

- config
- inference
- dataset preparation
- training
- evaluation
- logs

## Phase 2: Choose a Base Model

### Objective

Start with a pretrained base model that already has language ability, but has not been instruction tuned.

### Recommended First Choice

- `Qwen/Qwen2.5-0.5B`

### Recommended Second Choice

- `HuggingFaceTB/SmolLM2-1.7B`

### Decision Rule

- use the smaller model first if the goal is to learn the pipeline quickly
- use the larger model later if the first pipeline works and more capacity is needed

## Phase 3: Baseline Inference

### Objective

Understand how the base model behaves before any SFT.

### Tasks

- load base model
- load matching tokenizer
- run a fixed prompt set
- save outputs
- document observed weaknesses

### Why This Matters

SFT is easier to understand if base-model behavior is recorded first.

## Phase 4: Create a Small SFT Dataset

### Objective

Prepare a small instruction dataset with a consistent schema.

### Recommended Initial Schema

- `instruction`
- `input`
- `output`

### Rules

- keep the first dataset small
- keep the task narrow
- avoid mixing too many task styles at the start
- prefer clean, explicit supervision over large noisy data

### Suggested First Dataset Size

- 50 to 300 examples is enough for the first learning run

## Phase 5: Implement SFT Training

### Objective

Run one minimal supervised fine-tuning experiment.

### Tasks

- tokenize the SFT dataset
- format training samples correctly
- train on the chosen base model
- save checkpoints
- keep training settings simple and explicit

### First Success Criteria

- training runs end to end
- checkpoint is saved
- model can be loaded again
- before/after outputs can be compared

## Phase 6: Evaluate Before vs After

### Objective

Measure whether SFT improved behavior on the target task.

### Tasks

- run the same fixed prompts on the base model
- run the same fixed prompts on the SFT model
- compare outputs side by side
- write a short experiment log

### Evaluation Focus

- instruction following
- output format stability
- relevance to the task
- common failure modes

## Phase 7: Iterate Carefully

### Objective

Improve the SFT setup with controlled experiments.

### Only Change One Main Variable Per Run

Examples:

- dataset size
- prompt format
- base model
- learning rate
- number of epochs
- maximum sequence length

### Avoid Early Complexity

Do not introduce too many of these at once:

- multiple datasets
- multiple task families
- heavy optimization stacks
- complicated evaluation pipelines

## Suggested Initial Folder Layout

```text
sft-training/
├── AGENTS.md
├── plan.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── output/
├── experiments/
│   ├── eval_results/
│   └── logs/
└── src/
    ├── config.py
    ├── inference.py
    ├── dataset.py
    ├── train_sft.py
    ├── evaluate.py
    └── prompts.py
```

## Immediate Next Step

The next concrete step should be:

1. create the initial project skeleton
2. choose the first base model
3. run baseline inference
4. save those baseline outputs before writing any SFT code
