# SFT Project Plan

## Goal

Build a small supervised fine-tuning project on top of a pretrained base model and turn it into a **PyTorch API assistant**.

The target behavior is:

- answer common PyTorch API questions clearly
- explain what an API does
- explain important parameters and return values
- provide short, correct examples when useful
- compare similar APIs when the question asks for differences

This project should help answer these questions clearly:

1. What does the base model look like before PyTorch-specific SFT?
2. How should PyTorch API Q&A data be prepared?
3. How much can a small SFT run improve PyTorch API answers?
4. What dataset style and size work best for this narrow assistant goal?

## Scope

This is still a learning project, but the task is now more specific.

The first goal is no longer generic instruction following.
The first goal is to make the model useful for **PyTorch API question answering**.

The initial assistant should focus on:

- `torch` tensor creation APIs
- common tensor shape/manipulation APIs
- common reduction/math APIs
- basic `torch.nn` building blocks
- basic `torch.optim` usage
- short code examples

It does **not** need to cover the full PyTorch ecosystem in the first phase.

## Phase 1: Project Setup

### Objective

Keep a clean codebase for:

- baseline inference
- PyTorch API dataset preparation
- SFT training
- evaluation
- experiment logging

### Expected Output

A runnable project skeleton with a clear separation between:

- config
- dataset preparation
- training
- evaluation
- logs

## Phase 2: Choose a Base Model

### Objective

Start with a pretrained base model that already has language ability, but has not been instruction tuned.

### Current First Choice

- `Qwen/Qwen2.5-0.5B`

### Decision Rule

- start with the smaller model to learn the pipeline quickly
- only move to a larger base model after the PyTorch API dataset pipeline is stable

## Phase 3: Baseline Inference

### Objective

Understand how the base model behaves on **PyTorch API questions** before any SFT.

### Tasks

- load the base model
- load the matching tokenizer
- run a fixed PyTorch API prompt set
- save outputs
- document weaknesses

### Typical Weaknesses to Watch

- vague or generic answers
- incorrect parameter explanations
- missing or weak examples
- confusion between similar APIs
- incorrect tensor shape reasoning

## Phase 4: Create a PyTorch API SFT Dataset

### Objective

Prepare a small but high-quality PyTorch API Q&A dataset with a consistent schema.

### Schema

- `input`
- `output`

Optional:

- place a shared instruction in the training template instead of repeating it in every row

### Dataset Rules

- keep the task narrow: PyTorch API assistance only
- prefer correct, concise answers over broad coverage
- include small code examples when useful
- avoid mixing unrelated tasks such as rewriting, translation, or general chatting
- keep formatting consistent across examples

### Recommended Example Types

- what an API does
- how to use an API
- important parameters
- return value / shape behavior
- simple example code
- difference between two related APIs
- common beginner mistakes

### Suggested Initial Dataset Size

- first pilot: `50` training examples
- first validation set: `12` examples

## Phase 5: Implement SFT Training

### Objective

Run one minimal SFT experiment on the PyTorch API dataset.

### Tasks

- tokenize the dataset correctly
- build labels for answer-only loss
- train on the chosen base model
- save checkpoints
- keep training settings simple and explicit

### First Success Criteria

- training runs end to end
- checkpoint is saved
- model can be loaded again
- before/after PyTorch API outputs can be compared

## Phase 6: Evaluate Before vs After

### Objective

Measure whether SFT improved PyTorch API answers.

### Tasks

- run the same PyTorch API prompts on the base model
- run the same prompts on the SFT model
- compare outputs side by side
- write an experiment log

### Evaluation Focus

- API explanation correctness
- parameter explanation quality
- example usefulness
- correctness of API comparisons
- answer conciseness
- common failure modes

## Phase 7: Iterate Carefully

### Objective

Improve the PyTorch API assistant with controlled experiments.

### Only Change One Main Variable Per Run

Examples:

- dataset content
- dataset size
- question style
- answer format
- learning rate
- number of epochs

### Early Iteration Strategy

1. keep model, learning rate, batch size, and eval prompts fixed
2. iterate on dataset design first
3. identify the best small dataset
4. only then scale the dataset up

### Avoid Early Complexity

Do not introduce too many of these at once:

- multiple unrelated domains
- broad generic assistant behavior
- large noisy synthetic datasets
- complex training tricks
- changing model and data at the same time

## Longer-Term Direction

If the small PyTorch API pilot works:

1. expand from `50` examples to a few hundred
2. widen coverage across more PyTorch APIs
3. consider combining SFT with retrieval over official docs later

The immediate goal is **not** full PyTorch coverage.
The immediate goal is a small, correct, useful PyTorch API assistant.

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
    ├── dataset.py
    ├── training.py
    ├── evaluate.py
    ├── baseline.py
    └── prompts.py
```

## Immediate Next Step

The next concrete step should be:

1. keep the current base model
2. define a PyTorch API evaluation prompt set
3. create `dataset_1` for PyTorch API assistance
4. run baseline inference on the new prompt set
5. run the first PyTorch API SFT experiment
