# PyTorch API Assistant Plan

## Goal

Build a small **PyTorch API assistant** on top of a pretrained base model.

The assistant should:

- answer common PyTorch API questions directly
- explain what an API does
- explain key arguments and return behavior
- answer shape and reduction questions correctly
- explain common debugging failures
- refuse fake or nonexistent PyTorch APIs instead of hallucinating

The short-term product direction is:

1. train a better PyTorch API assistant
2. compare baseline vs adapted model behavior
3. expose the model behind a small frontend web app

## Current Direction

The project has already shown that:

- small full-parameter SFT runs can change answer style
- local full fine-tuning is too slow for fast iteration
- the remaining bottlenecks are:
  - hallucination refusal
  - shape reasoning
  - debugging / semantics reliability

Because of that, the next implementation direction is:

- keep the same base model
- switch from local full fine-tuning to **LoRA-based SFT**
- continue iterating on PyTorch API datasets
- keep evaluation prompts fixed

## Base Model

Current model:

- `Qwen/Qwen2.5-0.5B`

Rules:

- keep using the pretrained **base** model, not an instruct model
- keep tokenizer and model paired
- keep the base model name explicit in config, logs, and checkpoints

## Training Strategy

### Phase 1: Local LoRA SFT

The next training implementation should use:

- supervised fine-tuning
- LoRA adapters
- the existing PyTorch API dataset format

Why:

- much cheaper local iteration than full fine-tuning
- smaller checkpoints
- easier repeated dataset experiments
- more realistic path for continued local development

### Phase 2: Optional Remote Training

If local LoRA is still too slow or the project grows:

- move the training workflow to Runpod
- keep the same codepath where possible
- use the local LoRA path as the reference workflow first

## Dataset Direction

Current schema:

- `input`
- `output`

Dataset goals:

- optimize for API understanding, not generic chatbot style
- prioritize correctness over breadth
- include argument-level and behavior-level details
- include shape / dim / keepdim questions
- include debugging and failure explanations
- include fake API refusal examples

Avoid:

- repeated shared instruction text per row
- near-duplicate paraphrases that inflate dataset size
- broad generic chatting tasks
- non-PyTorch tasks such as rewrite / translation / general assistant data

## Evaluation Strategy

Keep one fixed PyTorch API evaluation set under version control.

Evaluation should compare:

- baseline model output
- adapted model output

Evaluation focus:

- directness of answers
- factual correctness
- shape reasoning correctness
- debugging usefulness
- fake API refusal
- API comparison correctness

Current evaluation objects already support richer metadata such as:

- `must_include`
- `must_not_include`
- `tags`
- `gold_type`

For now, continue using them mainly for structured manual review.
Automatic scoring can come later.

## Experiment Policy

For each run:

- keep the model fixed
- keep eval prompts fixed
- change one main thing at a time
- log what changed and what stayed fixed

Current main experimental variable:

- dataset content and dataset distribution

Do not mix too many new changes at once, especially:

- changing model
- changing eval prompts
- changing data format
- changing training method
- changing deployment stack

The one intentional training-method change now is:

- full fine-tune -> LoRA

That should be implemented first and stabilized before larger architectural changes.

## Frontend Direction

After the LoRA-based PyTorch API assistant is good enough, build a frontend app on top of it.

Working product direction:

- web app name can be something like `AskTorch`
- the app should accept PyTorch questions and return concise API-focused answers

Initial frontend scope:

- single prompt box
- generated answer area
- simple example prompts
- possibly a note when the assistant refuses a fake API

This frontend is not the next coding step.
The next coding step is to make the training workflow cheaper and more stable with LoRA.

## Immediate Next Steps

1. update the training path from full fine-tuning to LoRA
2. keep checkpoint / evaluate / resume behavior coherent with LoRA
3. run `dataset_3` with LoRA
4. compare baseline vs LoRA-SFT outputs
5. decide whether dataset iteration should continue locally or move to Runpod
