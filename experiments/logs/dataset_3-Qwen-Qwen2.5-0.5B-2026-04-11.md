# Dataset 3 Log

## Run Info

- model: `Qwen/Qwen2.5-0.5B`
- dataset: `dataset_3`
- learning rate: `2e-5`
- batch size: `16`
- checkpoint: `data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_3-16-2e-05.pt`
- baseline file: `experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-154448.json`
- post-SFT file: `experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-11-164042.json`
- timestamp: `2026-04-11`

## Goal

This run aimed to make the comparison setup more controlled than `dataset_2`.

Main focus:

- `instruction_following`
- markdown table formatting
- light rewrite/style control

Compared with `dataset_2`, this run used the fixed comparison size rule:

- training set: `50`
- validation set: `12`

## Findings

- Markdown table generation improved compared with `dataset_2`.
- Weekly study plan table remained usable.
- JSON output remained acceptable.
- Strict instruction following still failed.
- One-sentence summary still returned an empty output.
- `exactly 20 words` still failed badly.
- `List 3 pros and 3 cons` still expanded into a long uncontrolled list.
- French translation regressed to an empty output.
- Rewrite quality regressed.
- The professional rewrite prompt returned an empty output.
- Pirate-style rewriting still failed.
- YAML output remained unreliable and still included unrelated extra text.

## Next Step

- Narrow the next comparison run to `instruction_following` only.
- Build `dataset_4` with no formatting or rewrite tasks mixed in.
- Focus `dataset_4` on:
  - one-sentence summary
  - exact word-count answers
  - bullet-only responses
  - strict list constraints
  - translation as a direct instruction-following task
- Keep `model`, `learning rate`, `batch size`, and evaluation prompts fixed.

## Other Important Info

- Baseline and post-SFT evaluation both used:
  - `max_new_token = 100`
  - `use_chat_template = false`
- Current checkpoint metadata:
  - `epoch = 2`
  - `training_loss = 0.7009153664112091`
  - `validation_loss = 1.2972513437271118`
- This run is more useful as a process correction than as a best-performing dataset.
- Main lesson:
  - fixing dataset size helped experimental rigor
  - narrowing the objective is now more important than mixing more task types
