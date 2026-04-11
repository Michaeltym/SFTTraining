# Dataset 2 Log

## Run Info

- model: `Qwen/Qwen2.5-0.5B`
- dataset: `dataset_2`
- learning rate: `2e-5`
- batch size: `16`
- checkpoint: `data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_2-16-2e-05.pt`
- baseline file: `experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-154448.json`
- post-SFT file: `experiments/eval_results/post_sft/dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-11-160817.json`
- timestamp: `2026-04-11`

## Goal

This run focused on a narrower version of the same objective as `dataset_1`, with heavier emphasis on:

- `instruction_following`
- `formatting_structure`
- strict output constraints
- light `rewrite` improvement

The main change from `dataset_1` was stronger targeting of exact word count, JSON/YAML-only output, and no-extra-text response behavior.

## Findings

- `dataset_2` improved strict JSON output compared with both baseline and `dataset_1`.
- The JSON prompt returned valid JSON instead of mixed plain text.
- `dataset_2` improved the YAML prompt partially, but it still included extra explanation before the YAML block.
- `dataset_2` did not improve strict instruction following enough.
- One-sentence summary still failed with an empty output.
- `exactly 20 words` still failed badly.
- The remote-work pros/cons prompt became worse than `dataset_1` because it expanded into a long repetitive list.
- Markdown table quality regressed compared with `dataset_1`.
- Weekly study plan table remained good.
- Rewrite quality did not improve overall.
- The professional rewrite prompt regressed and returned almost unchanged text.
- Pirate-style rewrite still failed.

## Comparison With Dataset 1

- `dataset_1` was better for markdown table generation.
- `dataset_1` was better for the professional rewrite prompt.
- `dataset_2` was better for strict JSON output.
- `dataset_2` was slightly better for the YAML prompt because it at least included a usable YAML fragment.
- Both datasets failed on one-sentence summary.
- Both datasets failed on exact 20-word control.
- Neither dataset produced a convincing pirate-style rewrite.
- This comparison is directional, not fully controlled, because `dataset_1` and `dataset_2` were not the same size.

## Next Step

- Keep `dataset_2` ideas for strict JSON and YAML supervision.
- Build `dataset_3` as a targeted follow-up dataset instead of simply making `dataset_2` larger.
- From `dataset_3` onward, keep dataset sizes matched across comparison runs.
- Add dense samples for:
  - one-sentence summary with output-only targets
  - exact 20-word answers
  - bullet-only answers with no intro sentence
  - markdown table generation
  - pirate-style rewriting
- Keep `model`, `learning rate`, `batch size`, and evaluation prompts fixed for the next comparison run.

## Other Important Info

- Baseline and post-SFT evaluation both used:
  - `max_new_token = 100`
  - `use_chat_template = false`
- Current checkpoint metadata:
  - `epoch = 1`
  - `training_loss = 1.1139597694079082`
  - `validation_loss = 1.6765620708465576`
- Dataset size caveat:
  - `dataset_1` training set: `105`
  - `dataset_1` validation set: `16`
  - `dataset_2` training set: `36`
  - `dataset_2` validation set: `12`
- Compared with `dataset_1`, this run looks more specialized but not clearly better overall.
- The main lesson from this run is that stricter dataset wording can help JSON formatting, but it does not automatically solve exact-constraint following.
