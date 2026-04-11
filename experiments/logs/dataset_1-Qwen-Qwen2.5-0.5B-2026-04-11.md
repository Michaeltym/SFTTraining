# Test 1 Log

## Run Info

- model: `Qwen/Qwen2.5-0.5B`
- dataset: `dataset_1`
- learning rate: `2e-5`
- batch size: `16`
- checkpoint: `data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_1-16-2e-05.pt`
- baseline file: `experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-154448.json`
- post-SFT file: `experiments/eval_results/post_sft/Qwen-Qwen2.5-0.5B-dataset_1-16-2e-05-2026-04-11-153943.json`
- timestamp: `2026-04-11`

## Goal

This first SFT run focused on:

- `instruction_following`
- `formatting_structure`
- light `rewrite` improvement

The intent was to see whether a small explicit SFT dataset could make the base model more obedient to formatting and instruction constraints.

## Findings

- `formatting_structure` improved partially.
- Markdown table generation improved clearly.
- Weekly study plan table generation improved clearly.
- `rewrite` improved partially.
- The professional rewrite prompt no longer failed with a refusal-style answer.
- Formal apology email was more usable after SFT.
- `instruction_following` did not improve enough.
- One-sentence summary failed with an empty output.
- `exactly 20 words` still failed badly.
- `bullet points only` still included extra prose before bullets.
- `JSON` and `YAML` outputs were still unreliable.
- JSON prompt returned plain key-value text instead of valid JSON.
- YAML prompt still drifted into unrelated requirement text.

## Next Step

- Build `dataset_2` with heavier coverage on strict constraint-following.
- Add more samples for:
  - one-sentence summary
  - exact word-count answers
  - bullet-only responses
  - strict JSON output
  - strict YAML output
  - stronger rewrite style contrast
- Keep `model`, `learning rate`, `batch size`, baseline prompts, and eval config fixed.
- Run the same train/validate/evaluate flow again and compare `dataset_1` vs `dataset_2`.

## Other Important Info

- Baseline and post-SFT evaluation both used:
  - `max_new_token = 100`
  - `use_chat_template = false`
- Current checkpoint metadata:
  - `epoch = 1`
  - `training_loss = 0.638204004083361`
  - `validation_loss = 1.193137764930725`
- This run should be treated as the first controlled experiment, not as a final successful SFT recipe.
- The main lesson from this run is that table formatting is easier to teach than strict instruction constraints such as exact word count or exact output schema.
