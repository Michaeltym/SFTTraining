## Run Info

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: `dataset_2`
- Task Type: `PyTorch API assistant`
- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `2`
- Baseline Result: [Qwen-Qwen2.5-0.5B-2026-04-11-201754.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-201754.json)
- Post-SFT Result: [dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json)
- Checkpoint: [Qwen-Qwen2.5-0.5B-dataset_2-16-2e-05.pt](/Users/michaeltan/Desktop/training/sft-training/data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_2-16-2e-05.pt)
- Timestamp: `2026-04-12`

## Goal

Second PyTorch API SFT run. The goal was a targeted scale-up rather than broader API coverage. This run focused on three weak areas:

- fake API refusal / hallucination resistance
- shape / dim / keepdim reasoning
- debugging and API-behavior explanations

## Findings

- `dataset_2` was clearly better than `dataset_1` on fake API refusal:
  - `torch.memory_portal()` started being recognized as nonexistent
  - `nn.SuperLayer` started being rejected more explicitly
  - `torch.quantum_backprop()` also moved in the refusal direction, but the answer still had repetition and poor finish quality
- `shape_reasoning` improved in a few places:
  - `x.unsqueeze(1)` changed from wrong to correct
  - `torch.argmax(x, dim=2)` started giving the correct shape `[5, 7]`
- `shape_reasoning` was still not stable:
  - `x.sum(dim=1, keepdim=True)` was still wrong, which means keepdim / reduction semantics were not learned reliably yet
- `debugging` remained the biggest weak area:
  - `view` after `permute` still failed to reliably explain the non-contiguous / `contiguous()` / `reshape()` distinction
  - `.item()` was closer to correct than baseline, but the explanation was still not clean enough
  - `model.eval()` vs `torch.no_grad()` introduced a serious error and even claimed that `model.eval()` was not a real PyTorch method
- Some comparison / semantics prompts still had regression risk:
  - `reshape` vs `view` was still unreliable
  - `from_numpy` vs `tensor` still sounded too much like forum continuation instead of a direct API answer

## Next Step

The next run, `dataset_3`, should keep the targeted scale-up size but redistribute the data budget:

- Keep hallucination refusal, but do not let it dominate the dataset
- Increase coverage for:
  - `view` / `reshape` / `contiguous`
  - `.item()`
  - `model.eval()` vs `torch.no_grad()`
  - `CrossEntropyLoss` / `BCEWithLogitsLoss`
  - `sum` / `mean` / `argmax` + `dim` / `keepdim`
- For shape prompts, prioritize facts the model is still getting wrong rather than adding many near-duplicate variants of facts it already partially learned

## Other Important Info

- This was the first run that moved from a small pilot comparison to a larger targeted run, with `train 200 / validation 24`.
- The result suggests that scaling data size does help, but if the dataset distribution leans too heavily toward fake API refusal, refusal improves faster than debugging / semantics.
- The highest-value next targets are not broader API coverage yet, but:
  - shape correctness
  - reduction semantics
  - debugging explanations
  - evaluation/training mode semantics
