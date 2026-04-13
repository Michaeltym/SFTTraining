## Run Info

- Model: `Qwen/Qwen2.5-1.5B`
- Dataset: `dataset_3`
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT`
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs: `1`
- LoRA Config:
  - `r = 8`
  - `alpha = 16`
  - `dropout = 0.05`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Baseline Reference: [Qwen-Qwen2.5-0.5B-2026-04-11-201754.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-201754.json)
- Previous Best Full Fine-Tune Reference: [dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json)
- Previous Best LoRA Reference: [dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-12-202539.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-12-202539.json)
- Post-SFT Result: [dataset_3-Qwen-Qwen2.5-1.5B-8-0.0001-2026-04-12-235600.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-1.5B-8-0.0001-2026-04-12-235600.json)
- Timestamp: `2026-04-12`

## Goal

This run tested whether moving from `Qwen/Qwen2.5-0.5B` to `Qwen/Qwen2.5-1.5B` would improve the same `dataset_3` PyTorch API assistant task without changing the dataset itself.

The main question was:

- is the current bottleneck mainly model capacity?

## Findings

- The result did not clearly improve over the best `0.5B` LoRA run.
- Some shape prompts were correct:
  - `unsqueeze(1)` returned `[2, 1, 3, 4]`
  - `sum(dim=1, keepdim=True)` returned `[8, 1]`
  - `argmax(dim=2)` was also correct
- However, the run regressed badly on high-value refusal and semantics prompts:
  - `torch.memory_portal()` was treated like a real API
  - `nn.SuperLayer` was treated like a usable layer instead of being rejected
  - `torch.quantum_backprop()` was explained as if it were a real concept or library API
- Debugging and semantics prompts were still weak:
  - `view` after `permute` still failed to explain the non-contiguous / `contiguous()` / `reshape()` distinction
  - `model.eval()` vs `torch.no_grad()` still contained incorrect statements
  - `.item()` still did not produce a clean API-grounded explanation
- Some direct API explanation prompts also regressed into long, repetitive, forum-style continuation:
  - `torch.tensor`
  - `torch.from_numpy` vs `torch.tensor`
  - `reshape` vs `view`
  - `to(dtype=..., device=...)`

## Conclusion

- This run does not become the new best checkpoint.
- Bigger model size alone did not fix the main failure modes.
- The current best LoRA result remains the earlier `Qwen/Qwen2.5-0.5B` `dataset_3` run at `lr = 1e-4`, `epoch 1`.

## Next Step

Do not continue changing model size right away.

- Freeze the training conclusions from:
  - `dataset_2` full fine-tune
  - `dataset_3` best LoRA run on `0.5B`
  - this `1.5B` comparison run
- Choose one of two directions next:
  - ship a prototype using the current best `0.5B` checkpoint
  - or redesign the training objective instead of only scaling model size

The recommended next move is to stop expanding training experiments and start building a minimal product/demo around the current best checkpoint.

## Other Important Info

- This run was useful because it invalidated a simple assumption:
  - moving to `1.5B` did not automatically produce a better PyTorch API assistant
- That means the next meaningful improvement is less likely to come from random model-size hopping and more likely to come from:
  - product iteration
  - inference constraints
  - retrieval / docs grounding
  - or a more deliberate training redesign
