## Run Info

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: `dataset_3`
- Task Type: `PyTorch API assistant`
- Training Method: `LoRA SFT`
- Learning Rate: `1e-4`
- Batch Size: `8`
- Epochs Tested: `1` and `2`
- LoRA Config:
  - `r = 16`
  - `alpha = 32`
  - `dropout = 0.05`
  - `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Baseline Result: [Qwen-Qwen2.5-0.5B-2026-04-11-201754.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-201754.json)
- Previous Best Full Fine-Tune Result: [dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_2-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-12-083732.json)
- Earlier Weak LoRA Result: [dataset_3-Qwen-Qwen2.5-0.5B-8-2e-05-2026-04-12-194520.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-0.5B-8-2e-05-2026-04-12-194520.json)
- Best LoRA Result So Far: [dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-12-202539.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-12-202539.json)
- Mixed Epoch 2 Result: [dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-13-083619.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_3-Qwen-Qwen2.5-0.5B-8-0.0001-2026-04-13-083619.json)
- Checkpoint: [Qwen-Qwen2.5-0.5B-dataset_3-8-0.0001.pt](/Users/michaeltan/Desktop/training/sft-training/data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_3-8-0.0001.pt)
- Adapter: [Qwen-Qwen2.5-0.5B-dataset_3-8-0.0001](/Users/michaeltan/Desktop/training/sft-training/data/adapters/Qwen-Qwen2.5-0.5B-dataset_3-8-0.0001)
- Timestamp: `2026-04-12` to `2026-04-13`

## Goal

Third PyTorch API run. This run kept the targeted `dataset_3` data but switched from full fine-tuning to LoRA. The main goal was to test whether a local LoRA setup could recover the same behavior improvements that the earlier full fine-tune run showed on:

- fake API refusal
- shape reasoning
- debugging / semantics prompts

## Findings

- The first LoRA setting for `dataset_3` was too weak:
  - `lr = 2e-5` with LoRA caused outputs to stay close to baseline forum-style continuation
  - hallucination refusal, shape reasoning, and debugging all remained weak
- Increasing LoRA strength helped clearly:
  - changing to `lr = 1e-4`, `r = 16`, `alpha = 32` produced the best `dataset_3` result so far
  - this was a real improvement over the earlier `dataset_3` LoRA runs
- The improved LoRA run was better than baseline on several prompts:
  - `torch.memory_portal()` started being rejected as a non-standard PyTorch API
  - `unsqueeze(1)` became correct and returned `[2, 3, 1, 4]`
  - `torch.tensor` and `to(dtype=..., device=...)` started sounding more like direct API answers instead of pure forum continuation
- The run still did not fully solve the hardest `dataset_3` targets:
  - `x.sum(dim=1, keepdim=True)` was still wrong and answered `[1, 16]` instead of `[8, 1]`
  - `view` after `permute` still did not reliably explain the non-contiguous / `contiguous()` / `reshape()` distinction
  - `.item()` remained noisy and not cleanly API-grounded
  - `torch.quantum_backprop()` still repeated instead of refusing cleanly
  - `from_numpy` vs `tensor` was still unstable
- Running a second epoch with the stronger LoRA setting produced a mixed result rather than a clean improvement:
  - `x.sum(dim=1, keepdim=True)` improved and became correct
  - however, hallucination refusal became less stable again
  - `model.eval()` vs `torch.no_grad()` regressed into a serious error
  - `.item()` regressed and incorrectly treated `.item()` as if it were not a real PyTorch method
  - because of these regressions, the epoch 2 result is not the best overall checkpoint
- Compared with the earlier full fine-tune `dataset_2` run:
  - the new LoRA run closed part of the gap
  - but it still did not clearly beat `dataset_2` on the most important reasoning / refusal prompts

## Next Step

Do not create a new dataset immediately. The current bottleneck is no longer obviously the dataset itself.

- Keep `dataset_3` fixed for now
- Treat the `epoch 1` run at `lr = 1e-4`, `r = 16`, `alpha = 32` as the current best local LoRA checkpoint
- If more local tuning is attempted, compare against that epoch 1 run instead of the weaker `2e-5` LoRA runs
- If the remaining shape/debugging errors still do not improve, the next decision should be whether:
  - this base model is too small for the target behavior, or
  - training should move to stronger hardware / a larger base model

## Other Important Info

- `dataset_3` is the first run where LoRA plumbing was fully wired through:
  - adapter save
  - checkpoint metadata
  - resume path
  - evaluate path
- Local LoRA is workable, but stronger LoRA settings made training much slower on the current machine.
- The current evidence suggests:
  - the earlier LoRA failures were not mainly caused by dataset size
  - a major issue was that the original LoRA learning rate was too low
  - `Qwen/Qwen2.5-0.5B` may still be near its practical limit for this PyTorch API assistant goal
