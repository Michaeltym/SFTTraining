## Run Info

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: `dataset_1`
- Task Type: `PyTorch API assistant`
- Learning Rate: `2e-5`
- Batch Size: `16`
- Epochs: `2`
- Baseline Result: [Qwen-Qwen2.5-0.5B-2026-04-11-201754.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/baseline/Qwen-Qwen2.5-0.5B-2026-04-11-201754.json)
- Post-SFT Result: [dataset_1-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-11-203543.json](/Users/michaeltan/Desktop/training/sft-training/experiments/eval_results/post_sft/dataset_1-Qwen-Qwen2.5-0.5B-16-2e-05-2026-04-11-203543.json)
- Checkpoint: [Qwen-Qwen2.5-0.5B-dataset_1-16-2e-05.pt](/Users/michaeltan/Desktop/training/sft-training/data/checkpoints/Qwen-Qwen2.5-0.5B-dataset_1-16-2e-05.pt)
- Timestamp: `2026-04-11`

## Goal

First PyTorch API SFT run. The goal was not to cover the entire PyTorch API surface. The goal was to verify whether a small SFT run could push the base model away from “continuing forum-style questions” and toward “directly answering PyTorch API questions,” while checking three key capability areas:

- hallucination refusal
- shape reasoning
- debugging explanation

## Findings

- Training metrics were healthy. Training loss dropped from `1.5477` to `0.8923`, and validation loss dropped from `1.5681` to `1.4729`.
- Answer style improved. Compared with baseline, the model was less likely to continue the prompt in a StackOverflow-style voice and more likely to attempt a direct API answer.
- Some definition / comparison prompts improved, especially:
  - `torch.tensor`
  - `torch.topk`
  - `torch.argmax`
  - `DataLoader`
  - `torch.from_numpy vs torch.tensor`
- `hallucination_check` barely improved:
  - `torch.memory_portal()`
  - `nn.SuperLayer`
  - `torch.quantum_backprop()`
  were still treated as if they were real APIs / modules.
- `shape_reasoning` was still unstable:
  - `x.unsqueeze(1)` was still wrong
  - `x.sum(dim=1, keepdim=True)` was still wrong
  - `argmax(dim=2)` and `cat(dim=0)` were closer to correct than baseline, but still not reliable
- `debugging` remained weak:
  - `view` after `permute` drifted off-target
  - `.item()` error explanation was incorrect
  - `zero_grad()` drifted into unrelated API explanation
  - `cat` mismatch explanation was unstable

## Next Step

The next run should not chase broader API coverage yet. It should narrow focus to the most important failure modes first:

- fake API refusal
- shape / dim / keepdim reasoning
- common debugging cases

More concretely, the next dataset should emphasize:

- `unsqueeze`
- `argmax(dim=...)`
- `sum(..., keepdim=True)`
- `cat` / `stack`
- `view` vs `reshape`
- `.item()`
- `optimizer.zero_grad()`
- 明确拒绝不存在的 PyTorch API

## Other Important Info

- This run used `train 50 / validation 12`, which was appropriate for a pilot, but likely too small to stabilize the target behaviors.
- The result suggests that `50` examples are enough to change answer style, but not enough to build stable hallucination refusal or debugging reliability.
- There is no need to rush into an automatic scorer yet. It is more useful to run a few more focused dataset iterations first.
- The current `eval_prompts` schema is already sufficient for later semi-automatic scoring. The more important task right now is to get the dataset direction right.
