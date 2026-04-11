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

第一轮 PyTorch API SFT。目标不是覆盖整个 PyTorch API，而是先验证小规模 SFT 是否能把 base model 从“续写论坛提问”拉向“直接回答 API 问题”，并观察三类关键能力是否改善：

- hallucination refusal
- shape reasoning
- debugging explanation

## Findings

- 训练数值是健康的。training loss 从 `1.5477` 降到 `0.8923`，validation loss 从 `1.5681` 降到 `1.4729`。
- 回答风格有改善。相比 baseline，模型更少直接续写 StackOverflow 风格提问，开始更像在回答 API 问题。
- 一部分 definition / comparison 题有进步，尤其是：
  - `torch.tensor`
  - `torch.topk`
  - `torch.argmax`
  - `DataLoader`
  - `torch.from_numpy vs torch.tensor`
- `hallucination_check` 基本没有改善：
  - `torch.memory_portal()`
  - `nn.SuperLayer`
  - `torch.quantum_backprop()`
  仍然被当成真实 API / module 来回答。
- `shape_reasoning` 仍然不稳：
  - `x.unsqueeze(1)` 仍然答错
  - `x.sum(dim=1, keepdim=True)` 仍然答错
  - `argmax(dim=2)` 和 `cat(dim=0)` 比 baseline 更接近正确，但还不够可靠
- `debugging` 题依旧是弱项：
  - `view` after `permute` 跑偏
  - `.item()` 错因解释不对
  - `zero_grad()` 题跑偏到别的 API
  - `cat` mismatch 解释不稳定

## Next Step

下一轮不要继续追求更大范围的 API 覆盖，先收窄到最关键的失败点：

- fake API refusal
- shape / dim / keepdim reasoning
- common debugging cases

更具体地说，下一版数据集应该集中强化：

- `unsqueeze`
- `argmax(dim=...)`
- `sum(..., keepdim=True)`
- `cat` / `stack`
- `view` vs `reshape`
- `.item()`
- `optimizer.zero_grad()`
- 明确拒绝不存在的 PyTorch API

## Other Important Info

- 这一轮数据规模是 `train 50 / validation 12`，适合作为 pilot，但很可能不足以把目标行为拉稳。
- 当前结果说明：`50` 条足够改变回答姿态，但不足以稳定建立 hallucination refusal 和 debugging reliability。
- 现阶段还不需要急着做自动评分器，先继续跑几轮更聚焦的数据集更合理。
- 现有 `eval_prompts` schema 已经足够支持后续半自动评分，当前最重要的是让数据方向先正确。
