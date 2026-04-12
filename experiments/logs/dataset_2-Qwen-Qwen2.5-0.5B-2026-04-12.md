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

第二轮 PyTorch API SFT。目标是做一次 targeted scale-up，不再追求更广 API 覆盖，而是集中强化三类短板：

- fake API refusal / hallucination resistance
- shape / dim / keepdim reasoning
- debugging and API-behavior explanations

## Findings

- `dataset_2` 在 fake API refusal 上明显比 `dataset_1` 更好：
  - `torch.memory_portal()` 开始被明确识别为不存在的 API
  - `nn.SuperLayer` 开始被明确拒绝
  - `torch.quantum_backprop()` 也开始走拒答方向，但输出仍有重复和收尾不稳的问题
- `shape_reasoning` 有局部进步：
  - `x.unsqueeze(1)` 从错误变成正确
  - `torch.argmax(x, dim=2)` 开始给出正确 shape `[5, 7]`
- `shape_reasoning` 仍然不稳定：
  - `x.sum(dim=1, keepdim=True)` 仍然答错，说明 keepdim / reduction 这块还没学稳
- `debugging` 依旧是最大短板：
  - `view` after `permute` 仍然没有稳定抓住 non-contiguous / `contiguous()` / `reshape()` 的核心
  - `.item()` 虽然比 baseline 更接近正确，但解释还是不够干净
  - `model.eval()` vs `torch.no_grad()` 反而出现明显错误，甚至说 `model.eval()` 不是 real PyTorch method
- 一些 comparison / semantics 题有回退风险：
  - `reshape` vs `view` 仍然不可靠
  - `from_numpy` vs `tensor` 的回答风格还是偏论坛续写，不够直接

## Next Step

下一轮 `dataset_3` 继续保持 targeted scale-up，但要重新分配预算：

- 保留 hallucination refusal，但不要再占太多比例
- 增加：
  - `view` / `reshape` / `contiguous`
  - `.item()`
  - `model.eval()` vs `torch.no_grad()`
  - `CrossEntropyLoss` / `BCEWithLogitsLoss`
  - `sum` / `mean` / `argmax` + `dim` / `keepdim`
- 对 shape 题，优先强化当前仍答错的事实点，不再给太多已经会做的近似题

## Other Important Info

- 这一轮是第一次从 pilot comparison 升级到 larger targeted run，规模为 `train 200 / validation 24`。
- 这轮结果说明：放大数据量本身是有用的，但如果数据分布偏向 fake API refusal，模型会在 refusal 上进步更快，而 debugging / semantics 仍然掉队。
- 当前最值得继续打的不是更宽的 API surface，而是：
  - shape correctness
  - reduction semantics
  - debugging explanations
  - evaluation/training mode semantics
