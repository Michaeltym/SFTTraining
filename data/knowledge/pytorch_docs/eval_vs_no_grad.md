id: eval_vs_no_grad
title: Difference between model.eval and torch.no_grad
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
source_type: official_docs
tags: eval, no_grad, inference, autograd, dropout, batchnorm

# Summary

`model.eval()` changes module behavior, while `torch.no_grad()` disables gradient tracking.

# Key facts

- `model.eval()` is a real `torch.nn.Module` method.
- `model.eval()` switches the module into evaluation mode.
- Evaluation mode affects modules such as dropout and batch normalization.
- `model.eval()` does not disable gradient tracking by itself.
- `torch.no_grad()` disables gradient calculation for operations inside its context.
- `torch.no_grad()` reduces memory use for inference and prevents autograd from tracking the computation.

# Useful assistant behavior

- Never claim that `model.eval()` is not a real PyTorch API.
- Never say that `model.eval()` alone disables gradients.
- Explain the distinction clearly:
  - `model.eval()` changes module behavior
  - `torch.no_grad()` changes autograd behavior
- It is normal to use both together during inference.
