id: eval_vs_no_grad
title: Difference between model.eval and torch.no_grad
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
source_type: official_docs
tags: eval, no_grad, inference, autograd

# Summary

`model.eval()` changes module behavior, while `torch.no_grad()` disables gradient tracking. They operate on different axes and are commonly used together during inference.

# Key facts

- `model.eval()` is a real `torch.nn.Module` method.
- `model.eval()` switches the module into evaluation mode; mode-sensitive modules like `nn.Dropout` and `nn.BatchNorm*d` read that flag and change their behavior accordingly. The specific mechanisms are documented in `dropout_basics.md` and `batchnorm_basics.md`.
- `model.eval()` does not disable gradient tracking by itself.
- `torch.no_grad()` disables gradient calculation for operations inside its context.
- `torch.no_grad()` reduces memory use for inference and prevents autograd from tracking the computation.
- `torch.no_grad()` does not put modules into eval mode; dropout and BN still behave according to whatever mode the module is currently in.

# Useful assistant behavior

- Never claim that `model.eval()` is not a real PyTorch API.
- Never say that `model.eval()` alone disables gradients.
- Never say that `torch.no_grad()` alone switches BN to running statistics or disables dropout.
- Explain the distinction clearly:
  - `model.eval()` changes module behavior
  - `torch.no_grad()` changes autograd behavior
- It is normal to use both together during inference.
