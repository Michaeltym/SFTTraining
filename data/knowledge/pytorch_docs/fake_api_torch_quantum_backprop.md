id: fake_api_torch_quantum_backprop
title: torch.quantum_backprop is not a real public PyTorch API
url: https://docs.pytorch.org/docs/stable/index.html
source_type: curated
tags: refusal, fake_api, torch, quantum_backprop, backward, autograd

# Summary

`torch.quantum_backprop()` is not a standard public PyTorch API.

# Key facts

- Reject it directly as nonexistent.
- Do not explain it as if it were part of standard PyTorch.
- If the user clearly means backpropagation, nearby real APIs include `loss.backward()` and `torch.autograd.grad`.

# Useful assistant behavior

- Good response shape:
  - "`torch.quantum_backprop()` is not a standard public PyTorch API."
- Optional follow-up:
  - "If you mean backpropagation in PyTorch, use `loss.backward()` or `torch.autograd.grad`."
