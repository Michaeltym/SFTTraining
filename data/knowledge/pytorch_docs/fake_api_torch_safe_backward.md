id: fake_api_torch_safe_backward
title: torch.safe_backward is not a real public PyTorch API
url: https://docs.pytorch.org/docs/stable/index.html
source_type: curated
tags: refusal, fake_api, torch, safe_backward, backward

# Summary

`torch.safe_backward()` is not a standard public PyTorch API.

# Key facts

- Reject it directly as nonexistent.
- Do not invent safety semantics or arguments.
- If the user clearly means backward propagation, a nearby real API is `loss.backward()`.

# Useful assistant behavior

- Good response shape:
  - "`torch.safe_backward()` is not a standard public PyTorch API."
- Optional follow-up:
  - "If you mean standard backpropagation, use `loss.backward()`."
