id: fake_api_refusal
title: Refuse nonexistent PyTorch APIs and redirect to nearby real APIs
url: https://docs.pytorch.org/docs/stable/index.html
source_type: curated
tags: refusal, fake_api, hallucination, guardrail

# Summary

If a user asks about a nonexistent PyTorch API, the assistant should reject the premise directly instead of inventing behavior.

# Core rule

- If the API is not a standard public PyTorch API, say so clearly.
- Do not fabricate arguments, return values, side effects, or examples.
- If helpful, redirect to a nearby real API.

# Useful patterns

- Fake: `torch.memory_portal()`  
  Better response: this is not a standard public PyTorch API; if the user means memory management, discuss real APIs such as device placement, memory format, or tensor views only if relevant.

- Fake: `torch.quantum_backprop()`  
  Better response: this is not a standard public PyTorch API; if the user means backpropagation, mention real APIs such as `loss.backward()` or `torch.autograd.grad`.

- Fake: `nn.SuperLayer`  
  Better response: this is not a standard public `torch.nn` module; do not explain it like a real built-in layer.

- Fake: `torch.safe_backward()`  
  Better response: this is not a standard public PyTorch API; if relevant, redirect to `loss.backward()`.

# Failures to avoid

- Do not reinterpret a fake API as if it belongs to some other real PyTorch subpackage.
- Do not answer with forum-style speculation.
- Do not treat an obviously fake API as a custom layer unless the user explicitly says they wrote it themselves.
