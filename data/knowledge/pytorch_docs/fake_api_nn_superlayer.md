id: fake_api_nn_superlayer
title: nn.SuperLayer is not a real public torch.nn module
url: https://docs.pytorch.org/docs/stable/nn.html
source_type: curated
tags: refusal, fake_api, nn, superlayer, module

# Summary

`nn.SuperLayer` is not a standard public `torch.nn` module.

# Key facts

- Reject it directly as nonexistent in standard PyTorch.
- Do not explain it as if it were a real built-in layer.
- Do not assume the user means a custom class unless they explicitly say they implemented it themselves.

# Useful assistant behavior

- Good response shape:
  - "`nn.SuperLayer` is not a standard public `torch.nn` module."
