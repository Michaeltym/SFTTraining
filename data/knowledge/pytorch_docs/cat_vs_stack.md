id: cat_vs_stack
title: torch.cat versus torch.stack
url: https://docs.pytorch.org/docs/stable/generated/torch.cat.html
source_type: official_docs
tags: cat, stack, dimension, shape, comparison

# Summary

`torch.cat(...)` concatenates tensors along an existing dimension, while `torch.stack(...)` inserts a new dimension and stacks tensors along it.

# Key facts

- `torch.cat(tensors, dim=...)` joins tensors along an existing dimension.
- `torch.stack(tensors, dim=...)` creates a new dimension, then stacks tensors along that new dimension.
- `torch.cat(...)` and `torch.stack(...)` are not view operations.
- `torch.stack(...)` requires all input tensors to have the same shape.
- `torch.stack(...)` increases the output rank by one.

# Useful assistant behavior

- If the user asks for the difference between `torch.cat` and `torch.stack`, answer with one direct contrast: existing dimension versus new dimension.
- If the user asks about shape, mention that `stack` increases rank by one.
- Do not say that `cat` or `stack` share storage with the input tensors.
