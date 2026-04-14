id: stack_shape_reasoning
title: Shape reasoning for torch.stack
url: https://docs.pytorch.org/docs/stable/generated/torch.stack.html
source_type: official_docs
tags: stack, shape, new_dimension, dim

# Summary

`torch.stack(...)` inserts a new dimension and stacks tensors along that new dimension.

# Key facts

- If each input tensor has shape `[2, 3, 4]`, then `torch.stack([x, x], dim=0)` returns shape `[2, 2, 3, 4]`.
- `torch.stack(...)` increases rank by one because it creates a new dimension.
- The size of the new dimension equals the number of tensors being stacked.

# Useful assistant behavior

- If the user asks for the output shape of `torch.stack([x, x], dim=0)` with `x` of shape `[2, 3, 4]`, answer `[2, 2, 3, 4]`.
- Explain that `stack` adds a new dimension.
- Do not confuse `stack` with `cat`.
