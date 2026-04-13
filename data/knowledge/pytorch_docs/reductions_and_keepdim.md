id: reductions_and_keepdim
title: Reduction semantics for sum, mean, argmax, and keepdim
url: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.sum.html
source_type: official_docs
tags: sum, mean, argmax, reduction, keepdim, shapes

# Summary

Reduction operations remove or retain dimensions depending on the arguments used.

# Key facts

- `sum(dim=...)` reduces the specified dimension.
- `mean(dim=...)` reduces the specified dimension.
- When `keepdim=False` or omitted, the reduced dimension is removed.
- When `keepdim=True`, the reduced dimension stays in the result with size `1`.
- `argmax(dim=...)` returns indices and removes the reduced dimension unless the API explicitly supports keeping it.

# Useful assistant behavior

- For shape questions, answer with the result shape directly.
- For `x.sum(dim=1, keepdim=True)` on shape `[8, 16]`, the correct output shape is `[8, 1]`.
- For `x.mean(dim=0, keepdim=True)` on shape `[5, 7]`, the correct output shape is `[1, 7]`.
- For `torch.argmax(x, dim=2)` on shape `[5, 7, 9]`, the correct output shape is `[5, 7]`.

# Common failure to avoid

- Do not keep the wrong axis.
- Do not answer `[1, 16]` for `x.sum(dim=1, keepdim=True)` when `x` has shape `[8, 16]`.
- Do not describe `argmax` as returning values; it returns indices.
