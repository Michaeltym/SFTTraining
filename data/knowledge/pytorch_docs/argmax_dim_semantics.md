id: argmax_dim_semantics
title: torch.argmax dim argument and output shape
url: https://docs.pytorch.org/docs/stable/generated/torch.argmax.html
source_type: official_docs
tags: argmax, dim, reduction, shapes, indices

# Summary

`torch.argmax` behaves differently depending on whether `dim` is provided. Without `dim`, it flattens the input and returns a single 0-D scalar index. With `dim=k`, it reduces along axis `k` and returns an index tensor whose shape is the input shape with axis `k` removed (or kept as size 1 when `keepdim=True`).

# Key facts

- `torch.argmax(x)` with no `dim` argument flattens `x` and returns a 0-D tensor (scalar) containing the index of the maximum element in the flattened input.
- `torch.argmax(x, dim=k)` returns a tensor of indices along axis `k`. The output shape is the input shape with axis `k` removed.
- `torch.argmax(x, dim=k, keepdim=True)` returns a tensor with axis `k` kept as size 1 instead of removed.
- `argmax` returns indices, not values. It does not return the maximum elements themselves.
- The returned dtype is `torch.int64` (long).

# Useful assistant behavior

- For shape questions, answer with the explicit output shape.
- For `torch.argmax(x)` on any shape, the output is a 0-D scalar tensor. It is not the same shape as `x`.
- For `torch.argmax(x, dim=1)` on shape `[B, C]`, the output shape is `[B]`.
- For `torch.argmax(x, dim=2)` on shape `[5, 7, 9]`, the output shape is `[5, 7]`.
- For `torch.argmax(x, dim=1, keepdim=True)` on shape `[B, C]`, the output shape is `[B, 1]`.
- Clearly distinguish the no-`dim` case (flatten, scalar) from the `dim=k` case (reduce one axis). Do not merge them.

# Common failure to avoid

- Do not say `torch.argmax(x)` returns a tensor with the same shape as `x`.
- Do not say `torch.argmax(x)` returns the maximum value; it returns an index.
- Do not confuse `dim=k` removal behavior with `keepdim=True` retention behavior.
- Do not describe `argmax` as returning values for classification logits; it returns the predicted class indices.
