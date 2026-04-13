id: tensor_views_and_memory
title: Tensor views, reshape, permute, and contiguous memory
url: https://docs.pytorch.org/docs/stable/tensor_view.html
source_type: official_docs
tags: view, reshape, permute, transpose, contiguous, memory_layout

# Summary

PyTorch view operations often return tensors that share storage with the original tensor instead of copying data.

# Key facts

- `view()` returns a tensor with the same data but a different shape.
- `view()` only works when the requested shape is compatible with the tensor's current size and stride layout.
- `permute()` and `transpose()` are view operations and often produce non-contiguous tensors.
- A tensor can be non-contiguous even when its values are valid and readable.
- `contiguous()` returns the tensor itself if it is already contiguous; otherwise it returns a new contiguous tensor by copying data.
- `reshape()` is safer than `view()` when contiguity is unclear because `reshape()` may return either a view or a copy.

# Useful assistant behavior

- If a user asks why `view()` fails after `permute()` or `transpose()`, explain that the tensor is often non-contiguous.
- Recommend either:
  - `x = x.contiguous().view(...)`
  - or `x = x.reshape(...)`
- Do not say that `permute()` changes the stored numeric values. It changes dimension order and stride interpretation.

# Examples

- `x = x.permute(2, 0, 1)` often creates a non-contiguous tensor.
- `x.view(...)` may fail after that because the stride layout is no longer compatible.
- `x.reshape(...)` may still work because it can allocate a copy when needed.
