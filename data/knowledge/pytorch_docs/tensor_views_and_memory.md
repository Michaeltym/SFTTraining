id: tensor_views_and_memory
title: Tensor views, reshape, permute, and contiguous memory
url: https://docs.pytorch.org/docs/stable/tensor_view.html
source_type: official_docs
tags: view, reshape, permute, transpose, contiguous, memory_layout

# Summary

`view()` reshapes without copying when the existing layout is compatible, while `reshape()` may return either a view or a copy.

# Key facts

- `view()` reshapes without copying and requires a compatible size-and-stride layout.
- `permute()` and `transpose()` often produce non-contiguous tensors.
- `contiguous()` returns the tensor itself if it is already contiguous; otherwise it returns a new contiguous tensor by copying data.
- `reshape()` is safer when contiguity is unclear because it may return either a view or a copy.

# Useful assistant behavior

- If a user asks why `view()` fails after `permute()` or `transpose()`, answer with one main reason: the tensor is often non-contiguous.
- Recommend either `x = x.contiguous().view(...)` or `x = x.reshape(...)`.
- Do not say that `view()` is safer than `reshape()`.
- Do not say that `permute()` changes stored numeric values.
