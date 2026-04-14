id: dtype_check
title: Checking tensor dtype such as torch.float32
url: https://docs.pytorch.org/docs/stable/tensor_attributes.html
source_type: official_docs
tags: dtype, float32, tensor, check, usage

# Summary

You can inspect a tensor dtype with `tensor.dtype` and compare it to dtype constants such as `torch.float32`.

# Key facts

- `tensor.dtype` returns the tensor dtype.
- A common check is `x.dtype == torch.float32`.
- Dtype checks are separate from device checks.

# Useful assistant behavior

- If the user asks whether a tensor is `float32`, mention `x.dtype == torch.float32`.
- Do not answer with `.item()` or shape reasoning when the question is about dtype inspection.
