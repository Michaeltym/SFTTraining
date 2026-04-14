id: cat_shape_mismatch
title: torch.cat shape mismatch in non-concatenated dimensions
url: https://docs.pytorch.org/docs/stable/generated/torch.cat.html
source_type: official_docs
tags: cat, shape, mismatch, dimension, debugging

# Summary

`torch.cat(...)` requires input tensors to have the same size in every dimension except the concatenation dimension.

# Key facts

- When using `torch.cat(tensors, dim=k)`, all non-`k` dimensions must match.
- If non-concatenated dimensions differ, `torch.cat(...)` raises a size mismatch error.
- Concatenation does not pad or broadcast mismatched tensor sizes automatically.

# Useful assistant behavior

- If the user asks why `torch.cat` fails on mismatched shapes, say that all non-concatenated dimensions must match.
- Mention the concatenation dimension versus the other dimensions.
- Do not explain the failure as a contiguity or view issue.
