id: as_tensor_vs_tensor
title: torch.as_tensor versus torch.tensor
url: https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html
source_type: official_docs
tags: as_tensor, tensor, copy, reuse, numpy

# Summary

`torch.as_tensor(...)` may reuse existing data, while `torch.tensor(...)` creates a new tensor copy.

# Key facts

- `torch.tensor(...)` copies input data.
- `torch.as_tensor(...)` reuses existing data when possible.
- `torch.as_tensor(...)` avoids a copy only when it can safely reuse the source representation.

# Useful assistant behavior

- If the user asks for the difference between `torch.as_tensor` and `torch.tensor`, explain possible reuse vs guaranteed copy.
- Do not claim that `torch.as_tensor(...)` always copies.
- Do not claim that `torch.tensor(...)` reuses input storage.
