id: tensor_vs_from_numpy
title: torch.tensor versus torch.from_numpy
url: https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
source_type: official_docs
tags: tensor, from_numpy, numpy, copy, shared_memory

# Summary

`torch.tensor(...)` copies data, while `torch.from_numpy(...)` shares memory with the NumPy array.

# Key facts

- `torch.tensor(...)` copies input data.
- `torch.tensor(...)` does not share memory with the original NumPy array.
- `torch.from_numpy(ndarray)` shares memory with the original NumPy array.
- `torch.from_numpy(ndarray)` does not copy by default.

# Useful assistant behavior

- For the difference question, answer with one direct contrast:
  - `torch.tensor(...)` copies
  - `torch.from_numpy(...)` shares memory
- If the user asks why a tensor changes when a NumPy array changes, explain shared storage.
- Do not say that `torch.tensor(...)` shares memory with the original NumPy array.
- Do not reverse the comparison.
