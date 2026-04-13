id: tensor_vs_from_numpy
title: torch.tensor versus torch.from_numpy
url: https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
source_type: official_docs
tags: tensor, from_numpy, numpy, copy, shared_memory

# Summary

`torch.tensor(...)` copies data, while `torch.from_numpy(...)` shares memory with the NumPy array.

# Key facts

- `torch.tensor(...)` creates a new tensor by copying input data.
- `torch.tensor(...)` does not share memory with the original NumPy array.
- `torch.from_numpy(ndarray)` creates a tensor that shares memory with the NumPy array.
- `torch.from_numpy(ndarray)` does not copy by default.
- If memory is shared, changing the NumPy array can change the tensor, and changing the tensor can change the NumPy array.

# Useful assistant behavior

- If the user asks for the difference between `torch.tensor` and `torch.from_numpy`, explain copy vs shared memory directly.
- If the user asks why a tensor changes when a NumPy array changes, explain shared storage.
- Do not say that `torch.tensor(...)` shares memory with the original NumPy array.
- Do not reverse the comparison: `torch.tensor(...)` copies, while `torch.from_numpy(...)` shares memory.
