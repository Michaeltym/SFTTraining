id: tensor_creation_and_numpy_interop
title: torch.tensor, torch.as_tensor, and torch.from_numpy
url: https://docs.pytorch.org/docs/stable/generated/torch.tensor.html
source_type: official_docs
tags: tensor, as_tensor, from_numpy, numpy, copy, shared_memory

# Summary

These APIs mainly differ in whether they copy data or share existing storage.

# Key facts

- `torch.tensor(...)` creates a new tensor by copying input data.
- `torch.as_tensor(...)` reuses existing data when possible and avoids copies when it can.
- `torch.from_numpy(ndarray)` creates a tensor that shares memory with the NumPy array.
- If the tensor and NumPy array share memory, writing through one can affect the other.

# Useful assistant behavior

- For `from_numpy` vs `tensor`, explain shared memory vs copy.
- For `as_tensor` vs `tensor`, explain possible reuse vs guaranteed copy.
- Do not say that `to(dtype=..., device=...)` cannot move tensors across devices. It can return a tensor on the requested device and/or dtype.
- If a user asks why a NumPy-backed tensor changes when the array changes, explain shared storage.

# Example distinctions

- `torch.tensor(arr)` copies.
- `torch.as_tensor(arr)` may reuse.
- `torch.from_numpy(arr)` shares memory with `arr`.
