id: as_tensor_vs_tensor
title: torch.as_tensor versus torch.tensor
url: https://docs.pytorch.org/docs/stable/generated/torch.as_tensor.html
source_type: official_docs
tags: as_tensor, tensor, copy, reuse, numpy

# Summary

`torch.as_tensor(...)` may reuse existing data, while `torch.tensor(...)` creates a new tensor copy.

# Key facts

- `torch.tensor(...)` always copies input data into a new tensor.
- `torch.as_tensor(...)` reuses the underlying storage when the input is already a tensor or a NumPy array with matching dtype and device. In that case no copy is made and the two objects share memory.
- `torch.as_tensor(...)` falls back to copying when a conversion is needed, for example when dtype or device differs, or when the input is a Python list.
- Because `torch.as_tensor(...)` can share memory, modifying the source NumPy array afterwards can change the tensor.

# Useful assistant behavior

- If the user asks for the difference between `torch.as_tensor` and `torch.tensor`, explain possible reuse vs guaranteed copy.
- If the user asks when `torch.as_tensor` actually shares memory, mention that it requires a matching-dtype, matching-device NumPy array or existing tensor.
- Do not claim that `torch.as_tensor(...)` always copies.
- Do not claim that `torch.tensor(...)` reuses input storage.
