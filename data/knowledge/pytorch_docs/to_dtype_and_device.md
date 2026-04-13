id: to_dtype_and_device
title: Tensor.to for dtype and device conversion
url: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html
source_type: official_docs
tags: to, dtype, device, conversion

# Summary

`Tensor.to(...)` returns a tensor converted to the requested dtype and/or device.

# Key facts

- `.to(dtype=...)` can change tensor dtype.
- `.to(device=...)` can move a tensor to another device.
- `.to(dtype=..., device=...)` can change both dtype and device in one call.
- `.to(...)` returns a tensor; it does not necessarily modify the original tensor in place.

# Useful assistant behavior

- Do not say that `.to(dtype=..., device=...)` cannot move tensors across devices.
- If the user expects in-place behavior, explain that they should reassign the result.

# Example distinctions

- `x = x.to(dtype=torch.float32)`
- `x = x.to(device="cuda")`
