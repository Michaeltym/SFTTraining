id: device_mismatch
title: Device mismatch errors across cpu and cuda tensors
url: https://docs.pytorch.org/docs/stable/tensor_attributes.html
source_type: official_docs
tags: device, cpu, cuda, mismatch, debugging

# Summary

PyTorch operations usually require tensors involved in the same operation to be on compatible devices.

# Key facts

- A common device mismatch error happens when one tensor is on CPU and another is on CUDA.
- PyTorch does not automatically move tensors across devices for you in normal tensor operations.
- The recommended fix is to move tensors to the same device with `.to(device)`; `.cuda()` and `.cpu()` still work but are older, less flexible forms.
- Model parameters, inputs, and targets used together should generally be on the same device.

# Useful assistant behavior

- If the user asks about CPU versus CUDA mismatch, say that the tensors must be moved to the same device.
- Mention `.to(device)` as the usual fix.
- Do not answer with dtype advice when the question is clearly about device mismatch.
