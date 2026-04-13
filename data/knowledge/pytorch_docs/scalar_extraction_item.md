id: scalar_extraction_item
title: Tensor.item returns a Python scalar only for one-element tensors
url: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
source_type: official_docs
tags: item, scalar, tensor_method, one_element

# Summary

`Tensor.item()` returns the tensor value as a standard Python number.

# Key facts

- `.item()` only works when the tensor has exactly one element.
- The rule depends on total element count, not on tensor rank.
- A tensor with shape `[1]` works because it has one element.
- A tensor with shape `[1, 1]` also works because it still has one element.
- A tensor with shape `[2]` does not work because it has two elements.
- `.item()` is not differentiable.

# Useful assistant behavior

- Never claim that `.item()` is not a real PyTorch method.
- If the user gets an `.item()` error, explain that the tensor has more than one element.
- Suggest reducing or indexing first, for example:
  - `x[0].item()`
  - `x.mean().item()`

# Examples

- `torch.tensor([42]).item()` works.
- `torch.tensor([[42]]).item()` works.
- `torch.tensor([1, 2]).item()` fails because there is more than one element.
