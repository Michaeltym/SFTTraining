id: requires_grad_and_backward
title: loss.backward and requires_grad
url: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
source_type: official_docs
tags: backward, requires_grad, autograd, gradients, debugging

# Summary

`loss.backward()` computes gradients through the autograd graph, so the relevant tensors must participate in a differentiable graph with `requires_grad=True`.

# Key facts

- `loss.backward()` computes gradients for leaf tensors that require gradients.
- If tensors in the graph do not require gradients, autograd cannot produce gradients for them.
- A tensor with `requires_grad=False` is treated as not requiring gradient computation.
- `optimizer.step()` does not fix a missing autograd graph.

# Useful assistant behavior

- If the user asks why `loss.backward()` fails when a tensor does not require gradients, explain that autograd needs a graph built from tensors that require gradients.
- Mention `requires_grad=True` when relevant.
- Do not answer with `.item()` behavior when the question is about backpropagation.
