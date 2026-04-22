id: requires_grad_and_backward
title: loss.backward and requires_grad
url: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html
source_type: official_docs
tags: backward, requires_grad, autograd, gradients, zero_grad, accumulation, debugging

# Summary

`loss.backward()` computes gradients through the autograd graph and **accumulates** them into the `.grad` attribute of each leaf tensor that requires gradients. The relevant tensors must participate in a differentiable graph with `requires_grad=True`.

# Key facts

- `loss.backward()` computes gradients for leaf tensors that have `requires_grad=True`.
- `loss.backward()` accumulates gradients into `.grad`; it does not overwrite. Call `optimizer.zero_grad()` (or `param.grad = None`) before the next backward pass if you do not want accumulation across steps.
- Parameters that should be optimized must be created or marked with `requires_grad=True`.
- If no tensor in the forward graph requires gradients, autograd has nothing to differentiate and `loss.backward()` cannot produce useful gradients.
- By default the graph is freed after `backward()`. Use `loss.backward(retain_graph=True)` if you need to call backward on the same graph more than once.
- `optimizer.step()` only applies gradients that already exist in `.grad`; it does not build or repair a missing autograd graph.

# Useful assistant behavior

- If the user asks why `loss.backward()` fails when a tensor does not require gradients, explain that autograd needs a graph built from tensors that require gradients.
- If the user asks why their gradients seem to grow across iterations, explain accumulation and recommend `optimizer.zero_grad()` at the start of each step.
- Mention `requires_grad=True` when relevant.
- Do not answer with `.item()` behavior when the question is about backpropagation.
- Do not say `loss.backward()` overwrites `.grad`; it accumulates.
