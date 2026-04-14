id: train_vs_eval
title: model.train versus model.eval
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
source_type: official_docs
tags: train, eval, dropout, batchnorm, mode

# Summary

`model.train()` enables training mode, while `model.eval()` enables evaluation mode.

# Key facts

- `model.train()` enables training-time behavior in modules such as dropout and batch normalization.
- `model.eval()` enables evaluation-time behavior in modules such as dropout and batch normalization.
- These methods switch module mode; they do not control autograd.
- Both are real `nn.Module` methods.

# Useful assistant behavior

- If the user asks for the difference between `model.train()` and `model.eval()`, answer with one direct contrast: training mode versus evaluation mode.
- Mention dropout and batch normalization only if needed.
- Do not say that either method is not a real API.
