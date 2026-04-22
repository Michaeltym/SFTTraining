id: train_vs_eval
title: model.train versus model.eval
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
source_type: official_docs
tags: train, eval, mode, module

# Summary

`model.train()` enables training mode, while `model.eval()` enables evaluation mode. These methods only switch module mode; they do not touch autograd.

# Key facts

- `model.train()` switches the module (and all submodules) into training mode.
- `model.eval()` switches the module (and all submodules) into evaluation mode.
- The mode flag is read by stateful modules whose behavior differs between training and inference. The two most common examples are `nn.Dropout` and `nn.BatchNorm*d`, but they use the flag for different reasons. See `dropout_basics.md` and `batchnorm_basics.md` for the actual mechanisms.
- Neither `model.train()` nor `model.eval()` enables or disables gradient tracking. Use `torch.no_grad()` or `torch.inference_mode()` for that.
- Both are real `nn.Module` methods.

# Useful assistant behavior

- If the user asks for the difference between `model.train()` and `model.eval()`, answer with one direct contrast: training mode versus evaluation mode.
- If the user asks how the mode affects dropout or batch normalization, refer to the mechanism in the dedicated files rather than merging them into one bullet.
- Do not say that either method is not a real API.
- Do not say that `model.eval()` disables gradients.

# Common failure to avoid

- Do not describe dropout and batch normalization as if they behave the same way under `eval()`. They share the mode switch, but dropout becomes an identity while batch normalization uses stored running statistics.
