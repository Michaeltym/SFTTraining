id: dropout_basics
title: nn.Dropout mechanics and train/eval behavior
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html
source_type: official_docs
tags: dropout, nn.Dropout, train, eval, regularization

# Summary

`nn.Dropout(p)` randomly zeros elements of its input with probability `p` during training, and is a no-op (identity) during evaluation.

# Key facts

- During training, each element of the input is independently zeroed with probability `p`.
- The surviving elements are scaled by `1 / (1 - p)` so that the expected value of the output matches the input.
- During evaluation (`model.eval()`), `nn.Dropout` is an identity function; it does not zero anything and does not scale.
- `nn.Dropout` has no learnable parameters.
- `nn.Dropout` has no running statistics. It does not track any state across batches.
- `nn.Dropout` does not change the shape or dtype of its input.
- `F.dropout(x, p=..., training=...)` is the functional form; it needs the `training` flag passed explicitly because it has no module state to read.

# Useful assistant behavior

- If the user asks what `nn.Dropout` does in `train()` vs `eval()` mode, answer with the direct contrast: active with scaling in train, identity in eval.
- Do not describe dropout as using "running statistics" or a "running estimate"; those belong to batch normalization, not dropout.
- Do not claim dropout changes tensor shape.
- Do not claim dropout is disabled by `torch.no_grad()`; `torch.no_grad()` only disables autograd, not module mode.
- Keep the mechanism description concrete: zero with probability `p`, scale survivors by `1 / (1 - p)` in train; identity in eval.

# Common failure to avoid

- Do not say dropout uses a running mean or running variance.
- Do not say dropout accumulates statistics across batches.
- Do not conflate dropout's train/eval switch with batch normalization's train/eval switch; they share the switch but not the mechanism (see `batchnorm_basics.md`).
