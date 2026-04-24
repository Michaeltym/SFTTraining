id: batchnorm_basics
title: nn.BatchNorm mechanics and train/eval behavior
url: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
source_type: official_docs
tags: batchnorm, nn.BatchNorm1d, nn.BatchNorm2d, running_mean, running_var, train, eval

# Summary

`nn.BatchNorm*d` normalizes its input using batch statistics during training and using stored running statistics during evaluation. This train/eval asymmetry is intrinsic to batch normalization and is the reason `model.eval()` matters for BN.

# Key facts

- During training, BN normalizes each feature using the mean and variance of the current batch.
- During training, BN also updates its internal `running_mean` and `running_var` buffers as an exponential moving average of per-batch statistics.
- During evaluation (`model.eval()`), BN does not use the current batch statistics. It uses the stored `running_mean` and `running_var` instead.
- BN has two kinds of state: learnable affine parameters (`weight`, `bias`) and non-learnable running statistics buffers (`running_mean`, `running_var`, `num_batches_tracked`).
- `track_running_stats=False` disables the running-statistics branch; in that mode BN uses batch statistics in both train and eval.
- BN's train/eval asymmetry is the reason inference accuracy can degrade sharply if you forget to call `model.eval()` and the per-batch statistics do not represent the deployment distribution.

# Useful assistant behavior

- If the user asks what `nn.BatchNorm2d` does in `train()` vs `eval()` mode, answer with the direct contrast: batch statistics in train (and update running stats), stored running statistics in eval.
- When describing eval-time behavior, be explicit that BN reads `running_mean` / `running_var`, not dropout-style identity.
- Do not claim BN is an identity function in eval; that is dropout, not BN.
- Do not claim BN has no running statistics; running statistics are core to BN's eval behavior.

# Common failure to avoid

- Do not say BN "is disabled" in eval mode; it still normalizes, just using stored statistics.
- Do not conflate BN with dropout. They share the `train()` / `eval()` switch, but the mechanism is different (see `dropout_basics.md`).
- Do not claim `torch.no_grad()` alone makes BN use running statistics; only `model.eval()` does that.
