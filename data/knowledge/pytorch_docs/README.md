# PyTorch Docs Knowledge Base

This directory stores a small curated knowledge base for the first RAG prototype.

Scope of v1:

- keep the document set intentionally small
- focus on the highest-value PyTorch failure cases seen in evaluation
- prefer official PyTorch documentation pages
- allow a small number of curated notes when official docs do not directly express the desired assistant behavior
- prefer smaller, single-purpose knowledge chunks; avoid bundling unrelated APIs into one file

Current file format:

- plain Markdown
- a small metadata header at the top of each file
- human-editable content

Expected metadata keys:

- `id`
- `title`
- `url`
- `source_type`
- `tags`

The first RAG implementation should treat each file as one knowledge item or split the body into smaller chunks if needed.

## Current files

Tensor construction and memory:

- `as_tensor_vs_tensor.md` — `torch.as_tensor` (possible reuse) vs `torch.tensor` (guaranteed copy)
- `tensor_vs_from_numpy.md` — `torch.tensor` (copy) vs `torch.from_numpy` (shared memory)
- `tensor_views_and_memory.md` — `view`, `reshape`, `permute`, `contiguous`

Shape and reduction:

- `cat_shape_mismatch.md` — `torch.cat` non-concatenated dimension rule
- `cat_vs_stack.md` — existing-dim concatenation vs new-dim stacking
- `stack_shape_reasoning.md` — explicit `torch.stack` shape examples
- `reductions_and_keepdim.md` — `sum` / `mean` / `argmax` with `keepdim`
- `argmax_dim_semantics.md` — `torch.argmax` no-`dim` (flatten, scalar) vs `dim=k` (reduce one axis)

Dtype, device, scalars:

- `dtype_check.md` — `tensor.dtype` and comparison against `torch.float32`
- `device_mismatch.md` — CPU / CUDA tensor mismatch and `.to(device)`
- `to_dtype_and_device.md` — `Tensor.to` for dtype / device conversion
- `scalar_extraction_item.md` — `Tensor.item` one-element rule

Training modes and autograd:

- `train_vs_eval.md` — `model.train()` vs `model.eval()` (module mode switch only)
- `eval_vs_no_grad.md` — `model.eval()` (module behavior) vs `torch.no_grad()` (autograd)
- `dropout_basics.md` — `nn.Dropout` mechanics: random zero + `1 / (1 - p)` scale in train, identity in eval, no running state
- `batchnorm_basics.md` — `nn.BatchNorm*d` mechanics: batch stats + running-stats update in train, stored running stats in eval
- `requires_grad_and_backward.md` — `loss.backward()` and `requires_grad=True`

Fake-API refusals:

- `fake_api_refusal_rules.md` — general refusal policy
- `fake_api_nn_superlayer.md`
- `fake_api_torch_memory_portal.md`
- `fake_api_torch_quantum_backprop.md`
- `fake_api_torch_safe_backward.md`
