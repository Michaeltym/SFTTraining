id: fake_api_refusal_rules
title: Refuse nonexistent public PyTorch APIs
url: https://docs.pytorch.org/docs/stable/index.html
source_type: curated
tags: refusal, fake_api, hallucination, guardrail

# Summary

If a user asks about a nonexistent public PyTorch API, reject the premise directly instead of inventing behavior.

# Core rule

- If the API is not a standard public PyTorch API, say so clearly.
- Do not fabricate arguments, return values, side effects, or examples.
- Do not reinterpret the fake API as if it belongs to some obscure PyTorch subpackage.
- Do not treat an obviously fake built-in API as a custom layer or custom helper unless the user explicitly says they wrote it themselves.

# Useful assistant behavior

- Prefer a short direct refusal first.
- Stop after the refusal unless the user explicitly asks for a real alternative.

# Failures to avoid

- Do not answer with forum-style speculation.
- Do not mix several fake APIs into one answer.
- Do not redirect to another fake API.
- Do not volunteer nearby real APIs unless the user explicitly asks for one.
