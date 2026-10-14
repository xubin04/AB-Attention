"""Minimal Hopper example; run after installing the package."""
import torch

from ab_attention import ab_attention, apply_plan, build_plan

q, k, v = [torch.randn(1, 2, 4096, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3)]

# Fresh selection and attention in one call.
out = ab_attention(q, k, v, sample_pages=16)
print(out.shape)  # [1, 2, 4096, 128]

# Reuse the same selection with current Q/K/V at a later diffusion step.
plan = build_plan(q, k, sample_pages=16)
out, log2_lse = apply_plan(q, k, v, plan, return_lse=True)
print(out.shape, log2_lse.shape)
