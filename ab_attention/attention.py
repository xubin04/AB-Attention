"""AB-Attention and explicit sampling-plan reuse."""
from typing import NamedTuple

import torch

from .kernels import CudaBackend


class SamplePlan(NamedTuple):
    """Selected KV4 block IDs and their inverse-inclusion log2 weights."""

    descriptor: torch.Tensor
    log_weights: torch.Tensor


@torch.no_grad()
def build_plan(q, k, *, sample_pages=64, seed=42):
    """Score and sample ``sample_pages * 128`` keys per 128-query block.

    Q and K have shape [batch, heads, tokens, 128], BF16 dtype, and live on
    the same Hopper GPU. Tokens must be divisible by 4 and exceed the budget.
    Uses Q8 clustering, KV4/K2 pooling and systematic PPS without replacement.
    """
    return SamplePlan(*CudaBackend().sample_all(q, k, sample_pages, seed))


@torch.no_grad()
def apply_plan(q, k, v, plan, *, return_lse=False):
    """Compute weighted attention using current Q/K/V and a sampled plan.

    A plan can be reused across denoising steps with the same tensor layout.
    The optional LSE is the log2 normalizer, for merging attention partitions.
    """
    output, lse = CudaBackend().state(q, k, v, plan.descriptor, plan.log_weights)
    return (output, lse) if return_lse else output


@torch.no_grad()
def ab_attention(q, k, v, *, sample_pages=64, seed=42):
    """AB-Attention with a fresh sampling plan; output has Q's shape and dtype."""
    return apply_plan(q, k, v, build_plan(q, k, sample_pages=sample_pages, seed=seed))
