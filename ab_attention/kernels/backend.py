"""Binding for the Hopper AB-Attention kernels."""
import importlib
import math

import torch

# Bound the selector's temporary [heads, Q-blocks, Q-centroids, KV4-blocks]
# LSE tensor while retaining enough heads per launch for Hopper occupancy.
_LSE_BUDGET_BYTES = 1536 * 1024 * 1024


def extension_available():
    try:
        importlib.import_module("ab_attention._C")
    except (ImportError, OSError):
        return False
    return True


class CudaBackend:
    def __init__(self):
        self.ops = importlib.import_module("ab_attention._C")

    def sample_all(self, q, k, pages, seed):
        if q.ndim != 4 or q.shape != k.shape or q.shape[2] % 4:
            raise ValueError("Q/K must match [B,H,N,D] with N divisible by 4")
        batch_heads = q.shape[0] * q.shape[1]
        tokens, dim = q.shape[2:]
        q_blocks = math.ceil(tokens / 128)
        rows = batch_heads * q_blocks
        samples = pages * 32
        if pages < 1 or pages * 128 >= tokens:
            raise ValueError("sampled key count must be positive and smaller than N")

        # Generate all row offsets before chunking so launch sizes cannot
        # change the seeded sampling sequence.
        generator = torch.Generator(device=q.device).manual_seed(seed)
        offsets = torch.rand(rows, device=q.device, generator=generator)
        flat_q = q.reshape(batch_heads, tokens, dim)
        flat_k = k.reshape(batch_heads, tokens, dim)
        lse_bytes_per_head = q_blocks * 8 * (tokens // 4) * 2
        heads_per_chunk = max(1, min(batch_heads, _LSE_BUDGET_BYTES // lse_bytes_per_head))
        sampled = torch.empty((rows, samples), dtype=torch.int32, device=q.device)
        log_weights = torch.empty((rows, samples), dtype=torch.float32, device=q.device)

        for start in range(0, batch_heads, heads_per_chunk):
            stop = min(batch_heads, start + heads_per_chunk)
            first, last = start * q_blocks, stop * q_blocks
            scores = self.ops.candidate_group_scores_q8_kv4_k2_cuda(
                flat_q[start:stop].unsqueeze(0).contiguous(),
                flat_k[start:stop].unsqueeze(0).contiguous(),
                4, 128, "cluster", "pool", "weighted_mean",
            )
            empty = torch.empty((last - first, 0), dtype=torch.int64, device=q.device)
            ids, logs = self.ops.pure_pps_wor_cuda(
                scores.reshape(last - first, -1).float().contiguous(),
                empty, offsets[first:last], samples,
            )
            sampled[first:last].copy_(ids)
            log_weights[first:last].copy_(logs)
        return (sampled.reshape(batch_heads, q_blocks, pages, 32),
                log_weights.reshape(rows, pages, 32))

    def state(self, q, k, v, descriptor, log_weights):
        pages = descriptor.shape[-2]
        return self.ops.descriptor_attention_state_wgmma_weighted_kv4_sm90a_cuda(
            q.contiguous(), k.contiguous(), v.contiguous(),
            descriptor.reshape(-1, pages, 32).to(torch.int32).contiguous(),
            log_weights.reshape(-1, pages, 32).float().contiguous(),
        )
