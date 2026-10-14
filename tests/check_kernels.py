#!/usr/bin/env python3
"""Check the CUDA kernels against the PyTorch reference (no model weights).

1. Block scores (Q-centroid x K-pool, weighted mean) vs ``selector_score``.
2. PPS-WOR sampling vs ``pure_wor`` (same scores and random offsets).
3. Weighted descriptor attention vs ``partition_state``.
4. Full-length runs at the model sequence lengths and sampling budgets.
Requires an SM90 (Hopper) GPU and the installed package.
"""
import json
import math

import torch

from ab_attention import ab_attention, apply_plan, build_plan
from ab_attention.reference import partition_state, pure_wor, selector_score
from ab_attention.kernels import CudaBackend

# Video tokens and sampled pages of each model at 720x1280.
FULL_LENGTH = {
    "wan2.1-t2v (81x720x1280)": (21 * 45 * 80, 64),
    "hunyuanvideo (129x720x1280)": (33 * 45 * 80, 71),
}


def random_qkv(heads, tokens, generator):
    shape = (1, heads, tokens, 128)
    return [
        torch.randn(shape, generator=generator).to("cuda", torch.bfloat16)
        for _ in range(3)
    ]


def check_scores(backend, generator):
    q, k, _ = random_qkv(2, 388, generator)  # 388 exercises partial Q blocks
    actual = backend.ops.candidate_group_scores_q8_kv4_k2_cuda(
        q, k, 4, 128, "cluster", "pool", "weighted_mean"
    )
    expected = selector_score(q, k, "weighted_mean").to(actual.device)
    # The kernel scores in bf16 with fast exp2; the reference is float32.
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=1e-3)
    torch.testing.assert_close(actual.sum(-1), torch.ones_like(actual[:, 0]), rtol=0, atol=1e-3)
    return {"shape": list(actual.shape), "max_abs_error": float((actual - expected).abs().max())}


def check_sampling(backend, generator):
    rows, population, pages, seed = 6, 2048, 16, 42
    mass = torch.rand(rows, population, generator=generator).pow(4).cuda()
    mass = mass / mass.sum(-1, keepdim=True)
    support = torch.empty(rows, 0, dtype=torch.int64, device="cuda")
    # Same systematic-sampling offsets as the reference (float64 draw).
    offsets = torch.rand(rows, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    offsets = offsets.float().cuda()
    ids, logs = backend.ops.pure_pps_wor_cuda(mass, support, offsets, pages * 32)
    ref_ids, ref_logs = pure_wor(mass.cpu(), support.cpu()[:, None], pages, seed)
    ref_ids = ref_ids.reshape(rows, -1).long()
    ref_logs = ref_logs.reshape(rows, -1)
    ids = ids.reshape(rows, -1).long().cpu()
    logs = logs.reshape(rows, -1).cpu()
    for row in range(rows):
        if ids[row].unique().numel() != ids.shape[1]:
            raise AssertionError("PPS-WOR returned a duplicate block")
    match = sum(
        len(set(ids[row].tolist()) & set(ref_ids[row].tolist())) for row in range(rows)
    ) / ids.numel()
    if match < 0.99:
        raise AssertionError(f"PPS-WOR sample overlap with the reference is {match:.4f}")
    # Compare inverse-inclusion log2 weights of the blocks both sides sampled.
    errors = []
    for row in range(rows):
        reference = dict(zip(ref_ids[row].tolist(), ref_logs[row].tolist()))
        errors += [abs(w - reference[i]) for i, w in zip(ids[row].tolist(), logs[row].tolist()) if i in reference]
    if max(errors) > 1e-3:
        raise AssertionError(f"PPS-WOR log weights differ by {max(errors):.2e}")
    return {"sample_overlap": match, "max_log2_weight_error": max(errors)}


def check_attention(backend, generator):
    q, k, v = random_qkv(1, 4096, generator)
    plan = build_plan(q, k, sample_pages=16, seed=42)
    output, lse = apply_plan(q, k, v, plan, return_lse=True)
    ref_output, ref_lse = partition_state(q, k, v, *plan)
    torch.testing.assert_close(output.float(), ref_output.float(), atol=0.012, rtol=0.04)
    torch.testing.assert_close(lse.float(), ref_lse.float(), atol=0.025, rtol=0.005)
    torch.testing.assert_close(ab_attention(q, k, v, sample_pages=16), output)
    # Reusing a plan must still consume the new value tensor.
    torch.testing.assert_close(apply_plan(q, k, v * 2, plan), output * 2)
    return {
        "max_output_abs_error": float((output.float() - ref_output.float()).abs().max()),
        "max_lse_abs_error": float((lse.float() - ref_lse.float()).abs().max()),
    }


def check_full_length(backend, generator):
    results = {}
    for name, (tokens, pages) in FULL_LENGTH.items():
        q, k, v = random_qkv(2, tokens, generator)
        descriptor, log_weights = backend.sample_all(q, k, pages, 42)
        output, lse = backend.state(q, k, v, descriptor, log_weights)
        flat = descriptor.reshape(-1, pages * 32).long()
        if not all(row.unique().numel() == row.numel() for row in flat):
            raise AssertionError(f"{name}: duplicate sampled blocks")
        if int(flat.max()) >= tokens // 4 or not torch.isfinite(output).all() or not torch.isfinite(lse).all():
            raise AssertionError(f"{name}: invalid descriptor or non-finite output")
        results[name] = {
            "tokens": tokens,
            "sample_pages": pages,
            "density": pages * 128 / tokens,
            "descriptor": list(descriptor.shape),
        }
    return results


@torch.inference_mode()
def main():
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("an SM90 (Hopper) GPU is required")
    backend = CudaBackend()
    generator = torch.Generator().manual_seed(0)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "block_scores": check_scores(backend, generator),
        "pps_wor": check_sampling(backend, generator),
        "attention": check_attention(backend, generator),
        "full_length": check_full_length(backend, generator),
    }
    print(json.dumps(report, indent=2))
    print("all kernel checks passed")


if __name__ == "__main__":
    main()
