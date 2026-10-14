"""Readable PyTorch references for IBCS, SIP and weighted attention."""
from __future__ import annotations

import math
import torch


def _centroids(x: torch.Tensor, count: int):
    mean = x.mean(0)
    best = ((x - mean) ** 2).sum(-1)
    centers = []
    for _ in range(min(count, len(x))):
        index = int(best.argmax())
        centers.append(index)
        best = torch.minimum(best, ((x - x[index]) ** 2).sum(-1))
    selected = x[torch.tensor(centers, device=x.device)]
    assign = ((x[:, None] - selected[None]) ** 2).sum(-1).argmin(-1)
    populations = torch.stack([(assign == i).sum() for i in range(len(centers))])
    values = torch.stack([
        x[assign == i].mean(0) if populations[i] else torch.zeros_like(mean)
        for i in range(len(centers))
    ])
    return values, populations


def _regular_pool(x: torch.Tensor, count: int):
    """Contiguous, nearly equal groups with their exact populations."""
    boundaries = [index * len(x) // count for index in range(count + 1)]
    groups = [x[boundaries[index] : boundaries[index + 1]] for index in range(count)]
    values = torch.stack([
        group.mean(0) if len(group) else torch.zeros_like(x[0])
        for group in groups
    ])
    populations = torch.tensor(
        [len(group) for group in groups], device=x.device, dtype=torch.int64
    )
    return values, populations


def selector_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    include_simple=False,
    query_representation="cluster",
    key_representation="pool",
):
    """Return population mean/RMS and optionally equal-representative mean."""
    if q.ndim != 4 or q.shape != k.shape or q.shape[2] % 4:
        raise ValueError("Q/K must match [B,H,N,D] and N must divide KV4")
    batch, heads, tokens, dim = q.shape
    query_blocks = math.ceil(tokens / 128)
    fine_blocks = tokens // 4
    means = torch.empty((batch * heads * query_blocks, fine_blocks), device=q.device, dtype=torch.float32)
    rms = torch.empty_like(means)
    simple_means = torch.empty_like(means) if include_simple else None
    flat_q, flat_k = q.reshape(batch * heads, tokens, dim).float(), k.reshape(batch * heads, tokens, dim).float()
    query_reducer = _centroids if query_representation == "cluster" else _regular_pool
    key_reducer = _centroids if key_representation == "cluster" else _regular_pool
    if query_representation not in {"cluster", "pool"}:
        raise ValueError(f"unsupported query representation: {query_representation}")
    if key_representation not in {"cluster", "pool"}:
        raise ValueError(f"unsupported key representation: {key_representation}")
    for row in range(batch * heads):
        key_reps, key_pops = [], []
        for block in flat_k[row].reshape(fine_blocks, 4, dim):
            reps, pops = key_reducer(block, 2)
            key_reps.append(reps)
            key_pops.append(pops)
        key_reps = torch.stack(key_reps)
        key_pops = torch.stack(key_pops).float()
        for block_index in range(query_blocks):
            query_block = flat_q[row, block_index * 128:min((block_index + 1) * 128, tokens)]
            query_reps, query_pops = query_reducer(query_block, 8)
            query_pops = query_pops.float()
            logits = torch.einsum("rd,fcd->rfc", query_reps, key_reps) / math.sqrt(dim)
            logits = logits + key_pops.log()[None]
            probabilities = logits.logsumexp(-1).softmax(-1)
            index = row * query_blocks + block_index
            means[index] = (probabilities * query_pops[:, None]).sum(0) / query_pops.sum()
            rms[index] = ((probabilities.square() * query_pops[:, None]).sum(0) / query_pops.sum()).sqrt()
            if include_simple:
                simple_means[index] = probabilities[query_pops > 0].mean(0)
    return (means, rms, simple_means) if include_simple else (means, rms)


def selector_score(
    q: torch.Tensor,
    k: torch.Tensor,
    aggregation: str,
    *,
    query_representation="cluster",
    key_representation="pool",
):
    """Select one matched aggregation over the same Q8 representatives."""
    mean, rms, simple_mean = selector_scores(
        q,
        k,
        include_simple=True,
        query_representation=query_representation,
        key_representation=key_representation,
    )
    scores = {
        "weighted_rms": rms,
        "weighted_mean": mean,
        "simple_mean": simple_mean,
    }
    try:
        return scores[aggregation]
    except KeyError as error:
        raise ValueError(f"unsupported score aggregation: {aggregation}") from error


def pure_wor(mass, support, pages, seed, *, groups_per_page=32):
    """Systematic PPS without replacement with per-sample log weights."""
    rows = support.shape[0] * support.shape[1]
    mass = mass.reshape(rows, -1).double().clamp_min(1e-30)
    support_flat = support.reshape(rows, -1).long()
    samples = int(pages) * groups_per_page
    if support_flat.shape[1] + samples >= mass.shape[1]:
        raise ValueError("sample count must fit inside the population")
    mass = mass.clone()
    if support_flat.shape[1]:
        mass.scatter_(1, support_flat, 0)
    result, weights = [], []
    generator = torch.Generator(device=mass.device).manual_seed(int(seed))
    offsets = torch.rand(rows, generator=generator, device=mass.device, dtype=torch.float64)
    for row in range(rows):
        values = mass[row]
        positive = values[values > 0]
        lo, hi = 0.0, float(samples / positive.min())
        for _ in range(80):
            mid = (lo + hi) / 2
            if torch.minimum(values * mid, torch.ones_like(values)).sum() < samples:
                lo = mid
            else:
                hi = mid
        inclusion = torch.minimum(values * hi, torch.ones_like(values))
        certain = torch.nonzero(inclusion >= 1 - 1e-10).flatten()
        residual = inclusion.clone()
        residual[certain] = 0
        target = samples - len(certain)
        chosen = torch.empty(0, dtype=torch.long, device=mass.device)
        if target:
            cdf = residual.cumsum(0)
            thresholds = offsets[row] + torch.arange(target, device=mass.device, dtype=torch.float64)
            chosen = torch.searchsorted(cdf, thresholds, right=True)
        ids = torch.cat((certain, chosen)).long()
        result.append(ids)
        weights.append(-inclusion[ids].log2().float())
    ids, logs = torch.stack(result), torch.stack(weights)
    return ids.reshape(*support.shape[:2], pages, groups_per_page).int(), logs.reshape(rows, pages, groups_per_page)


def partition_state(q, k, v, desc, log_weights=None):
    """Readable weighted descriptor attention fallback."""
    batch, heads, tokens, dim = q.shape
    rows, query_blocks = batch * heads, math.ceil(tokens / 128)
    qf, kf, vf = q.reshape(rows, tokens, dim).float(), k.reshape(rows, tokens, dim).float(), v.reshape(rows, tokens, dim).float()
    out = torch.empty_like(qf)
    lse = torch.empty((rows, tokens), device=q.device, dtype=torch.float32)
    descriptor = desc.reshape(rows, query_blocks, -1)
    weights = None if log_weights is None else log_weights.reshape(rows, query_blocks, -1)
    for row in range(rows):
        for block in range(query_blocks):
            query_slice = slice(block * 128, min((block + 1) * 128, tokens))
            ids = descriptor[row, block].long()
            token_ids = (ids[:, None] * 4 + torch.arange(4, device=q.device)[None]).reshape(-1)
            scores = qf[row, query_slice] @ kf[row, token_ids].T / math.sqrt(dim)
            if weights is not None:
                scores = scores + weights[row, block].repeat_interleave(4)[None] * math.log(2)
            out[row, query_slice] = scores.softmax(-1) @ vf[row, token_ids]
            lse[row, query_slice] = scores.logsumexp(-1) / math.log(2)
    return out.reshape_as(q).to(q.dtype), lse.reshape(batch, heads, tokens)
