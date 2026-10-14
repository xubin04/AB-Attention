import torch

from ab_attention.reference import partition_state, pure_wor


def test_full_descriptor_matches_dense():
    gen = torch.Generator().manual_seed(17)
    q, k, v = [torch.randn(1, 1, 256, 8, generator=gen) for _ in range(3)]
    descriptor = torch.arange(64).repeat(1, 2, 1).reshape(1, 2, 2, 32)
    result, lse = partition_state(q, k, v, descriptor)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(result, expected)
    expected_lse = (q @ k.transpose(-1, -2) / 8**0.5).logsumexp(-1) / torch.log(torch.tensor(2.0))
    torch.testing.assert_close(lse, expected_lse)


def test_pps_unique_and_reproducible():
    mass = torch.arange(1, 65, dtype=torch.float32).reshape(1, -1)
    support = torch.empty(1, 1, 0, dtype=torch.int64)
    ids, weights = pure_wor(mass, support, pages=1, seed=42)
    ids2, weights2 = pure_wor(mass, support, pages=1, seed=42)
    assert ids.unique().numel() == 32
    assert torch.all(weights >= 0) and torch.isfinite(weights).all()
    torch.testing.assert_close(ids, ids2)
    torch.testing.assert_close(weights, weights2)
