# AB-Attention

**Accurate Block Attention for Accelerating Video Diffusion Transformers**

Bin Xu, Mo Zhang, Zhengfeiyang Zhang, Xinfeng Yang, Yixing Ma, Wei Wang,
HeXin Wang, Pengfei Zhou, Wangbo Zhao

[**Project page**](https://xubin04.github.io/AB-Attention/) ·
[**Paper**](https://xubin04.github.io/AB-Attention/static/papers/ab-attention.pdf) ·
[**Video comparisons**](https://xubin04.github.io/AB-Attention/#comparisons)

Core implementation of AB-Attention: intra-block clustering selection,
systematic sampling with inclusion-probability weighting, and Hopper
sparse attention kernels.

## Installation

Requires Linux, Python 3.10+, PyTorch, and a CUDA toolkit compatible with
PyTorch. The build configuration used here is PyTorch 2.6.0, CUDA 12.4,
GCC 12. Kernels target **Hopper GPUs** (H20/H100/GH200), BF16 inputs and
head dimension 128. Inference only.

```bash
# In an environment with PyTorch and CUDA installed:
pip install setuptools wheel ninja
pip install --no-build-isolation .
```

The required CUTLASS/FlashInfer/FlashAttention headers are included.
No additional runtime packages or Git submodules are needed. If several
CUDA toolkits are installed, select one with `CUDA_HOME` before building.

## Usage

```python
import torch
from ab_attention import ab_attention

q, k, v = [torch.randn(1, 8, 16384, 128, device="cuda", dtype=torch.bfloat16)
           for _ in range(3)]
out = ab_attention(q, k, v, sample_pages=64, seed=42)
```

Inputs and output have shape `[batch, heads, tokens, 128]`.
`sample_pages` controls the key budget: each 128-token query block attends
to `sample_pages × 128` selected keys. The token count must be divisible by
4 and exceed the selected key count. Compiled budgets are 16, 32, 48, 52,
56, 60, 64, 71, 79, 87, 95, 103, 111, 127, 128 and 159 pages.

To reuse a sampling plan across diffusion steps:

```python
from ab_attention import build_plan, apply_plan

plan = build_plan(q, k, sample_pages=64, seed=42)
out = apply_plan(q, k, v, plan)
# Reuse with updated Q/K/V of the same shape.
out, log2_lse = apply_plan(q_next, k_next, v_next, plan, return_lse=True)
```

The paper uses a four-step reuse interval, 64 pages for Wan2.1 and 71 for
HunyuanVideo. Dense warm-up steps, prompt attention and model-specific
scheduling belong in the caller.

## Example and checks

```bash
python examples/attention.py
python tests/check_kernels.py
```

`tests/check_kernels.py` compares scoring, sampling and weighted attention
against the PyTorch reference, and checks the paper's sequence lengths.
Small CPU reference tests: `pip install pytest`, then `python -m pytest`.

Core code is in `ab_attention/attention.py` and `ab_attention/kernels/`;
the readable reference is `ab_attention/reference.py`.
Third-party sources and licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
