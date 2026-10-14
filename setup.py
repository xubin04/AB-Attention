"""Build the paper kernels with PyTorch's extension tooling."""
import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
CSRC = "ab_attention/kernels/csrc"
os.environ.setdefault("MAX_JOBS", "4")
flags = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3", "-gencode=arch=compute_90a,code=sm_90a",
        "--expt-extended-lambda", "--expt-relaxed-constexpr", "--use_fast_math",
    ],
}

setup(
    ext_modules=[
        CUDAExtension(
            "ab_attention._C",
            sources=[f"{CSRC}/{name}" for name in (
                "ab_ops.cu", "pps_wor_cuda.cu", "selector_cuda.cu",
                "sampled_attention_sm90a.cu",
            )],
            include_dirs=[str(ROOT / path) for path in (
                "third_party/ab", "third_party/ab/cutlass/include",
                "third_party/ab/flashinfer/include",
            )],
            extra_compile_args=flags,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
