# Third-party headers

The core kernels include the required headers from these upstream revisions.
The source files and their license notices are retained unchanged.

| Project | Revision | License |
| --- | --- | --- |
| [CUTLASS](https://github.com/NVIDIA/cutlass) | `ae6bccf341fb4410241f696ba06873023d5ce4ed` | [BSD-3-Clause](third_party/ab/cutlass/LICENSE.txt) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | `f6717ff6bc6061c4eb0474576746ee1b42bd6325` | [Apache-2.0](third_party/ab/flashinfer/LICENSE) |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` | [BSD-3-Clause](third_party/ab/flash-attention/LICENSE) |

These are build-time headers under `third_party/ab/`; installing their Python
packages is not required.
