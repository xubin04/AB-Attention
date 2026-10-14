#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include "ab_ops.h"

namespace {
constexpr int kThreads = 512;

__global__ void pps_wor_kernel(
    const float* masses, const int64_t* support, const float* offsets,
    int64_t* sampled, float* weights, int population, int retained, int samples) {
  using Reduce = cub::BlockReduce<double, kThreads>;
  using CountScan = cub::BlockScan<int, kThreads>;
  using MassScan = cub::BlockScan<double, kThreads>;
  union Scratch {
    Reduce::TempStorage reduce;
    CountScan::TempStorage count;
    MassScan::TempStorage mass;
  };
  __shared__ Scratch scratch;
  __shared__ double scale;
  __shared__ double prefix;
  __shared__ int certainty;
  __shared__ int new_certainty;
  __shared__ int certainty_prefix;
  __shared__ int last_residual;
  __shared__ int certainty_pass;
  __shared__ double cumulative_tile[kThreads + 1];
  extern __shared__ float values[];
  const int lane = threadIdx.x;
  const int64_t row = blockIdx.x;
  const float* input = masses + row * population;
  int64_t* output = sampled + row * samples;
  float* logs = weights + row * samples;
  for (int index = lane; index < population; index += kThreads)
    values[index] = fmaxf(input[index], 1e-30f);
  if (lane == 0) certainty = 0;
  __syncthreads();
  for (int index = lane; index < retained; index += kThreads)
    values[support[row * retained + index]] = 0.;
  __syncthreads();
  do {
    double local_sum = 0.;
    double local_max = 0.;
    for (int index = lane; index < population; index += kThreads) {
      if (values[index] > 0.f) {
        const double value = fmax(static_cast<double>(values[index]), 1e-30);
        local_sum += value;
        local_max = fmax(local_max, value);
      }
    }
    const double total = Reduce(scratch.reduce).Sum(local_sum);
    // CUB reductions reuse shared scratch; all threads must finish Sum first.
    __syncthreads();
    const double maximum = Reduce(scratch.reduce).Reduce(local_max, cub::Max());
    if (lane == 0) {
      scale = (samples - certainty) / total;
      new_certainty = 0;
      // A second full population pass is needed only when at least one
      // unit can become certain.  If max(mass) * scale < 1, all inclusion
      // probabilities are strictly fractional and the marking pass is
      // provably empty.
      certainty_pass = maximum * scale >= 1.0;
    }
    __syncthreads();
    if (certainty_pass) {
      int count = 0;
      for (int index = lane; index < population; index += kThreads) {
        if (values[index] > 0.f && fmax(static_cast<double>(values[index]), 1e-30) * scale >= 1.) {
          values[index] = -values[index];
          ++count;
        }
      }
      atomicAdd(&new_certainty, count);
    }
    __syncthreads();
    if (lane == 0) certainty += new_certainty;
    __syncthreads();
  } while (new_certainty != 0);
  if (lane == 0) {
    prefix = 0.;
    certainty_prefix = 0;
    last_residual = population - 1;
    while (last_residual >= 0 && values[last_residual] <= 0.) --last_residual;
  }
  __syncthreads();
  for (int tile = 0; tile < population; tile += kThreads) {
    const int index = tile + lane;
    const double mass = index < population ? values[index] : 0.;
    const int certain = mass < 0.;
    int rank = 0;
    CountScan(scratch.count).InclusiveSum(certain, rank);
    if (certain) {
      const int slot = certainty_prefix + rank - 1;
      output[slot] = index;
      logs[slot] = 0.f;
    }
    __syncthreads();
    if (lane == kThreads - 1) certainty_prefix += rank;
    __syncthreads();
    double cumulative = 0.;
    MassScan(scratch.mass).InclusiveSum((mass > 0. ? fmax(mass, 1e-30) : 0.) * scale, cumulative);
    const double base = prefix;
    if (lane == 0) cumulative_tile[0] = base;
    cumulative_tile[lane + 1] = index >= last_residual ? samples - certainty : base + cumulative;
    __syncthreads();
    const double offset = static_cast<double>(offsets[row]);
    const int first_slot = max(0, static_cast<int>(ceil(base - offset)));
    const int end_slot = min(samples - certainty, static_cast<int>(ceil(cumulative_tile[kThreads] - offset)));
    for (int slot = first_slot + lane; slot < end_slot; slot += kThreads) {
      const double threshold = offset + slot;
      int lower = 0, upper = min(kThreads, population - tile);
      while (lower < upper) {
        const int middle = lower + (upper - lower) / 2;
        if (cumulative_tile[middle + 1] <= threshold) lower = middle + 1;
        else upper = middle;
      }
      const int selected = tile + lower;
      output[certainty + slot] = selected;
      logs[certainty + slot] = static_cast<float>(-log2(fmax(static_cast<double>(input[selected]), 1e-30) * scale));
    }
    __syncthreads();
    if (lane == kThreads - 1) prefix = cumulative_tile[kThreads];
    __syncthreads();
  }
}
}

std::vector<torch::Tensor> pure_pps_wor_cuda(
    torch::Tensor masses, torch::Tensor support, torch::Tensor offsets, int64_t samples) {
  TORCH_CHECK(masses.is_cuda() && support.device() == masses.device() && offsets.device() == masses.device(), "inputs must share a CUDA device");
  TORCH_CHECK(masses.scalar_type() == torch::kFloat32 && support.scalar_type() == torch::kInt64 && offsets.scalar_type() == torch::kFloat32, "expected float32 mass/offset and int64 support");
  TORCH_CHECK(masses.dim() == 2 && support.dim() == 2 && offsets.dim() == 1, "expected mass [rows,N], support [rows,S], offsets [rows]");
  TORCH_CHECK(masses.is_contiguous() && support.is_contiguous() && offsets.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(masses.size(0) == support.size(0) && masses.size(0) == offsets.size(0), "row counts must agree");
  TORCH_CHECK(samples > 0 && samples < masses.size(1) - support.size(1), "sample count must be inside omitted population");
  c10::cuda::CUDAGuard guard(masses.device());
  const size_t bytes = masses.size(1) * sizeof(float);
  cudaFuncAttributes attributes;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, pps_wor_kernel));
  int limit;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, masses.get_device()));
  TORCH_CHECK(bytes + attributes.sharedSizeBytes <= static_cast<size_t>(limit), "population exceeds single-block shared memory capacity");
  C10_CUDA_CHECK(cudaFuncSetAttribute(pps_wor_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  auto ids = torch::empty({masses.size(0), samples}, support.options());
  auto logs = torch::empty({masses.size(0), samples}, masses.options());
  pps_wor_kernel<<<masses.size(0), kThreads, bytes, at::cuda::getCurrentCUDAStream()>>>(
      masses.data_ptr<float>(), support.data_ptr<int64_t>(), offsets.data_ptr<float>(),
      ids.data_ptr<int64_t>(), logs.data_ptr<float>(), masses.size(1), support.size(1), samples);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {ids, logs};
}
