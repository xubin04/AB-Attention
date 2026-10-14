#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "ab_ops.h"
#include "flashinfer/permuted_smem.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using Bf16 = __nv_bfloat16;
using SwizzledSmem = flashinfer::smem_t<flashinfer::SwizzleMode::k128B>;
using MmaMode = flashinfer::mma::MMAMode;

constexpr int kHeadDim = 128;
constexpr int kBlockSize = 128;
constexpr int kQueryCentroids = 8;
constexpr int kMicroFineSize = 4;
constexpr int kMicroKCentroids = 2;
constexpr int kMicroKBlocksPerTile = 8;
constexpr int kMmaTile = 16;
constexpr int kMmaTilesHeadDim = kHeadDim / kMmaTile;
constexpr int kFragmentElements = 8;
constexpr int kBf16PerVector = sizeof(uint4) / sizeof(Bf16);
constexpr int kVectorsPerRow = kHeadDim / kBf16PerVector;
constexpr int kStageElements = kMmaTile * kHeadDim;
constexpr size_t kStageBytes = kStageElements * sizeof(Bf16);
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kMassWarps = kQueryCentroids;
constexpr int kMassThreads = kMassWarps * 32;
constexpr int kQBlocksPerGroup = 2;
constexpr size_t kSharedBytes = 2 * kStageBytes + kWarps * kStageBytes;

static_assert(kQueryCentroids * kQBlocksPerGroup == kMmaTile);
static_assert(kMicroKCentroids * kMicroKBlocksPerTile == kMmaTile);

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, mask));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, mask);
    }
    return value;
}

__device__ __forceinline__ void compute_qk_tile(
    SwizzledSmem k_smem,
    uint32_t (&q_frag)[kMmaTilesHeadDim][4],
    float (&score_frag)[kFragmentElements],
    int lane) {
#pragma unroll
    for (int mma_k = 0; mma_k < kMmaTilesHeadDim; ++mma_k) {
        uint32_t k_frag[4];
        const int row = 8 * (lane / 16) + lane % 8;
        const int vector_col = mma_k * 2 + (lane % 16) / 8;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        k_smem.ldmatrix_m8n8x4(smem_offset, k_frag);
        if (mma_k == 0) {
            flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<Bf16, MmaMode::kInit>(
                score_frag, q_frag[mma_k], k_frag);
        } else {
            flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<Bf16>(
                score_frag, q_frag[mma_k], k_frag);
        }
    }
}

// Block-wide argmax over one value per thread, ties resolved to the lowest
// index. All threads participate so the syncs stay uniform.
__device__ __forceinline__ int block_argmax(
    float value, int index, float* scratch_value, int* scratch_index) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffffU, value, offset);
        const int other_index = __shfl_down_sync(0xffffffffU, index, offset);
        if (other > value) {
            value = other;
            index = other_index;
        }
    }
    const int warp = threadIdx.x / 32;
    if ((threadIdx.x & 31) == 0) {
        scratch_value[warp] = value;
        scratch_index[warp] = index;
    }
    __syncthreads();
    float best_value = scratch_value[0];
    int best = scratch_index[0];
#pragma unroll
    for (int other = 1; other < kWarps; ++other) {
        if (scratch_value[other] > best_value) {
            best_value = scratch_value[other];
            best = scratch_index[other];
        }
    }
    return best;
}

__global__ __launch_bounds__(kHeadDim)
void q_centroid8_pack_kernel(
    const Bf16* __restrict__ q,
    Bf16* __restrict__ packed_q,
    half* __restrict__ populations,
    int64_t seq_len,
    int64_t q_blocks) {
    __shared__ Bf16 queries[kBlockSize * kHeadDim];
    __shared__ float mean[kHeadDim];
    __shared__ float min_distance[kBlockSize];
    __shared__ int assignments[kBlockSize];
    __shared__ int centers[kQueryCentroids + 1];
    __shared__ int counts[kQueryCentroids];
    __shared__ float scratch_value[kWarps];
    __shared__ int scratch_index[kWarps];

    const int dim = threadIdx.x;
    const int warp = dim / 32;
    const int lane = dim & 31;
    const int64_t q_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t token_base = q_block * kBlockSize;
    const int valid_queries = static_cast<int>(
        min(static_cast<int64_t>(kBlockSize), seq_len - token_base));

    float sum = 0.0f;
#pragma unroll
    for (int query = 0; query < kBlockSize; ++query) {
        Bf16 value = __float2bfloat16_rn(0.0f);
        if (query < valid_queries) {
            value = q[(bh * seq_len + token_base + query) * kHeadDim + dim];
            sum += __bfloat162float(value);
        }
        queries[query * kHeadDim + dim] = value;
    }
    mean[dim] = sum / static_cast<float>(valid_queries);
    __syncthreads();

    for (int query = warp; query < valid_queries; query += kWarps) {
        float distance = 0.0f;
        for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
            const float delta =
                __bfloat162float(queries[query * kHeadDim + local_dim]) -
                mean[local_dim];
            distance += delta * delta;
        }
        distance = warp_sum(distance);
        if (lane == 0) {
            min_distance[query] = distance;
        }
    }
    __syncthreads();

    // One query per thread: the farthest-first scan is a block reduction, not a
    // serial walk by thread 0 with the other 127 threads waiting on it.
    centers[0] = block_argmax(
        dim < valid_queries ? min_distance[dim] : -INFINITY, dim,
        scratch_value, scratch_index);
    __syncthreads();
    if (dim < valid_queries) {
        min_distance[dim] = INFINITY;
        assignments[dim] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int center_slot = 0; center_slot < kQueryCentroids; ++center_slot) {
        const int center = centers[center_slot];
        for (int query = warp; query < valid_queries; query += kWarps) {
            float distance = 0.0f;
            for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
                const float delta =
                    __bfloat162float(queries[query * kHeadDim + local_dim]) -
                    __bfloat162float(queries[center * kHeadDim + local_dim]);
                distance += delta * delta;
            }
            distance = warp_sum(distance);
            if (lane == 0 && distance < min_distance[query]) {
                min_distance[query] = distance;
                assignments[query] = center_slot;
            }
        }
        __syncthreads();
        // The extra centers slot lets every iteration take the same path, so
        // the reduction's syncs stay uniform across the block.
        centers[center_slot + 1] = block_argmax(
            dim < valid_queries ? min_distance[dim] : -INFINITY, dim,
            scratch_value, scratch_index);
        __syncthreads();
    }

    if (dim < kQueryCentroids) {
        counts[dim] = 0;
    }
    __syncthreads();
    if (dim < valid_queries) {
        atomicAdd(&counts[assignments[dim]], 1);
    }
    __syncthreads();
    if (dim < kQueryCentroids) {
        populations[(bh * q_blocks + q_block) * kQueryCentroids + dim] =
            __float2half_rn(static_cast<float>(counts[dim]));
    }
    __syncthreads();

    float centroid_sum[kQueryCentroids];
#pragma unroll
    for (int cluster = 0; cluster < kQueryCentroids; ++cluster) {
        centroid_sum[cluster] = 0.0f;
    }
    // Indexing centroid_sum by the assignment would spill it to local memory,
    // so select over the eight clusters instead and keep the sums in registers.
    for (int query = 0; query < valid_queries; ++query) {
        const float value = __bfloat162float(queries[query * kHeadDim + dim]);
        const int cluster = assignments[query];
#pragma unroll
        for (int slot = 0; slot < kQueryCentroids; ++slot) {
            centroid_sum[slot] += (slot == cluster) ? value : 0.0f;
        }
    }
    const int64_t output_base =
        (bh * q_blocks + q_block) * kQueryCentroids * kHeadDim;
#pragma unroll
    for (int cluster = 0; cluster < kQueryCentroids; ++cluster) {
        packed_q[output_base + cluster * kHeadDim + dim] =
            counts[cluster] > 0
            ? __float2bfloat16_rn(
                  centroid_sum[cluster] / static_cast<float>(counts[cluster]))
            : __float2bfloat16_rn(0.0f);
    }
}


__global__ __launch_bounds__(kHeadDim)
void q_pool8_pack_kernel(
    const Bf16* __restrict__ q,
    Bf16* __restrict__ packed_q,
    half* __restrict__ populations,
    int64_t seq_len,
    int64_t q_blocks) {
    const int dim = threadIdx.x;
    const int64_t q_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t token_base = q_block * kBlockSize;
    const int valid_queries = static_cast<int>(
        min(static_cast<int64_t>(kBlockSize), seq_len - token_base));
    const int64_t output_base =
        (bh * q_blocks + q_block) * kQueryCentroids * kHeadDim;
#pragma unroll
    for (int representative = 0; representative < kQueryCentroids;
         ++representative) {
        const int begin = representative * valid_queries / kQueryCentroids;
        const int end = (representative + 1) * valid_queries / kQueryCentroids;
        const int count = end - begin;
        float sum = 0.0f;
        for (int query = begin; query < end; ++query) {
            sum += __bfloat162float(
                q[((bh * seq_len + token_base + query) * kHeadDim) + dim]);
        }
        packed_q[output_base + representative * kHeadDim + dim] =
            count > 0
            ? __float2bfloat16_rn(sum / static_cast<float>(count))
            : __float2bfloat16_rn(0.0f);
        if (dim == 0) {
            populations[
                (bh * q_blocks + q_block) * kQueryCentroids + representative] =
                __float2half_rn(static_cast<float>(count));
        }
    }
}


__global__ __launch_bounds__(kHeadDim)
void kv4_k2_pack_kernel(
    const Bf16* __restrict__ k,
    Bf16* __restrict__ packed_k,
    half* __restrict__ log_counts,
    int64_t fine_blocks) {
    __shared__ Bf16 keys[kMicroFineSize * kHeadDim];
    __shared__ float mean[kHeadDim];
    __shared__ float min_distance[kMicroFineSize];
    __shared__ int assignments[kMicroFineSize];
    __shared__ int centers[kMicroKCentroids];
    __shared__ int counts[kMicroKCentroids];

    const int dim = threadIdx.x;
    const int warp = dim / 32;
    const int lane = dim & 31;
    const int64_t fine_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t source_base =
        (bh * fine_blocks + fine_block) * kMicroFineSize * kHeadDim;

    float sum = 0.0f;
#pragma unroll
    for (int key = 0; key < kMicroFineSize; ++key) {
        const Bf16 value = k[source_base + key * kHeadDim + dim];
        keys[key * kHeadDim + dim] = value;
        sum += __bfloat162float(value);
    }
    mean[dim] = sum * (1.0f / static_cast<float>(kMicroFineSize));
    __syncthreads();

    if (warp < kMicroFineSize) {
        float distance = 0.0f;
        for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
            const float delta =
                __bfloat162float(keys[warp * kHeadDim + local_dim]) -
                mean[local_dim];
            distance += delta * delta;
        }
        distance = warp_sum(distance);
        if (lane == 0) {
            min_distance[warp] = distance;
        }
    }
    __syncthreads();
    if (dim == 0) {
        int first = 0;
        float farthest = min_distance[0];
        for (int key = 1; key < kMicroFineSize; ++key) {
            if (min_distance[key] > farthest) {
                farthest = min_distance[key];
                first = key;
            }
        }
        centers[0] = first;
#pragma unroll
        for (int key = 0; key < kMicroFineSize; ++key) {
            min_distance[key] = INFINITY;
            assignments[key] = 0;
        }
    }
    __syncthreads();

#pragma unroll
    for (int center_slot = 0; center_slot < kMicroKCentroids; ++center_slot) {
        const int center = centers[center_slot];
        if (warp < kMicroFineSize) {
            float distance = 0.0f;
            for (int local_dim = lane; local_dim < kHeadDim; local_dim += 32) {
                const float delta =
                    __bfloat162float(keys[warp * kHeadDim + local_dim]) -
                    __bfloat162float(keys[center * kHeadDim + local_dim]);
                distance += delta * delta;
            }
            distance = warp_sum(distance);
            if (lane == 0 && distance < min_distance[warp]) {
                min_distance[warp] = distance;
                assignments[warp] = center_slot;
            }
        }
        __syncthreads();
        if (dim == 0 && center_slot + 1 < kMicroKCentroids) {
            int next = 0;
            float farthest = min_distance[0];
            for (int key = 1; key < kMicroFineSize; ++key) {
                if (min_distance[key] > farthest) {
                    farthest = min_distance[key];
                    next = key;
                }
            }
            centers[center_slot + 1] = next;
        }
        __syncthreads();
    }

    if (dim == 0) {
#pragma unroll
        for (int cluster = 0; cluster < kMicroKCentroids; ++cluster) {
            counts[cluster] = 0;
        }
#pragma unroll
        for (int key = 0; key < kMicroFineSize; ++key) {
            ++counts[assignments[key]];
        }
#pragma unroll
        for (int cluster = 0; cluster < kMicroKCentroids; ++cluster) {
            log_counts[
                (bh * fine_blocks + fine_block) * kMicroKCentroids + cluster] =
                __float2half_rn(log2f(static_cast<float>(counts[cluster])));
        }
    }
    __syncthreads();

    float centroid_sum[kMicroKCentroids] = {0.0f, 0.0f};
#pragma unroll
    for (int key = 0; key < kMicroFineSize; ++key) {
        centroid_sum[assignments[key]] +=
            __bfloat162float(keys[key * kHeadDim + dim]);
    }
    const int64_t output_base =
        (bh * fine_blocks + fine_block) * kMicroKCentroids * kHeadDim;
#pragma unroll
    for (int cluster = 0; cluster < kMicroKCentroids; ++cluster) {
        packed_k[output_base + cluster * kHeadDim + dim] =
            counts[cluster] > 0
            ? __float2bfloat16_rn(
                  centroid_sum[cluster] / static_cast<float>(counts[cluster]))
            : __float2bfloat16_rn(0.0f);
    }
}


__global__ __launch_bounds__(kHeadDim)
void kv4_pool2_pack_kernel(
    const Bf16* __restrict__ k,
    Bf16* __restrict__ packed_k,
    half* __restrict__ log_counts,
    int64_t fine_blocks) {
    const int dim = threadIdx.x;
    const int64_t fine_block = blockIdx.x;
    const int64_t bh = blockIdx.y;
    const int64_t source_base =
        (bh * fine_blocks + fine_block) * kMicroFineSize * kHeadDim;
    const int64_t output_base =
        (bh * fine_blocks + fine_block) * kMicroKCentroids * kHeadDim;
#pragma unroll
    for (int representative = 0; representative < kMicroKCentroids;
         ++representative) {
        const int first = representative * 2;
        const float value = 0.5f * (
            __bfloat162float(k[source_base + first * kHeadDim + dim]) +
            __bfloat162float(k[source_base + (first + 1) * kHeadDim + dim]));
        packed_k[output_base + representative * kHeadDim + dim] =
            __float2bfloat16_rn(value);
        if (dim == 0) {
            log_counts[
                (bh * fine_blocks + fine_block) * kMicroKCentroids +
                representative] = __float2half_rn(1.0f);
        }
    }
}

__device__ __forceinline__ void load_q_centroid_pair(
    const Bf16* __restrict__ packed_q,
    SwizzledSmem q_smem,
    int lane,
    int64_t bh,
    int64_t q_block_base,
    int64_t q_blocks,
    float qk_scale_log2) {
    constexpr int kTileVectors = kMmaTile * kVectorsPerRow;
    for (int vector_idx = lane; vector_idx < kTileVectors; vector_idx += 32) {
        const int row = vector_idx / kVectorsPerRow;
        const int vector_col = vector_idx - row * kVectorsPerRow;
        const int local_q_block = row / kQueryCentroids;
        const int centroid = row - local_q_block * kQueryCentroids;
        const int64_t q_block = q_block_base + local_q_block;
        uint4 packed = make_uint4(0, 0, 0, 0);
        Bf16* values = reinterpret_cast<Bf16*>(&packed);
        if (q_block < q_blocks) {
            const int64_t source_base =
                ((bh * q_blocks + q_block) * kQueryCentroids + centroid) *
                kHeadDim + vector_col * kBf16PerVector;
#pragma unroll
            for (int elem = 0; elem < kBf16PerVector; ++elem) {
                values[elem] = __float2bfloat16_rn(
                    __bfloat162float(packed_q[source_base + elem]) *
                    qk_scale_log2);
            }
        }
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        q_smem.base[smem_offset] = packed;
    }
}

__device__ __forceinline__ void stage_kv4_k2_octet_async(
    const Bf16* __restrict__ packed_k,
    SwizzledSmem k_smem,
    int thread,
    int64_t bh,
    int fine_block_base,
    int fine_blocks) {
    constexpr int kTileVectors = kMmaTile * kVectorsPerRow;
    for (int vector_idx = thread; vector_idx < kTileVectors;
         vector_idx += kThreads) {
        const int row = vector_idx / kVectorsPerRow;
        const int vector_col = vector_idx - row * kVectorsPerRow;
        const int local_block = row / kMicroKCentroids;
        const int representative = row - local_block * kMicroKCentroids;
        const int fine_block = min(fine_block_base + local_block, fine_blocks - 1);
        const int64_t source_base =
            ((bh * fine_blocks + fine_block) * kMicroKCentroids + representative) *
            kHeadDim + vector_col * kBf16PerVector;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(row, vector_col);
        k_smem.load_128b_async(smem_offset, packed_k + source_base);
    }
}

// Fold one log-sum-exp term into a running (max, denominator) pair.
__device__ __forceinline__ void accumulate_lse(
    float& maximum, float& denominator, float value) {
    const float next_max = fmaxf(maximum, value);
    denominator = denominator * exp2f(maximum - next_max) +
        exp2f(value - next_max);
    maximum = next_max;
}

__global__ __launch_bounds__(kThreads)
void qcentroid8_kv4_k2_lse_kernel(
    const Bf16* __restrict__ packed_q,
    const Bf16* __restrict__ packed_k,
    const half* __restrict__ log_counts,
    half* __restrict__ output,
    float* __restrict__ row_max,
    float* __restrict__ row_den,
    int64_t q_blocks,
    int fine_blocks,
    float qk_scale_log2) {
    extern __shared__ __align__(32) unsigned char smem_raw[];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int64_t bh = blockIdx.y;
    const int64_t q_group = static_cast<int64_t>(blockIdx.x) * kWarps + warp;
    const int64_t q_block_base = q_group * kQBlocksPerGroup;
    unsigned char* warp_q_smem =
        smem_raw + 2 * kStageBytes + warp * kStageBytes;
    SwizzledSmem q_stage(warp_q_smem);
    SwizzledSmem k_stage0(smem_raw);
    SwizzledSmem k_stage1(smem_raw + kStageBytes);

    load_q_centroid_pair(
        packed_q,
        q_stage,
        lane,
        bh,
        q_block_base,
        q_blocks,
        qk_scale_log2);
    __syncwarp();
    uint32_t q_frag[kMmaTilesHeadDim][4];
#pragma unroll
    for (int mma_k = 0; mma_k < kMmaTilesHeadDim; ++mma_k) {
        const int q_row = lane % 16;
        const int vector_col = mma_k * 2 + lane / 16;
        const uint32_t smem_offset =
            SwizzledSmem::get_permuted_offset<kVectorsPerRow>(q_row, vector_col);
        q_stage.ldmatrix_m8n8x4(smem_offset, q_frag[mma_k]);
    }
    __syncwarp();

    stage_kv4_k2_octet_async(
        packed_k, k_stage0, threadIdx.x, bh, 0, fine_blocks);
    flashinfer::cp_async::commit_group();
    const int tile_count =
        (fine_blocks + kMicroKBlocksPerTile - 1) / kMicroKBlocksPerTile;
    // Each lane owns centroid lane/4 for both query blocks of the group and
    // visits every key block of that centroid, so the softmax normalizer is
    // accumulated here instead of in a second pass over the LSE array.
    float max_low = -INFINITY, den_low = 0.0f;
    float max_high = -INFINITY, den_high = 0.0f;
    for (int tile = 0; tile < tile_count; ++tile) {
        const bool use_stage0 = (tile & 1) == 0;
        SwizzledSmem current_stage = use_stage0 ? k_stage0 : k_stage1;
        SwizzledSmem next_stage = use_stage0 ? k_stage1 : k_stage0;
        flashinfer::cp_async::wait_group<0>();
        __syncthreads();
        if (tile + 1 < tile_count) {
            stage_kv4_k2_octet_async(
                packed_k,
                next_stage,
                threadIdx.x,
                bh,
                (tile + 1) * kMicroKBlocksPerTile,
                fine_blocks);
            flashinfer::cp_async::commit_group();
        }

        alignas(16) float score_frag[kFragmentElements];
        compute_qk_tile(current_stage, q_frag, score_frag, lane);

        const int block_low =
            tile * kMicroKBlocksPerTile + (lane & 3);
        const int block_high = block_low + 4;
        const int64_t count_base_low =
            (bh * fine_blocks + min(block_low, fine_blocks - 1)) *
            kMicroKCentroids;
        const int64_t count_base_high =
            (bh * fine_blocks + min(block_high, fine_blocks - 1)) *
            kMicroKCentroids;
        const float log_low0 = __half2float(log_counts[count_base_low]);
        const float log_low1 = __half2float(log_counts[count_base_low + 1]);
        const float log_high0 = __half2float(log_counts[count_base_high]);
        const float log_high1 = __half2float(log_counts[count_base_high + 1]);

        const auto pair_lse = [] __device__ (float left, float right) {
            const float maximum = fmaxf(left, right);
            return maximum + log2f(
                fmaxf(exp2f(left - maximum) + exp2f(right - maximum), 1.0e-20f));
        };
        const float low_q_low_k = pair_lse(
            score_frag[0] + log_low0, score_frag[1] + log_low1);
        const float high_q_low_k = pair_lse(
            score_frag[2] + log_low0, score_frag[3] + log_low1);
        const float low_q_high_k = pair_lse(
            score_frag[4] + log_high0, score_frag[5] + log_high1);
        const float high_q_high_k = pair_lse(
            score_frag[6] + log_high0, score_frag[7] + log_high1);
        const int centroid = lane / 4;
        const int64_t q_block_low = q_block_base;
        const int64_t q_block_high = q_block_base + 1;
        // Round before accumulating so the normalizer matches the stored LSE.
        const half stored_low_low = __float2half_rn(low_q_low_k);
        const half stored_low_high = __float2half_rn(low_q_high_k);
        const half stored_high_low = __float2half_rn(high_q_low_k);
        const half stored_high_high = __float2half_rn(high_q_high_k);
        if (q_block_low < q_blocks) {
            if (block_low < fine_blocks) {
                output[
                    ((bh * q_blocks + q_block_low) * kQueryCentroids + centroid) *
                        fine_blocks +
                    block_low] = stored_low_low;
                accumulate_lse(max_low, den_low, __half2float(stored_low_low));
            }
            if (block_high < fine_blocks) {
                output[
                    ((bh * q_blocks + q_block_low) * kQueryCentroids + centroid) *
                        fine_blocks +
                    block_high] = stored_low_high;
                accumulate_lse(max_low, den_low, __half2float(stored_low_high));
            }
        }
        if (q_block_high < q_blocks) {
            if (block_low < fine_blocks) {
                output[
                    ((bh * q_blocks + q_block_high) * kQueryCentroids + centroid) *
                        fine_blocks +
                    block_low] = stored_high_low;
                accumulate_lse(max_high, den_high, __half2float(stored_high_low));
            }
            if (block_high < fine_blocks) {
                output[
                    ((bh * q_blocks + q_block_high) * kQueryCentroids + centroid) *
                        fine_blocks +
                    block_high] = stored_high_high;
                accumulate_lse(max_high, den_high, __half2float(stored_high_high));
            }
        }
        __syncwarp();
    }

    // The four lanes of a centroid hold disjoint key blocks; combine them.
#pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
        const float other_max_low = __shfl_xor_sync(0xffffffffU, max_low, mask);
        const float other_den_low = __shfl_xor_sync(0xffffffffU, den_low, mask);
        const float other_max_high = __shfl_xor_sync(0xffffffffU, max_high, mask);
        const float other_den_high = __shfl_xor_sync(0xffffffffU, den_high, mask);
        const float next_low = fmaxf(max_low, other_max_low);
        den_low = den_low * exp2f(max_low - next_low) +
            other_den_low * exp2f(other_max_low - next_low);
        max_low = next_low;
        const float next_high = fmaxf(max_high, other_max_high);
        den_high = den_high * exp2f(max_high - next_high) +
            other_den_high * exp2f(other_max_high - next_high);
        max_high = next_high;
    }
    if ((lane & 3) == 0) {
        const int centroid = lane / 4;
        if (q_block_base < q_blocks) {
            const int64_t slot =
                (bh * q_blocks + q_block_base) * kQueryCentroids + centroid;
            row_max[slot] = max_low;
            row_den[slot] = den_low;
        }
        if (q_block_base + 1 < q_blocks) {
            const int64_t slot =
                (bh * q_blocks + q_block_base + 1) * kQueryCentroids + centroid;
            row_max[slot] = max_high;
            row_den[slot] = den_high;
        }
    }
}


template <
    int CentroidCount,
    bool PopulationWeighted,
    bool EmitMean,
    bool EmitRms,
    int GroupFactor>
__global__ __launch_bounds__(kMassThreads)
void weighted_centroid_mass_score_kernel(
    const half* __restrict__ lse,
    const half* __restrict__ populations,
    const float* __restrict__ lse_max,
    const float* __restrict__ lse_den,
    float* __restrict__ scores,
    float* __restrict__ rms_scores,
    int64_t rows,
    int fine_blocks) {
    __shared__ float row_max[CentroidCount];
    __shared__ float row_den[CentroidCount];
    __shared__ float row_weight[CentroidCount];
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // The scoring kernel already reduced each centroid's softmax normalizer,
    // so the LSE array is streamed exactly once here.
    if (threadIdx.x < CentroidCount) {
        const int centroid = threadIdx.x;
        row_max[centroid] = lse_max[row * CentroidCount + centroid];
        row_den[centroid] = lse_den[row * CentroidCount + centroid];
        const float population =
            __half2float(populations[row * CentroidCount + centroid]);
        row_weight[centroid] = PopulationWeighted
            ? population
            : (population > 0.0f ? 1.0f : 0.0f);
    }
    __syncthreads();

    float total_weight = 0.0f;
#pragma unroll
    for (int centroid = 0; centroid < CentroidCount; ++centroid) {
        total_weight += row_weight[centroid];
    }
    const int group_blocks = fine_blocks / GroupFactor;
    const int complete_fine_blocks = group_blocks * GroupFactor;
    for (int start = 0; start < complete_fine_blocks; start += kMassThreads) {
        const int fine_block = start + threadIdx.x;
        float score = 0.0f;
        float squared_score = 0.0f;
        if (fine_block < complete_fine_blocks) {
#pragma unroll
            for (int centroid = 0; centroid < CentroidCount; ++centroid) {
                const float value = __half2float(
                    lse[(row * CentroidCount + centroid) * fine_blocks + fine_block]);
                const float probability =
                    exp2f(value - row_max[centroid]) / row_den[centroid];
                if constexpr (EmitMean) {
                    score += row_weight[centroid] * probability;
                }
                if constexpr (EmitRms) {
                    squared_score +=
                        row_weight[centroid] * probability * probability;
                }
            }
        }
        if constexpr (EmitMean) {
            score /= total_weight;
        }
        if constexpr (EmitRms) {
            squared_score = sqrtf(squared_score / total_weight);
        }
#pragma unroll
        for (int offset = GroupFactor / 2; offset > 0; offset >>= 1) {
            if constexpr (EmitMean) {
                score += __shfl_down_sync(0xffffffffU, score, offset);
            }
            if constexpr (EmitRms) {
                squared_score +=
                    __shfl_down_sync(0xffffffffU, squared_score, offset);
            }
        }
        if (fine_block < complete_fine_blocks &&
            (threadIdx.x & (GroupFactor - 1)) == 0) {
            const int group_block = fine_block / GroupFactor;
            if constexpr (EmitMean) {
                scores[row * group_blocks + group_block] = score;
            }
            if constexpr (EmitRms) {
                rms_scores[row * group_blocks + group_block] = squared_score;
            }
        }
    }
}

template <int GroupFactor>
void launch_weighted_centroid_mass_score(
    const half* lse,
    const half* populations,
    const float* lse_max,
    const float* lse_den,
    float* scores,
    float* rms_scores,
    int64_t rows,
    int fine_blocks,
    bool population_weighted,
    bool emit_mean,
    bool emit_rms,
    cudaStream_t stream) {
    if (!population_weighted) {
        weighted_centroid_mass_score_kernel<
            kQueryCentroids, false, true, false, GroupFactor><<<
            static_cast<unsigned int>(rows), kMassThreads, 0, stream>>>(
            lse, populations, lse_max, lse_den, scores, nullptr, rows,
            fine_blocks);
    } else if (emit_mean && emit_rms) {
        weighted_centroid_mass_score_kernel<
            kQueryCentroids, true, true, true, GroupFactor><<<
            static_cast<unsigned int>(rows), kMassThreads, 0, stream>>>(
            lse, populations, lse_max, lse_den, scores, rms_scores, rows,
            fine_blocks);
    } else if (emit_mean) {
        weighted_centroid_mass_score_kernel<
            kQueryCentroids, true, true, false, GroupFactor><<<
            static_cast<unsigned int>(rows), kMassThreads, 0, stream>>>(
            lse, populations, lse_max, lse_den, scores, nullptr, rows,
            fine_blocks);
    } else {
        weighted_centroid_mass_score_kernel<
            kQueryCentroids, true, false, true, GroupFactor><<<
            static_cast<unsigned int>(rows), kMassThreads, 0, stream>>>(
            lse, populations, lse_max, lse_den, nullptr, rms_scores, rows,
            fine_blocks);
    }
}

template <int GroupFactor>
__global__ void expand_sampled_selector_groups_kernel(
    const int64_t* __restrict__ group_ids,
    const float* __restrict__ group_log_weights,
    int32_t* __restrict__ kv4_ids,
    float* __restrict__ kv4_log_weights,
    int64_t output_count) {
    for (int64_t output = static_cast<int64_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         output < output_count;
         output += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        const int member = static_cast<int>(output % GroupFactor);
        const int64_t input = output / GroupFactor;
        kv4_ids[output] =
            static_cast<int32_t>(group_ids[input] * GroupFactor + member);
        kv4_log_weights[output] = group_log_weights[input];
    }
}

template <int GroupFactor>
void launch_expand_sampled_selector_groups(
    const int64_t* group_ids,
    const float* group_log_weights,
    int32_t* kv4_ids,
    float* kv4_log_weights,
    int64_t output_count,
    int64_t blocks,
    cudaStream_t stream) {
    constexpr int threads = 256;
    expand_sampled_selector_groups_kernel<GroupFactor><<<
        static_cast<unsigned int>(blocks), threads, 0, stream>>>(
        group_ids,
        group_log_weights,
        kv4_ids,
        kv4_log_weights,
        output_count);
}

}  // namespace

static std::vector<torch::Tensor>
candidate_token_mass_scores_qcentroid_kv4_k2_impl(
    torch::Tensor q,
    torch::Tensor k,
    int64_t selector_kv_block_size,
    int64_t block_size,
    bool population_weighted,
    bool emit_mean,
    bool emit_rms,
    bool query_pool = false,
    bool key_pool = false) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda(), "q and k must be CUDA tensors");
    TORCH_CHECK(q.get_device() == k.get_device(),
                "q and k must use the same CUDA device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
                "q and k must be contiguous");
    TORCH_CHECK(q.sizes() == k.sizes() && q.dim() == 4,
                "q and k must have matching [batch, heads, seq_len, 128] shapes");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16 &&
                    k.scalar_type() == at::ScalarType::BFloat16,
                "Q-centroid KV4-K2 scoring requires bfloat16 q/k");
    TORCH_CHECK(q.size(3) == kHeadDim,
                "Q-centroid KV4-K2 scoring requires head_dim=128");
    TORCH_CHECK(
        (selector_kv_block_size == 4 || selector_kv_block_size == 8 ||
         selector_kv_block_size == 16 || selector_kv_block_size == 32 ||
         selector_kv_block_size == 64 || selector_kv_block_size == 128) &&
            block_size == kBlockSize,
        "Q-centroid KV4-K2 scoring requires selector KV block size in "
        "{4,8,16,32,64,128} and query block size 128");
    TORCH_CHECK(q.size(2) % kMicroFineSize == 0,
                "sequence length must be divisible by 4");
    const int64_t batch = q.size(0);
    const int64_t heads = q.size(1);
    const int64_t batch_heads = batch * heads;
    const int64_t seq_len = q.size(2);
    const int64_t q_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
    const int64_t fine_blocks = seq_len / kMicroFineSize;
    const int group_factor =
        static_cast<int>(selector_kv_block_size / kMicroFineSize);
    const int64_t group_blocks = fine_blocks / group_factor;
    TORCH_CHECK(fine_blocks > 0 && fine_blocks <= 32768,
                "KV4 block count must be in [1, 32768]");
    TORCH_CHECK(group_blocks > 0,
                "sequence is too short for the selector KV block size");
    const int64_t q_groups =
        (q_blocks + kQBlocksPerGroup - 1) /
        kQBlocksPerGroup;
    const int64_t rows = batch_heads * q_blocks;

    c10::cuda::CUDAGuard device_guard(q.device());
    auto packed_q = torch::empty(
        {batch, heads, q_blocks, kQueryCentroids, kHeadDim}, q.options());
    auto q_populations = torch::empty(
        {batch, heads, q_blocks, kQueryCentroids},
        q.options().dtype(torch::kFloat16));
    auto packed_k = torch::empty(
        {batch, heads, fine_blocks, kMicroKCentroids, kHeadDim}, k.options());
    auto log_counts = torch::empty(
        {batch, heads, fine_blocks, kMicroKCentroids},
        k.options().dtype(torch::kFloat16));
    auto lse = torch::empty(
        {rows, kQueryCentroids, fine_blocks},
        q.options().dtype(torch::kFloat16));
    auto lse_max = torch::empty(
        {rows, kQueryCentroids}, q.options().dtype(torch::kFloat32));
    auto lse_den = torch::empty(
        {rows, kQueryCentroids}, q.options().dtype(torch::kFloat32));
    auto scores = emit_mean
        ? torch::empty(
              {rows, group_blocks}, q.options().dtype(torch::kFloat32))
        : torch::Tensor();
    auto rms_scores = emit_rms
        ? torch::empty(
              {rows, group_blocks}, q.options().dtype(torch::kFloat32))
        : torch::Tensor();
    auto stream = at::cuda::getCurrentCUDAStream();

    const dim3 q_pack_grid(
        static_cast<unsigned int>(q_blocks),
        static_cast<unsigned int>(batch_heads));
    if (query_pool) {
        q_pool8_pack_kernel<<<q_pack_grid, kHeadDim, 0, stream>>>(
            reinterpret_cast<const Bf16*>(q.data_ptr()),
            reinterpret_cast<Bf16*>(packed_q.data_ptr()),
            reinterpret_cast<half*>(q_populations.data_ptr()),
            seq_len,
            q_blocks);
    } else {
        q_centroid8_pack_kernel<<<q_pack_grid, kHeadDim, 0, stream>>>(
            reinterpret_cast<const Bf16*>(q.data_ptr()),
            reinterpret_cast<Bf16*>(packed_q.data_ptr()),
            reinterpret_cast<half*>(q_populations.data_ptr()),
            seq_len,
            q_blocks);
    }
    const dim3 k_pack_grid(
        static_cast<unsigned int>(fine_blocks),
        static_cast<unsigned int>(batch_heads));
    if (key_pool) {
        kv4_pool2_pack_kernel<<<k_pack_grid, kHeadDim, 0, stream>>>(
            reinterpret_cast<const Bf16*>(k.data_ptr()),
            reinterpret_cast<Bf16*>(packed_k.data_ptr()),
            reinterpret_cast<half*>(log_counts.data_ptr()),
            fine_blocks);
    } else {
        kv4_k2_pack_kernel<<<k_pack_grid, kHeadDim, 0, stream>>>(
            reinterpret_cast<const Bf16*>(k.data_ptr()),
            reinterpret_cast<Bf16*>(packed_k.data_ptr()),
            reinterpret_cast<half*>(log_counts.data_ptr()),
            fine_blocks);
    }

    const float qk_scale_log2 = static_cast<float>(
        std::pow(static_cast<double>(kHeadDim), -0.5) *
        1.4426950408889634);
    const int q_group_tiles =
        static_cast<int>((q_groups + kWarps - 1) / kWarps);
    const dim3 score_grid(
        q_group_tiles, static_cast<unsigned int>(batch_heads));
    qcentroid8_kv4_k2_lse_kernel<<<
        score_grid, kThreads, kSharedBytes, stream>>>(
        reinterpret_cast<const Bf16*>(packed_q.data_ptr()),
        reinterpret_cast<const Bf16*>(packed_k.data_ptr()),
        reinterpret_cast<const half*>(log_counts.data_ptr()),
        reinterpret_cast<half*>(lse.data_ptr()),
        lse_max.data_ptr<float>(),
        lse_den.data_ptr<float>(),
        q_blocks,
        static_cast<int>(fine_blocks),
        qk_scale_log2);
    const half* lse_ptr = reinterpret_cast<const half*>(lse.data_ptr());
    const half* population_ptr =
        reinterpret_cast<const half*>(q_populations.data_ptr());
    float* score_ptr = emit_mean
        ? reinterpret_cast<float*>(scores.data_ptr())
        : nullptr;
    float* rms_score_ptr = emit_rms
        ? reinterpret_cast<float*>(rms_scores.data_ptr())
        : nullptr;
#define AB_LAUNCH_GROUPED_SCORE(factor)                                      \
    launch_weighted_centroid_mass_score<factor>(                            \
        lse_ptr,                                                             \
        population_ptr,                                                      \
        lse_max.data_ptr<float>(),                                           \
        lse_den.data_ptr<float>(),                                           \
        score_ptr,                                                           \
        rms_score_ptr,                                                       \
        rows,                                                                \
        static_cast<int>(fine_blocks),                                       \
        population_weighted,                                                 \
        emit_mean,                                                           \
        emit_rms,                                                            \
        stream)
    switch (group_factor) {
        case 1: AB_LAUNCH_GROUPED_SCORE(1); break;
        case 2: AB_LAUNCH_GROUPED_SCORE(2); break;
        case 4: AB_LAUNCH_GROUPED_SCORE(4); break;
        case 8: AB_LAUNCH_GROUPED_SCORE(8); break;
        case 16: AB_LAUNCH_GROUPED_SCORE(16); break;
        case 32: AB_LAUNCH_GROUPED_SCORE(32); break;
    }
#undef AB_LAUNCH_GROUPED_SCORE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (emit_mean && emit_rms) {
        return {scores, rms_scores};
    }
    return {emit_mean ? scores : rms_scores};
}

torch::Tensor candidate_group_scores_q8_kv4_k2_cuda(
    torch::Tensor q,
    torch::Tensor k,
    int64_t selector_kv_block_size,
    int64_t block_size,
    const std::string& query_representation,
    const std::string& key_representation,
    const std::string& score_aggregation) {
    TORCH_CHECK(
        query_representation == "cluster" || query_representation == "pool",
        "query representation must be cluster or pool");
    TORCH_CHECK(
        key_representation == "cluster" || key_representation == "pool",
        "key representation must be cluster or pool");
    const bool query_pool = query_representation == "pool";
    const bool key_pool = key_representation == "pool";
    bool population_weighted = true;
    bool emit_mean = false;
    bool emit_rms = false;
    if (score_aggregation == "weighted_mean") {
        emit_mean = true;
    } else if (score_aggregation == "weighted_rms") {
        emit_rms = true;
    } else if (score_aggregation == "simple_mean") {
        population_weighted = false;
        emit_mean = true;
    } else {
        TORCH_CHECK(false, "unsupported score aggregation: ", score_aggregation);
    }
    return candidate_token_mass_scores_qcentroid_kv4_k2_impl(
        std::move(q),
        std::move(k),
        selector_kv_block_size,
        block_size,
        population_weighted,
        emit_mean,
        emit_rms,
        query_pool,
        key_pool)[0];
}

std::vector<torch::Tensor> expand_sampled_selector_groups_cuda(
    torch::Tensor group_ids,
    torch::Tensor group_log_weights,
    int64_t selector_kv_block_size) {
    TORCH_CHECK(group_ids.is_cuda() && group_log_weights.is_cuda(),
                "group ids and log weights must be CUDA tensors");
    TORCH_CHECK(group_ids.get_device() == group_log_weights.get_device(),
                "group ids and log weights must use the same CUDA device");
    TORCH_CHECK(group_ids.is_contiguous() && group_log_weights.is_contiguous(),
                "group ids and log weights must be contiguous");
    TORCH_CHECK(group_ids.dim() == 2 &&
                    group_ids.sizes() == group_log_weights.sizes(),
                "group ids and log weights must have matching [rows, samples] shapes");
    TORCH_CHECK(group_ids.scalar_type() == at::ScalarType::Long &&
                    group_log_weights.scalar_type() == at::ScalarType::Float,
                "expected int64 group ids and float32 log weights");
    TORCH_CHECK(
        selector_kv_block_size == 4 || selector_kv_block_size == 8 ||
            selector_kv_block_size == 16 || selector_kv_block_size == 32 ||
            selector_kv_block_size == 64 || selector_kv_block_size == 128,
        "selector KV block size must be one of {4,8,16,32,64,128}");

    const int64_t group_factor = selector_kv_block_size / kMicroFineSize;
    const int64_t expanded_samples = group_ids.size(1) * group_factor;
    const int64_t output_count = group_ids.size(0) * expanded_samples;
    c10::cuda::CUDAGuard device_guard(group_ids.device());
    auto kv4_ids = torch::empty(
        {group_ids.size(0), expanded_samples},
        group_ids.options().dtype(at::ScalarType::Int));
    auto kv4_log_weights = torch::empty(
        {group_ids.size(0), expanded_samples}, group_log_weights.options());
    constexpr int threads = 256;
    const int64_t blocks = (output_count + threads - 1) / threads;
    if (blocks > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t* ids_ptr = group_ids.data_ptr<int64_t>();
        const float* input_logs_ptr = group_log_weights.data_ptr<float>();
        int32_t* output_ids_ptr = kv4_ids.data_ptr<int32_t>();
        float* output_logs_ptr = kv4_log_weights.data_ptr<float>();
#define AB_LAUNCH_GROUP_EXPANSION(factor)                                   \
    launch_expand_sampled_selector_groups<factor>(                          \
        ids_ptr,                                                            \
        input_logs_ptr,                                                     \
        output_ids_ptr,                                                     \
        output_logs_ptr,                                                    \
        output_count,                                                       \
        blocks,                                                             \
        stream)
        switch (group_factor) {
            case 1: AB_LAUNCH_GROUP_EXPANSION(1); break;
            case 2: AB_LAUNCH_GROUP_EXPANSION(2); break;
            case 4: AB_LAUNCH_GROUP_EXPANSION(4); break;
            case 8: AB_LAUNCH_GROUP_EXPANSION(8); break;
            case 16: AB_LAUNCH_GROUP_EXPANSION(16); break;
            case 32: AB_LAUNCH_GROUP_EXPANSION(32); break;
        }
#undef AB_LAUNCH_GROUP_EXPANSION
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {kv4_ids, kv4_log_weights};
}
