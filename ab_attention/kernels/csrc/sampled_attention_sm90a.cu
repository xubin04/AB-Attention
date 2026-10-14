#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/bfloat16.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <flash-attention/hopper/softmax.h>
#include "ab_ops.h"
#include "flashinfer/cp_async.cuh"

#include <cmath>
#include <cstdint>

namespace ab_attention_sm90a {

using namespace cute;

using Element = cutlass::bfloat16_t;

constexpr int kQueryBlockSize = 128;
constexpr int kKvBlockSize = 128;
constexpr int kHeadDim = 128;
constexpr int kProducerWarpGroups = 2;
constexpr int kProducerThreads = 128;
constexpr int kTotalProducerThreads =
    kProducerWarpGroups * kProducerThreads;
constexpr int kWarpGroupThreads = 128;
constexpr int kRowsPerWarpGroup = 64;
constexpr int kConsumerWarpGroups = kQueryBlockSize / kRowsPerWarpGroup;
constexpr int kConsumerThreads = kConsumerWarpGroups * kWarpGroupThreads;
constexpr int kThreads = kTotalProducerThreads + kConsumerThreads;
constexpr int kPipelineStages = 2;
constexpr uint32_t kLoadRegisters = 48;
constexpr uint32_t kMmaRegisters = 208;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kVectorElements = sizeof(uint4) / sizeof(Element);
constexpr int kVectorsPerRow = kHeadDim / kVectorElements;
constexpr int kQueryTileElements = kQueryBlockSize * kHeadDim;
constexpr int kKvTileElements = kKvBlockSize * kHeadDim;
constexpr size_t kQueryTileBytes = kQueryTileElements * sizeof(Element);
constexpr size_t kKvTileBytes = kKvTileElements * sizeof(Element);
constexpr size_t kPipelineSharedBytes = kQueryTileBytes + 4 * kKvTileBytes;

using MainloopPipeline = cutlass::PipelineAsync<kPipelineStages>;
using PipelineState = typename MainloopPipeline::PipelineState;
using QLayout = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    make_shape(Int<kQueryBlockSize>{}, Int<kHeadDim>{})));
using KLayout = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<Element>{},
    make_shape(Int<kKvBlockSize>{}, Int<kHeadDim>{})));
using VLayout = decltype(tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<Element>{},
    make_shape(Int<kHeadDim>{}, Int<kKvBlockSize>{})));
using MmaQK = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
using MmaPV = decltype(make_tiled_mma(
    SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));

static_assert(kVectorElements == 8);
static_assert(kTotalProducerThreads + kConsumerThreads == kThreads);
static_assert(kLoadRegisters * kTotalProducerThreads +
                  kMmaRegisters * kConsumerThreads <=
              65536);
static_assert(cosize_v<QLayout> == kQueryTileElements);
static_assert(cosize_v<KLayout> == kKvTileElements);
static_assert(cosize_v<VLayout> == kKvTileElements);
static_assert(size(MmaQK{}) == kWarpGroupThreads);
static_assert(size(MmaPV{}) == kWarpGroupThreads);

template <int Pages, int FineSize>
__global__ __launch_bounds__(kThreads, 1)
void synthetic_page_attention_wgmma_sm90a_ws_kernel(
    const Element* __restrict__ q,
    const Element* __restrict__ k,
    const Element* __restrict__ v,
    const int32_t* __restrict__ desc,
    float* __restrict__ out_float,
    float* __restrict__ lse2_out,
    int64_t seq_len,
    int64_t q_blocks,
    float qk_scale_log2,
    const float* __restrict__ page_log_weights,
    int64_t page_weight_stride) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ __align__(16) unsigned char pipeline_k_raw[
        sizeof(typename MainloopPipeline::SharedStorage)];
    __shared__ __align__(16) unsigned char pipeline_v_raw[
        sizeof(typename MainloopPipeline::SharedStorage)];
    auto& pipeline_k_storage =
        *reinterpret_cast<typename MainloopPipeline::SharedStorage*>(
            pipeline_k_raw);
    auto& pipeline_v_storage =
        *reinterpret_cast<typename MainloopPipeline::SharedStorage*>(
            pipeline_v_raw);

    Element* q_storage = reinterpret_cast<Element*>(smem_raw);
    Element* k0_storage = reinterpret_cast<Element*>(smem_raw + kQueryTileBytes);
    Element* k1_storage = reinterpret_cast<Element*>(
        smem_raw + kQueryTileBytes + kKvTileBytes);
    Element* v0_storage = reinterpret_cast<Element*>(
        smem_raw + kQueryTileBytes + 2 * kKvTileBytes);
    Element* v1_storage = reinterpret_cast<Element*>(
        smem_raw + kQueryTileBytes + 3 * kKvTileBytes);
    Tensor s_q = make_tensor(make_smem_ptr(q_storage), QLayout{});
    Tensor s_k0 = make_tensor(make_smem_ptr(k0_storage), KLayout{});
    Tensor s_k1 = make_tensor(make_smem_ptr(k1_storage), KLayout{});
    Tensor s_v0 = make_tensor(make_smem_ptr(v0_storage), VLayout{});
    Tensor s_v1 = make_tensor(make_smem_ptr(v1_storage), VLayout{});
    const int tid = threadIdx.x;
    const int warp_group = tid / kWarpGroupThreads;
    const bool is_k_producer = warp_group == 0;
    const bool is_v_producer = warp_group == 1;
    const bool is_producer = is_k_producer || is_v_producer;
    const bool is_consumer = warp_group >= kProducerWarpGroups;
    const int64_t launch_row = blockIdx.x;
    const int64_t qh = launch_row;
    const int64_t bh = qh / q_blocks;
    const int64_t q_block = qh - bh * q_blocks;
    const int64_t q_token_base = q_block * kQueryBlockSize;
    constexpr int kSubsPerPage = kKvBlockSize / FineSize;
    static_assert(kKvBlockSize % FineSize == 0);
    const int64_t desc_base = launch_row * Pages * kSubsPerPage;
    const int64_t kv_bh_base = bh * seq_len;

    typename MainloopPipeline::Params pipeline_k_params;
    pipeline_k_params.role = is_k_producer
        ? MainloopPipeline::ThreadCategory::Producer
        : (is_consumer ? MainloopPipeline::ThreadCategory::Consumer
                       : MainloopPipeline::ThreadCategory::NonParticipant);
    pipeline_k_params.producer_arv_count = kProducerThreads;
    pipeline_k_params.consumer_arv_count = kConsumerThreads;
    typename MainloopPipeline::Params pipeline_v_params;
    pipeline_v_params.role = is_v_producer
        ? MainloopPipeline::ThreadCategory::Producer
        : (is_consumer ? MainloopPipeline::ThreadCategory::Consumer
                       : MainloopPipeline::ThreadCategory::NonParticipant);
    pipeline_v_params.producer_arv_count = kProducerThreads;
    pipeline_v_params.consumer_arv_count = kConsumerThreads;
    MainloopPipeline pipeline_k(pipeline_k_storage, pipeline_k_params);
    MainloopPipeline pipeline_v(pipeline_v_storage, pipeline_v_params);

    if (is_producer) {
        for (int vector_idx = tid;
             vector_idx < kQueryBlockSize * kVectorsPerRow;
             vector_idx += kTotalProducerThreads) {
            const int row = vector_idx / kVectorsPerRow;
            const int vector_col = vector_idx - row * kVectorsPerRow;
            const int col = vector_col * kVectorElements;
            const int64_t q_token = q_token_base + row;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (q_token < seq_len) {
                packed = *reinterpret_cast<const uint4*>(
                    q + (bh * seq_len + q_token) * kHeadDim + col);
            }
            *reinterpret_cast<uint4*>(&s_q(row, col)) = packed;
        }
    }
    __syncthreads();

    if (is_k_producer) {
        cutlass::arch::warpgroup_reg_dealloc<kLoadRegisters>();
        const int producer_warp = tid >> 5;
        const int lane = tid & 31;
        PipelineState k_write_state =
            cutlass::make_producer_start_state<MainloopPipeline>();
        for (int page = 0; page < Pages; ++page) {
            pipeline_k.producer_acquire(k_write_state);
            const int k_stage = k_write_state.index();

#pragma unroll
            for (int slot = producer_warp;
                 slot < kSubsPerPage;
                 slot += kProducerThreads / 32) {
                int32_t fine_block = lane == 0
                    ? desc[desc_base + page * kSubsPerPage + slot]
                    : 0;
                fine_block = __shfl_sync(0xffffffffu, fine_block, 0);
                constexpr int kFineVectors = FineSize * kVectorsPerRow;
#pragma unroll
                for (int vector_idx = lane;
                     vector_idx < kFineVectors;
                     vector_idx += 32) {
                    const int row_in_fine = vector_idx / kVectorsPerRow;
                    const int vector_col =
                        vector_idx - row_in_fine * kVectorsPerRow;
                    const int col = vector_col * kVectorElements;
                    const int row = slot * FineSize + row_in_fine;
                    const int64_t kv_token =
                        static_cast<int64_t>(fine_block) * FineSize +
                        row_in_fine;
                    const int64_t element_offset =
                        (kv_bh_base + kv_token) * kHeadDim + col;
                    if (k_stage == 0) {
                        flashinfer::cp_async::load_128b<
                            flashinfer::cp_async::PrefetchMode::kPrefetch>(
                                &s_k0(row, col), k + element_offset);
                    } else {
                        flashinfer::cp_async::load_128b<
                            flashinfer::cp_async::PrefetchMode::kPrefetch>(
                                &s_k1(row, col), k + element_offset);
                    }
                }
            }
            pipeline_k.producer_commit(
                k_write_state, cutlass::arch::cpasync_barrier_arrive);
            ++k_write_state;
        }
        pipeline_k.producer_tail(k_write_state);
        return;
    }

    if (is_v_producer) {
        cutlass::arch::warpgroup_reg_dealloc<kLoadRegisters>();
        const int producer_warp = (tid - kProducerThreads) >> 5;
        const int lane = tid & 31;
        PipelineState v_write_state =
            cutlass::make_producer_start_state<MainloopPipeline>();

        for (int page = 0; page < Pages; ++page) {
            pipeline_v.producer_acquire(v_write_state);
            const int v_stage = v_write_state.index();
#pragma unroll
            for (int slot = producer_warp;
                 slot < kSubsPerPage;
                 slot += kProducerThreads / 32) {
                int32_t fine_block = lane == 0
                    ? desc[desc_base + page * kSubsPerPage + slot]
                    : 0;
                fine_block = __shfl_sync(0xffffffffu, fine_block, 0);
                constexpr int kFineVectors = FineSize * kVectorsPerRow;
#pragma unroll
                for (int vector_idx = lane;
                     vector_idx < kFineVectors;
                     vector_idx += 32) {
                    const int row_in_fine = vector_idx / kVectorsPerRow;
                    const int vector_col =
                        vector_idx - row_in_fine * kVectorsPerRow;
                    const int col = vector_col * kVectorElements;
                    const int row = slot * FineSize + row_in_fine;
                    const int64_t kv_token =
                        static_cast<int64_t>(fine_block) * FineSize +
                        row_in_fine;
                    const int64_t element_offset =
                        (kv_bh_base + kv_token) * kHeadDim + col;
                    if (v_stage == 0) {
                        flashinfer::cp_async::load_128b<
                            flashinfer::cp_async::PrefetchMode::kPrefetch>(
                                &s_v0(col, row), v + element_offset);
                    } else {
                        flashinfer::cp_async::load_128b<
                            flashinfer::cp_async::PrefetchMode::kPrefetch>(
                                &s_v1(col, row), v + element_offset);
                    }
                }
            }
            pipeline_v.producer_commit(
                v_write_state, cutlass::arch::cpasync_barrier_arrive);
            ++v_write_state;
        }
        pipeline_v.producer_tail(v_write_state);
        return;
    }

    cutlass::arch::warpgroup_reg_alloc<kMmaRegisters>();
    const int consumer_group = warp_group - kProducerWarpGroups;
    const int warp_group_lane = tid - warp_group * kWarpGroupThreads;
    Tensor s_q_wg = local_tile(
        s_q,
        make_shape(Int<kRowsPerWarpGroup>{}, Int<kHeadDim>{}),
        make_coord(consumer_group, 0));

    MmaQK mma_qk;
    MmaPV mma_pv;
    ThrMMA thread_mma_qk = mma_qk.get_slice(warp_group_lane);
    ThrMMA thread_mma_pv = mma_pv.get_slice(warp_group_lane);
    Tensor tCrQ = thread_mma_qk.make_fragment_A(
        thread_mma_qk.partition_A(s_q_wg));
    Tensor tCrK0 = thread_mma_qk.make_fragment_B(
        thread_mma_qk.partition_B(s_k0));
    Tensor tCrK1 = thread_mma_qk.make_fragment_B(
        thread_mma_qk.partition_B(s_k1));
    Tensor tCrV0 = thread_mma_pv.make_fragment_B(
        thread_mma_pv.partition_B(s_v0));
    Tensor tCrV1 = thread_mma_pv.make_fragment_B(
        thread_mma_pv.partition_B(s_v1));
    Tensor c_out = make_identity_tensor(
        make_shape(Int<kRowsPerWarpGroup>{}, Int<kHeadDim>{}));
    Tensor tCcOut = thread_mma_pv.partition_C(c_out);
    Tensor tCrOut = thread_mma_pv.make_fragment_C(tCcOut);
    clear(tCrOut);

    flash::Softmax<2> softmax(qk_scale_log2);
    Tensor score_coordinates = thread_mma_qk.partition_C(
        make_identity_tensor(
            make_shape(Int<kRowsPerWarpGroup>{}, Int<kKvBlockSize>{})));
    Tensor tCrScore = thread_mma_qk.make_fragment_C(score_coordinates);
    Tensor tOrPAcc = make_tensor(
        tCrScore.data(),
        flash::convert_layout_acc_Aregs<MmaPV>(tCrScore.layout()));
    Tensor tOrP = make_tensor_like<Element>(tOrPAcc);
    PipelineState k_read_state;
    PipelineState v_read_state;

    auto first_k_token = pipeline_k.consumer_try_wait(k_read_state);
    pipeline_k.consumer_wait(k_read_state, first_k_token);
    clear(tCrScore);
    warpgroup_fence_operand(tCrScore);
    warpgroup_arrive();
    cute::gemm(mma_qk, tCrQ, tCrK0, tCrScore);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrScore);
    pipeline_k.consumer_release(k_read_state);
    ++k_read_state;
    if (page_log_weights != nullptr) {
#pragma unroll
        for (int score_idx = 0; score_idx < size(tCrScore); ++score_idx) {
            const int64_t weight_index = launch_row * page_weight_stride +
                get<1>(score_coordinates(score_idx)) / FineSize;
            tCrScore(score_idx) +=
                page_log_weights[weight_index] / qk_scale_log2;
        }
    }
    softmax.max_get_scale<true, false>(tCrScore);
    softmax.online_softmax<true, false>(tCrScore);
    flash::convert_type_out(tOrPAcc, tOrP);

    for (int page = 1; page < Pages; ++page) {
        auto k_wait_token = pipeline_k.consumer_try_wait(k_read_state);
        pipeline_k.consumer_wait(k_read_state, k_wait_token);
        clear(tCrScore);
        warpgroup_fence_operand(tCrScore);
        warpgroup_arrive();
        if (k_read_state.index() == 0) {
            cute::gemm(mma_qk, tCrQ, tCrK0, tCrScore);
        } else {
            cute::gemm(mma_qk, tCrQ, tCrK1, tCrScore);
        }
        warpgroup_commit_batch();

        auto v_wait_token = pipeline_v.consumer_try_wait(v_read_state);
        pipeline_v.consumer_wait(v_read_state, v_wait_token);
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tCrOut);
        warpgroup_arrive();
        if (v_read_state.index() == 0) {
            cute::gemm(mma_pv, tOrP, tCrV0, tCrOut);
        } else {
            cute::gemm(mma_pv, tOrP, tCrV1, tCrOut);
        }
        warpgroup_commit_batch();

        // QK is the oldest group. Keep the previous PV in flight while CUDA
        // cores update softmax for the current page.
        warpgroup_wait<1>();
        warpgroup_fence_operand(tCrScore);
        pipeline_k.consumer_release(k_read_state);
        ++k_read_state;
        if (page_log_weights != nullptr) {
#pragma unroll
            for (int score_idx = 0; score_idx < size(tCrScore); ++score_idx) {
                const int64_t weight_index = launch_row * page_weight_stride +
                    page * (kKvBlockSize / FineSize) +
                    get<1>(score_coordinates(score_idx)) / FineSize;
                tCrScore(score_idx) +=
                    page_log_weights[weight_index] / qk_scale_log2;
            }
        }
        Tensor scores_scale =
            softmax.max_get_scale<false, false>(tCrScore);
        softmax.online_softmax<false, false>(tCrScore);
        warpgroup_wait<0>();
        warpgroup_fence_operand(tOrP);
        warpgroup_fence_operand(tCrOut);
        pipeline_v.consumer_release(v_read_state);
        ++v_read_state;
        softmax.rescale_o(tCrOut, scores_scale);
        flash::convert_type_out(tOrPAcc, tOrP);
    }

    auto last_v_token = pipeline_v.consumer_try_wait(v_read_state);
    pipeline_v.consumer_wait(v_read_state, last_v_token);
    warpgroup_fence_operand(tOrP);
    warpgroup_fence_operand(tCrOut);
    warpgroup_arrive();
    if (v_read_state.index() == 0) {
        cute::gemm(mma_pv, tOrP, tCrV0, tCrOut);
    } else {
        cute::gemm(mma_pv, tOrP, tCrV1, tCrOut);
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tOrP);
    warpgroup_fence_operand(tCrOut);
    pipeline_v.consumer_release(v_read_state);

    Tensor final_scale = softmax.finalize();
    softmax.rescale_o(tCrOut, final_scale);
    Tensor tCrOutRowcol = make_tensor(
        tCrOut.data(), flash::convert_layout_acc_rowcol(tCrOut.layout()));
    Tensor tCcOutRowcol = make_tensor(
        tCcOut.data(), flash::convert_layout_acc_rowcol(tCcOut.layout()));
#pragma unroll
    for (int mi = 0; mi < size<0>(tCrOutRowcol); ++mi) {
        const auto row_coord = tCcOutRowcol(mi, 0);
        const int local_row = get<0>(row_coord);
        const int row_col = get<1>(row_coord);
        const int64_t q_token =
            q_token_base + consumer_group * kRowsPerWarpGroup + local_row;
        if (row_col == 0 && q_token < seq_len) {
            lse2_out[bh * seq_len + q_token] =
                softmax.row_sum(mi) * kLog2e;
        }
#pragma unroll
        for (int ni = 0; ni < size<1>(tCrOutRowcol); ++ni) {
            const auto coord = tCcOutRowcol(mi, ni);
            const int col = get<1>(coord);
            if (q_token < seq_len) {
                const int64_t output_idx =
                    (bh * seq_len + q_token) * kHeadDim + col;
                out_float[output_idx] = tCrOutRowcol(mi, ni);
            }
        }
    }
#endif
}

void check_attention_input(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
                name, " must be BF16");
    TORCH_CHECK(tensor.dim() == 4, name, " must have shape [B, H, S, 128]");
    TORCH_CHECK(tensor.size(3) == kHeadDim,
                name, " head_dim must be 128");
}

template <int Pages, int FineSize>
void check_common_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& desc) {
    check_attention_input(q, "q");
    check_attention_input(k, "k");
    check_attention_input(v, "v");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, and v must have identical shapes");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same device");
    TORCH_CHECK(desc.is_cuda() && desc.is_contiguous(),
                "desc must be a contiguous CUDA tensor");
    TORCH_CHECK(desc.scalar_type() == torch::kInt32, "desc must be int32");
    constexpr int kSubsPerPage = kKvBlockSize / FineSize;
    static_assert(kKvBlockSize % FineSize == 0);
    TORCH_CHECK(desc.dim() == 3 && desc.size(1) == Pages &&
                    desc.size(2) == kSubsPerPage,
                "desc must have shape [QH, ", Pages, ", ",
                kSubsPerPage, "]");
    TORCH_CHECK(desc.device() == q.device(), "desc must be on the q device");
    const int64_t q_blocks =
        (q.size(2) + kQueryBlockSize - 1) / kQueryBlockSize;
    TORCH_CHECK(desc.size(0) == q.size(0) * q.size(1) * q_blocks,
                "desc.size(0) must equal B * H * ceil_div(S, ",
                kQueryBlockSize, ")");
}

cudaDeviceProp sm90_properties(const torch::Tensor& q) {
    cudaDeviceProp properties{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, q.get_device()));
    TORCH_CHECK(properties.major == 9 && properties.minor == 0,
                "the warp-specialized WGMMA executor requires an SM90 GPU");
    return properties;
}

}  // namespace ab_attention_sm90a

template <int Pages, int FineSize>
std::vector<torch::Tensor>
descriptor_attention_state_wgmma_sm90a_cuda_impl(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc,
    const float* page_log_weights,
    int64_t page_weight_stride) {
    using namespace ab_attention_sm90a;
    check_common_inputs<Pages, FineSize>(q, k, v, desc);
    c10::cuda::CUDAGuard device_guard(q.device());
    const cudaDeviceProp properties = sm90_properties(q);
    TORCH_CHECK(
        properties.sharedMemPerBlockOptin >= kPipelineSharedBytes + 1024,
        "the WGMMA state kernel requires at least ",
        kPipelineSharedBytes + 1024,
        " bytes of opt-in shared memory per block");

    const int64_t seq_len = q.size(2);
    const int64_t q_blocks =
        (seq_len + kQueryBlockSize - 1) / kQueryBlockSize;
    const int64_t qh = q.size(0) * q.size(1) * q_blocks;
    auto out = torch::empty(q.sizes(), q.options().dtype(torch::kFloat32));
    auto lse2 = torch::empty(
        {q.size(0), q.size(1), seq_len},
        q.options().dtype(torch::kFloat32));
    const auto kernel =
        synthetic_page_attention_wgmma_sm90a_ws_kernel<Pages, FineSize>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kPipelineSharedBytes)));
    const float qk_scale_log2 =
        static_cast<float>(std::pow(kHeadDim, -0.5) * kLog2e);
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel<<<static_cast<unsigned int>(qh), kThreads, kPipelineSharedBytes, stream>>>(
        reinterpret_cast<const Element*>(q.data_ptr()),
        reinterpret_cast<const Element*>(k.data_ptr()),
        reinterpret_cast<const Element*>(v.data_ptr()),
        desc.data_ptr<int32_t>(),
        out.data_ptr<float>(),
        lse2.data_ptr<float>(),
        seq_len,
        q_blocks,
        qk_scale_log2,
        page_log_weights,
        page_weight_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse2};
}

template <int FineSize>
std::vector<torch::Tensor>
descriptor_attention_state_wgmma_sm90a_dispatch_impl(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc,
    const float* page_log_weights,
    int64_t page_weight_stride) {
    const int64_t pages = desc.size(1);
    if (pages == 16) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<16, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 32) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<32, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 48) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<48, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 52) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<52, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 56) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<56, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 60) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<60, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 64) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<64, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 71) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<71, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 79) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<79, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 87) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<87, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 95) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<95, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 103) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<103, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 111) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<111, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 127) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<127, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 128) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<128, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    if (pages == 159) {
        return descriptor_attention_state_wgmma_sm90a_cuda_impl<159, FineSize>(
            q, k, v, desc, page_log_weights, page_weight_stride);
    }
    TORCH_CHECK(false, "WGMMA KV4 state attention supports 16, 32, 48, 52, 56, 60, 64, 71, 79, 87, 95, 103, 111, 127, 128, or 159 pages");
}

std::vector<torch::Tensor>
descriptor_attention_state_wgmma_kv4_sm90a_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc) {
    return descriptor_attention_state_wgmma_sm90a_dispatch_impl<4>(
        q, k, v, desc, nullptr, 0);
}

std::vector<torch::Tensor>
descriptor_attention_state_wgmma_weighted_kv4_sm90a_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc,
    torch::Tensor page_log_weights) {
    TORCH_CHECK(page_log_weights.is_cuda(), "page_log_weights must be CUDA");
    TORCH_CHECK(page_log_weights.is_contiguous(), "page_log_weights must be contiguous");
    TORCH_CHECK(page_log_weights.scalar_type() == torch::kFloat32,
                "page_log_weights must be float32");
    TORCH_CHECK(page_log_weights.device() == q.device(),
                "page_log_weights must be on the q device");
    TORCH_CHECK(
        page_log_weights.dim() == 3 &&
            page_log_weights.size(0) == desc.size(0) &&
            page_log_weights.size(1) == desc.size(1) &&
            page_log_weights.size(2) == 32,
        "block_log_weights must have shape [QH, Pages, 32]");
    const float* weights = page_log_weights.data_ptr<float>();
    const int64_t weight_stride = page_log_weights.stride(0);
    return descriptor_attention_state_wgmma_sm90a_dispatch_impl<4>(
        q, k, v, desc, weights, weight_stride);
}
