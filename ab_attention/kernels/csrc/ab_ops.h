#pragma once

#include <cstdint>
#include <string>

#include <torch/extension.h>

#include <vector>

torch::Tensor candidate_group_scores_q8_kv4_k2_cuda(
    torch::Tensor q,
    torch::Tensor k,
    int64_t selector_kv_block_size,
    int64_t block_size,
    const std::string& query_representation,
    const std::string& key_representation,
    const std::string& score_aggregation);

std::vector<torch::Tensor> expand_sampled_selector_groups_cuda(
    torch::Tensor group_ids,
    torch::Tensor group_log_weights,
    int64_t selector_kv_block_size);

std::vector<torch::Tensor> pure_pps_wor_cuda(
    torch::Tensor masses,
    torch::Tensor support,
    torch::Tensor offsets,
    int64_t sample_count);

std::vector<torch::Tensor> descriptor_attention_state_wgmma_kv4_sm90a_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc);

std::vector<torch::Tensor> descriptor_attention_state_wgmma_weighted_kv4_sm90a_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor desc,
    torch::Tensor page_log_weights);
