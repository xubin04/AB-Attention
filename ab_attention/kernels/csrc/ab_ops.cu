#include "ab_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "candidate_group_scores_q8_kv4_k2_cuda",
        &candidate_group_scores_q8_kv4_k2_cuda,
        "Grouped SampleAll scores with independently selected Q/K representatives.");
    module.def(
        "expand_sampled_selector_groups_cuda",
        &expand_sampled_selector_groups_cuda,
        "Expand sampled selector groups into fixed KV4 executor descriptors.");
    module.def(
        "pure_pps_wor_cuda",
        &pure_pps_wor_cuda,
        "Pure systematic PPS without replacement and individual KV4 weights.");
    module.def(
        "descriptor_attention_state_wgmma_kv4_sm90a_cuda",
        &descriptor_attention_state_wgmma_kv4_sm90a_cuda,
        "Hopper route-private KV4 descriptor attention state.");
    module.def(
        "descriptor_attention_state_wgmma_weighted_kv4_sm90a_cuda",
        &descriptor_attention_state_wgmma_weighted_kv4_sm90a_cuda,
        "Weighted Hopper KV4 descriptor attention state.");
}
