// Modified by satellitedown for Cinference: lattice verify trees with lookup chains.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"

namespace ninfer::ops::detail {
// edges is the lattice workspace (FP32 [16,16,K,B]).
void candidate_selector_tree_launch(
    const Tensor& candidate_ids, const Tensor& unary_scores, const Tensor& projected_hidden,
    const Tensor& anchors, const Tensor& predecessor_codebook, const Tensor& successor_codebook,
    const Tensor& current_extents, const Tensor& lookup_tokens, const Tensor& lookup_counts,
    const Tensor& lookup_log_probability, Tensor& drafts, Tensor& tree_parents, Tensor& tree_masks,
    Tensor& rope_positions, const Tensor& edges, cudaStream_t stream);
void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream);
}
