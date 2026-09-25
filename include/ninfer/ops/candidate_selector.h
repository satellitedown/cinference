// Modified by satellitedown for Cinference: lattice verify trees with prompt-lookup chains.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/tensor.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t candidate_selector_path_workspace_capacity_bytes(std::int32_t min_steps,
                                                                           std::int32_t max_steps,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

/**
 * Op: candidate_selector_path
 *
 * For K in [1,15] and B in [1,8], the inputs are contiguous candidate_ids I32 [16,K,B],
 * unary_scores FP32 [16,K,B], projected_hidden BF16 [256,K,B], anchors I32 [B],
 * predecessor_codebook and successor_codebook BF16 [256,248320], base_positions I32 [B], and a
 * device-resident SamplingConfig[B]. Candidate rank is the fastest axis. The 16 candidate ids in
 * each row are distinct, and all candidate and anchor token ids lie in [0,248077); the registered
 * vocabulary, artifact binding, and linear_topk producer establish that trusted value contract.
 *
 * Starting with predecessor=anchors[b], each position i in [0,K) computes:
 *
 *   edge[c] = unary_scores[c,i,b]
 *           + sum_r predecessor_codebook[r,predecessor]
 *                   * projected_hidden[r,i,b]
 *                   * successor_codebook[r,candidate_ids[c,i,b]].
 *
 * A row with configs[b].temperature<=0 selects the lowest candidate rank attaining max(edge) and
 * writes its exact one-hot distribution. A positive-temperature row writes the FP32 softmax of
 * edge/temperature, then draws a candidate with counter key
 * (configs[b].seed,base_positions[b]+i,kSamplePurposeDFlash2Proposal). The selected global id is
 * written to drafts[i,b] and becomes the next predecessor. The Op ignores all other
 * SamplingConfig fields and never updates token_counts.
 *
 * drafts is contiguous I32 [K,B] and proposal_q is contiguous FP32 [16,K,B]. Both outputs are
 * completely overwritten. Inputs, outputs, codebooks, and the config array must be pairwise
 * non-overlapping. The Op has no persistent state or internal allocation. Caller workspace is
 * transient and must not overlap any input or output.
 */
void candidate_selector_path(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& base_positions, const SamplingConfig* configs,
                             Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                             cudaStream_t stream);

// Capacity for every K/B pair in the inclusive intervals, K=1..15 and B=1..8.
[[nodiscard]] std::size_t candidate_selector_tree_workspace_capacity_bytes(std::int32_t min_steps,
                                                                           std::int32_t max_steps,
                                                                           std::int32_t min_batch,
                                                                           std::int32_t max_batch);

/**
 * Op: candidate_selector_tree
 *
 * Grows one speculative verify tree per row over the same selector lattice as
 * candidate_selector_path (inputs as there; no sampling config). A lattice node is candidate c
 * of step i reached from its parent node (the anchor for step 0); its conditional probability is
 * the softmax over the 16 candidates of edge/1.5 given the parent's candidate, and its score the
 * sum of log-probabilities along its root path. With P=clamp(current_extents[b],0,K), the tree
 * takes the P best-scoring nodes best-first (at depth <= P; ties by push order), which are the P
 * most likely root paths' union.
 *
 * An optional lookup chain proposes tokens lookup_tokens[0:n,b] for steps 0..n-1, n =
 * clamp(lookup_counts[b],0,K), each worth lookup_log_probability[b] (a log-probability <= 0): a
 * chain node at depth d scores d * lookup_log_probability[b]. A lattice node under a chain parent
 * whose candidate is the chain token takes the better of its two scores; otherwise the chain
 * continues as nodes outside the lattice whose only child is the next chain token. The three
 * lookup tensors are all empty (no chain) or I32 [K,B], I32 [B] and FP32 [B].
 *
 * Outputs, for the root column 0 and node columns 1..P in DFS pre-order (children by increasing
 * subtree size, so every node's largest subtree comes last): drafts[c-1,b] is column c's token,
 * tree_parents[c,b] its parent column (-1 at the root), tree_masks[c,b] its ancestor-or-self
 * column mask, and rope_positions[c,b] = rope_positions[0,b] + depth(c). Columns beyond P get
 * tree_parents = c-1, tree_masks = 1<<c and keep their drafts and RoPE positions. drafts is I32
 * [K,B]; tree_parents, tree_masks and rope_positions are I32 [K+1,B]; rope_positions[0,b] is an
 * input. Workspace per candidate_selector_tree_workspace_capacity_bytes().
 */
void candidate_selector_tree(const Tensor& candidate_ids, const Tensor& unary_scores,
                             const Tensor& projected_hidden, const Tensor& anchors,
                             const Tensor& predecessor_codebook, const Tensor& successor_codebook,
                             const Tensor& current_extents, const Tensor& lookup_tokens,
                             const Tensor& lookup_counts, const Tensor& lookup_log_probability,
                             Tensor& drafts, Tensor& tree_parents, Tensor& tree_masks,
                             Tensor& rope_positions, WorkspaceArena& workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops
