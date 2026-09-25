// Modified by satellitedown for Cinference: speculative verify-tree acceptance.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

struct SpeculativeAcceptExecutionEnvelope {
    // Execution promise: every row has temperature<=0 and both penalties disabled. When false,
    // the general route remains valid for any supported mixture of greedy and stochastic rows.
    bool all_rows_greedy_without_penalties = false;
};

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to speculative_prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Independently for each row b, greedy mode accepts the longest available draft prefix matching
 *   the per-column penalty-adjusted argmax and commits that argmax at the first mismatch (or the
 *   bonus column). With both penalties disabled, target_tokens is the exact raw-logit fast path.
 *   Sampling mode applies configs[b] to each valid verification column, accepts draft i with
 *   target probability p_i(draft_i), samples from the residual distribution on first rejection,
 *   and samples a bonus from column Pcur[b] when every available draft is accepted. The draft
 *   proposal distribution is one-hot at each greedy draft token.
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B], drafts is
 *   I32 [K,B], logits is BF16 [physical_rows,K+1,B], and current_extents/lengths/anchors/
 *   licensed_counts/accepted are I32 [B]. token_domain is in [1,physical_rows], K>=1, B>=1, and
 *   configs points to a device-resident SamplingConfig[B]. Tensor arguments, configs, and
 *   configs[b].token_counts do not overlap except for the explicitly mutated objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   For each row, let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives
 *   accepted drafts, licensed_tokens[A,b] receives the correction/bonus token, and the remaining
 *   physical slots are zero. licensed_counts[b]=L; accepted[b]=A; anchors[b] becomes the
 *   correction/bonus token; lengths[b]+=L. In every mode, each produced token increments
 *   configs[b].token_counts when that pointer is non-null. current_extents and all other inputs
 *   remain unchanged. Request statistics are
 *   deliberately outside this Op.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes().
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, cudaStream_t stream);

// Caller-owned transient capacity over the inclusive draft-count and batch intervals.
// The raw-greedy execution envelope requires no workspace.
[[nodiscard]] std::size_t speculative_accept_sparse_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, SpeculativeAcceptExecutionEnvelope envelope, std::int32_t min_drafts,
    std::int32_t max_drafts, std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_accept_sparse_drafts
 *
 * Algorithm:
 *   This is the variable-K, 16-candidate form of speculative rejection sampling.
 *   For row b, let P=clamp(current_extents[b],0,K). Only target columns 0..P are live.
 *   Greedy rows accept the longest prefix matching the penalty-adjusted target argmax,
 *   then emit that argmax as correction/bonus. Positive-temperature rows construct p
 *   using sampling.h penalties and filters. A live draft d is accepted with probability
 *   min(1,p(d)/q(d)); first rejection samples normalized max(p-q,0). After accepting all
 *   P drafts, the terminal token is sampled from target column P.
 *
 * Logical shapes and registered profile:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B].
 *   logits is BF16 [248320,K+1,B]; drafts is I32 [K,B]; candidate_ids is I32 [16,K,B];
 *   proposal_q is FP32 [16,K,B]. current_extents, round_lengths, round_anchors,
 *   licensed_counts, and accepted_drafts are I32 [B].
 *   The registered domain is token_domain=248077, K=1..15, B=1..8. Each live draft
 *   position has distinct global candidate ids in [0,token_domain). proposal_q is the
 *   normalized FP32 distribution used to draw that draft; the draft occurs with positive q.
 *   For greedy rows without penalties, live target_tokens are the unpenalized target argmax
 *   over the valid token domain, with lower ids breaking ties.
 *
 * Numeric:
 *   proposal_q is consumed directly; it is not reconstructed from selector scores or expanded to
 *   a dense vocabulary distribution. Target logits are interpreted through sampling.h. Column i's
 *   penalty overlay is drafts[0..i-1], because the column is consumed only after that prefix was
 *   accepted. RNG purposes are the existing speculative accept/correction/bonus domains and use
 *   logical positions derived from the old round length.
 *
 * Effects:
 *   Let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives accepted drafts,
 *   licensed_tokens[A,b] receives the correction/bonus token, and the physical tail is zero.
 *   licensed_counts[b]=L, accepted_drafts[b]=A, round_anchors[b] becomes the correction/bonus
 *   token, and round_lengths[b]+=L. These length/anchor values are provisional round buffers.
 *   configs and configs[b].token_counts are read-only; persistent token counts and model state are
 *   committed only after the caller chooses a final prefix of the licensed output. All other
 *   inputs remain unchanged.
 *
 * Execution:
 *   all_rows_greedy_without_penalties=true promises the matching device configs and enables the
 *   raw target_tokens route. A false flag selects the general route and supports mixed rows.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_sparse_drafts_workspace_capacity_bytes().
 */
void speculative_accept_sparse_drafts(
    const Tensor& target_tokens, const Tensor& logits, const Tensor& drafts,
    const Tensor& candidate_ids, const Tensor& proposal_q, const Tensor& current_extents,
    Tensor& round_lengths, Tensor& round_anchors, Tensor& licensed_tokens, Tensor& licensed_counts,
    Tensor& accepted_drafts, std::int32_t token_domain, const SamplingConfig* configs,
    SpeculativeAcceptExecutionEnvelope envelope, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_accept_tree_drafts
 *
 * Algorithm:
 *   Verify-tree form of speculative acceptance. For row b let P=clamp(current_extents[b],0,K);
 *   columns 0..P are live. Column 0 is the root (the round anchor) and live column c>=1 is a
 *   drafted node whose token is drafts[c-1,b] and whose parent is column tree_parents[c,b] < c;
 *   children of one node carry distinct tokens. Starting at the root, the walk takes the target's
 *   token for the current node's column and moves into the child carrying it; the first node
 *   without such a child ends the walk and its target token is the correction/bonus terminal.
 *   Greedy rows use the penalty-adjusted target argmax. Positive-temperature rows sample every
 *   visited node's token from its sampling.h distribution (RNG purpose speculative bonus, logical
 *   position old length + node depth + 1), which keeps the output distribution exactly the
 *   target's for a deterministic tree. Column c's penalty overlay is the drafted tokens on c's
 *   root path (c included).
 *
 * Logical shapes and registered profile:
 *   As speculative_accept_sparse_drafts, without candidate_ids/proposal_q, plus tree_parents I32
 *   [K+1,B] and accepted_columns I32 [K+1,B].
 *
 * Effects:
 *   Let A be the accepted path length and L=A+1. licensed_tokens[0:A,b] receives the path's
 *   drafts, licensed_tokens[A,b] the terminal, and the physical tail is zero. accepted_columns
 *   [j,b] is the column of the path's j-th node (the root at j=0) for j<=A and j beyond it.
 *   licensed_counts, accepted_drafts, round_anchors and round_lengths are as in the sparse form.
 *
 * Workspace:
 *   speculative_accept_sparse_drafts_workspace_capacity_bytes() of the same profile.
 */
void speculative_accept_tree_drafts(const Tensor& target_tokens, const Tensor& logits,
                                    const Tensor& drafts, const Tensor& tree_parents,
                                    const Tensor& current_extents, Tensor& round_lengths,
                                    Tensor& round_anchors, Tensor& licensed_tokens,
                                    Tensor& licensed_counts, Tensor& accepted_drafts,
                                    Tensor& accepted_columns, std::int32_t token_domain,
                                    const SamplingConfig* configs,
                                    SpeculativeAcceptExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_compact_columns
 *
 * Moves each row's accepted verify-tree path onto the chain columns in place: for batch row b
 * with A = accepted[b] and r = rows[b] (b when rows is empty), values[:, j, r] receives the
 * pre-call values[:, accepted_columns[j,b], r] for 1 <= j <= A. values is contiguous BF16 [D,W,R];
 * accepted_columns is I32 [W,B] with the path layout of speculative_accept_tree_drafts; accepted
 * and rows are I32 [B]. Other columns and rows are unchanged.
 */
void speculative_compact_columns(Tensor& values, const Tensor& rows, const Tensor& accepted_columns,
                                 const Tensor& accepted, cudaStream_t stream);

/**
 * Op: speculative_compact_k8v4_kv
 *
 * The K8V4 paged-cache counterpart for every given attention layer: for 1 <= j <= accepted[b],
 * row b's cache row at cache_positions[accepted_columns[j,b],b] moves to cache_positions[j,b]
 * (FP8 K codes and scale, NVFP4 V codes and scales, all KV heads), through table row
 * kv_table_rows[b]. cache_positions and accepted_columns are I32 [W,B], the verify columns' cache
 * positions and the path layout of speculative_accept_tree_drafts. The layers share one block
 * table. Other cache rows are unchanged.
 */
void speculative_compact_k8v4_kv(std::span<const PagedKVBatchLayerView> layers,
                                 const Tensor& kv_table_rows, const Tensor& cache_positions,
                                 const Tensor& accepted_columns, const Tensor& accepted,
                                 cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

} // namespace ninfer::ops
