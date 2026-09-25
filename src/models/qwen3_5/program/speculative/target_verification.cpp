// Modified by satellitedown for Cinference: verify-tree acceptance.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/context.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::models::qwen3_5::execution {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::CausalAttentionExecutionEnvelope envelope) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    const bool tree = frame.tree_parents.data != nullptr;
    card.set_verify_tree(tree ? &frame.tree_parents : nullptr, tree ? &frame.tree_masks : nullptr);
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens, *frame.feature_sink);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.state_source_slots,
                                 envelope, frame.target_hidden, frame.target_logits,
                                 frame.target_tokens);
    }
    card.set_verify_tree(nullptr, nullptr);
    if (tree) {
        ops::speculative_accept_tree_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.tree_parents,
            frame.current_extents, frame.frontiers, frame.anchors, frame.licensed_tokens,
            frame.licensed_counts, frame.accepted_drafts, frame.accepted_columns,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
        // Later readers (the continuation hidden below, a partial commit's correction) index the
        // accepted path by its chain position.
        ops::speculative_compact_columns(frame.target_hidden, Tensor{}, frame.accepted_columns,
                                         frame.accepted_drafts, execution.device.stream);
    } else if (frame.proposal_q.data != nullptr) {
        ops::speculative_accept_sparse_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.candidate_ids,
            frame.proposal_q, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            {false}, execution.work, execution.device.stream);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts,
            dimension(execution.parameters.model.resources().public_token_count), frame.sampling,
            execution.work, execution.device.stream);
    }
    ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                            frame.selected_hidden, execution.device.stream);
    ops::scatter(frame.selected_hidden, frame.state_destination_slots, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::models::qwen3_5::execution
