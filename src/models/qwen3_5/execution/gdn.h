// Modified by satellitedown for Cinference: verify-tree parents for the record projection;
// record projection overlapped with the control dots and the recurrence.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "core/device.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t gdn_projection_workspace_bytes(const GdnParameters& parameters,
                                                         std::int32_t first, std::int32_t last);
[[nodiscard]] std::size_t gdn_snapshot_workspace_bytes(const GdnParameters& parameters,
                                                       const GdnConfig& config, std::int32_t batch,
                                                       std::int32_t first_width,
                                                       std::int32_t last_width);
[[nodiscard]] std::size_t gdn_record_workspace_bytes(const GdnParameters& parameters,
                                                     const GdnConfig& config, std::int32_t batch,
                                                     std::int32_t first_width,
                                                     std::int32_t last_width);
void gdn_projection(const Tensor& hidden, const GdnParameters& parameters, Tensor& qkv, Tensor& z,
                    WorkspaceArena& workspace, cudaStream_t stream);
void gdn_norm_control(const Tensor& residual, const Tensor& norm, float epsilon,
                      const GdnParameters& parameters, Tensor& hidden, Tensor& g, Tensor& beta,
                      WorkspaceArena& workspace, DeviceExecutionView execution);
void gdn_projection_snapshot(const Tensor& hidden, const GdnParameters& parameters,
                             const GdnConfig& config, Tensor& conv_states,
                             const Tensor& valid_columns, const Tensor& initial_slots,
                             const Tensor& destination_slots, Tensor& query, Tensor& key,
                             Tensor& value, Tensor& z, WorkspaceArena& workspace,
                             cudaStream_t stream);
// tree_parents is empty for chains or I32 [T,B] DFS pre-order verify trees; trees require the
// single-parent FP8 projection.
void gdn_projection_record(const Tensor& hidden, const GdnParameters& parameters,
                           const GdnConfig& config, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slots,
                           const Tensor& tree_parents, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                           cudaStream_t stream);

// Events of DeviceContext::concurrent that gdn_projection_record_overlapped records on the
// concurrent stream: g/beta are complete at kGdnControlReady, z at kGdnOutputGateReady.
inline constexpr std::size_t kGdnControlReady    = 2;
inline constexpr std::size_t kGdnOutputGateReady = 3;

// Whether a record block of B rows and W columns has the overlapped form below: the FP8 parent's
// single-block A8 record route and the fused norm-gating route.
[[nodiscard]] bool gdn_record_overlap_admits(const GdnParameters& parameters, std::int32_t batch,
                                             std::int32_t width);

// The record block's input norm, controls and projection in the overlapped form. On `stream`: the
// normalized residual's A8 activation (codes, scales), then the query/key/value half of the
// record projection. On device.concurrent.stream, forked from `stream`: the control dots (g,
// beta) beside that half, then the output-gate half (z) beside whatever `stream` runs next (the
// recurrence). Every output equals gdn_norm_control followed by gdn_projection_record bit for bit
// (h is not written). Before reading g/beta or z, `stream` must wait for kGdnControlReady and
// kGdnOutputGateReady; residual, codes and scales must stay unmodified until then.
void gdn_projection_record_overlapped(const Tensor& residual, const Tensor& norm, float epsilon,
                                      const GdnParameters& parameters, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slots,
                                      const Tensor& tree_parents, Tensor& conv_record,
                                      Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                      Tensor& g, Tensor& beta, Tensor& codes, Tensor& scales,
                                      const DeviceContext& device);

// Makes `stream` wait for one of the events above.
void gdn_join_concurrent(const DeviceContext& device, std::size_t event, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
