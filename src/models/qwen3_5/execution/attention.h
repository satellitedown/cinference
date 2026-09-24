// Modified by satellitedown for Cinference: fused input norm-projection and q/k norm-RoPE.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t
attention_norm_projection_workspace_bytes(const AttentionParameters& parameters, std::int32_t first,
                                          std::int32_t last);
// Unit-offset RMSNorm of the residual then the Q/gate/K/V projection. `hidden` receives the
// normalized rows only where the projection reads them as BF16 (the paired-parent form).
void attention_norm_projection(const Tensor& residual, const Tensor& norm, float eps,
                               const AttentionParameters& parameters, Tensor& hidden, Tensor& query,
                               Tensor& gate, Tensor& key, Tensor& value, WorkspaceArena& workspace,
                               cudaStream_t stream);

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream);
// Unit-offset RMSNorm of every query and key head into the normalized tensors, then the configured
// RoPE on both.
void text_qk_rmsnorm_rope(const Tensor& positions, const RopeConfig& config, const Tensor& query,
                          const Tensor& key, const Tensor& query_norm, const Tensor& key_norm,
                          float eps, Tensor& normalized_query, Tensor& normalized_key,
                          cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
