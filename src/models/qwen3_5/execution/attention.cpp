// Modified by satellitedown for Cinference: fused input norm-projection and q/k norm-RoPE.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "models/qwen3_5/execution/attention.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/qk_rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_attn_input_proj.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

void require_rope_axes(const Tensor& positions, const RopeConfig& config) {
    if (positions.ne[1] != 3) { return; }
    for (std::size_t i = 0; i < config.pair_axes.size(); ++i) {
        if (config.pair_axes[i] != i % 3) {
            throw std::invalid_argument("text RoPE: this MRoPE axis mapping has no native route");
        }
    }
}

} // namespace

std::size_t attention_norm_projection_workspace_bytes(const AttentionParameters& parameters,
                                                      std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("attention projection: invalid column interval");
    }
    if (const auto* single = std::get_if<LinearParameters>(&parameters.projection)) {
        return ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(single->weight, single->policy,
                                                                     first, last);
    }
    return 0;
}

void attention_norm_projection(const Tensor& residual, const Tensor& norm, float eps,
                               const AttentionParameters& parameters, Tensor& hidden, Tensor& query,
                               Tensor& gate, Tensor& key, Tensor& value, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    if (const auto* pair = std::get_if<ops::PairedProjectionWeights>(&parameters.projection)) {
        ops::rmsnorm(residual, norm, eps, true, hidden, stream);
        ops::attn_input_proj(hidden, pair->first, pair->second, query, gate, key, value, stream);
    } else {
        const auto& single = std::get<LinearParameters>(parameters.projection);
        ops::rmsnorm_attn_input_proj(residual, norm, eps, true, single.weight, query, gate, key,
                                     value, single.policy, workspace, stream);
    }
}

void text_rope(const Tensor& positions, const RopeConfig& config, Tensor& query,
               cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::rope(positions, dimension(config.rotary_dim), config.rope_theta, query, stream);
}

void text_qk_rmsnorm_rope(const Tensor& positions, const RopeConfig& config, const Tensor& query,
                          const Tensor& key, const Tensor& query_norm, const Tensor& key_norm,
                          float eps, Tensor& normalized_query, Tensor& normalized_key,
                          cudaStream_t stream) {
    require_rope_axes(positions, config);
    ops::qk_rmsnorm_rope(query, key, query_norm, key_norm, eps, true, positions,
                         dimension(config.rotary_dim), config.rope_theta, normalized_query,
                         normalized_key, stream);
}

} // namespace ninfer::models::qwen3_5::execution
