#include "ninfer/ops/rmsnorm_attn_input_proj.h"

#include "core/layout.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

// The FP8 projection quantizes its activation on this route, so the norm writes the codes itself.
bool fused_route(const Weight& weight, LinearPolicy policy, std::int32_t tokens) {
    return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
           weight.n == detail::Fp8N14336K5120::kOutputRows &&
           weight.k == detail::kFp8RmsNormQuantizeWidth &&
           detail::fp8_attn_input_uses_a8(policy, tokens);
}

std::size_t fused_bytes(const Weight& weight, std::int32_t tokens) {
    return detail::fp8_a8_workspace_capacity_bytes(tokens, weight.k);
}

template <class Allocator>
Tensor allocate_hidden(Allocator& allocator, const Weight& weight, std::int32_t tokens) {
    return allocator.alloc(DType::BF16, {weight.k, tokens}, 256);
}

std::size_t composed_bytes(const Weight& weight, LinearPolicy policy, std::int32_t min_tokens,
                           std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_hidden(layout, weight, max_tokens);
    (void)layout.alloc_bytes(attn_input_proj_workspace_capacity_bytes(
        weight.qtype, weight.n, weight.k, policy, min_tokens, max_tokens));
    return layout.peak_bytes(1);
}

void require_output(const Tensor& tensor, std::int32_t rows, std::int32_t tokens,
                    const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != tokens ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("rmsnorm_attn_input_proj: invalid ") + label);
    }
}

} // namespace

std::size_t rmsnorm_attn_input_proj_workspace_capacity_bytes(const Weight& weight,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("rmsnorm_attn_input_proj workspace: invalid token interval");
    }
    // Each width takes one route; every run of composed widths is sized over its own interval.
    std::size_t maximum         = 0;
    std::int32_t composed_first = 0;
    std::int32_t composed_last  = 0;
    std::int32_t last_fused     = 0;
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        if (fused_route(weight, policy, tokens)) {
            if (composed_first != 0) {
                maximum = std::max(maximum,
                                   composed_bytes(weight, policy, composed_first, composed_last));
                composed_first = 0;
            }
            last_fused = tokens;
        } else {
            if (composed_first == 0) { composed_first = tokens; }
            composed_last = tokens;
        }
    }
    if (composed_first != 0) {
        maximum = std::max(maximum, composed_bytes(weight, policy, composed_first, composed_last));
    }
    if (last_fused != 0) { maximum = std::max(maximum, fused_bytes(weight, last_fused)); }
    return maximum;
}

void rmsnorm_attn_input_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                             bool unit_offset, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena& workspace,
                             cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (x.dtype != DType::BF16 || x.ne[0] != weight.k || tokens <= 0 || x.ne[2] != 1 ||
        x.ne[3] != 1 || !x.is_contiguous() || !aligned_to(x.data, 4)) {
        throw std::invalid_argument(
            "rmsnorm_attn_input_proj: x must be contiguous 4-byte aligned BF16 [K,T]");
    }
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != weight.k ||
        norm_weight.ne[1] != 1 || !norm_weight.is_contiguous() ||
        !aligned_to(norm_weight.data, 4)) {
        throw std::invalid_argument(
            "rmsnorm_attn_input_proj: norm weight must be contiguous 4-byte aligned BF16 [K]");
    }
    auto scope = workspace.scope();
    if (fused_route(weight, policy, tokens)) {
        constexpr std::int32_t kQueryRows = 6144;
        constexpr std::int32_t kKeyRows   = 1024;
        (void)detail::validate_fp8_weight(weight, "fp8 rmsnorm_attn_input_proj");
        require_output(q, kQueryRows, tokens, "q");
        require_output(gate, kQueryRows, tokens, "gate");
        require_output(k, kKeyRows, tokens, "k");
        require_output(v, kKeyRows, tokens, "v");
        const detail::Fp8A8Workspace hidden =
            detail::allocate_fp8_a8_workspace(workspace, tokens, weight.k);
        detail::fp8_rmsnorm_quantize_launch(x, norm_weight, eps, unit_offset, hidden, stream);
        detail::fp8_attn_input_a8_quantized_launch(weight, q, gate, k, v, hidden, tokens, stream);
        return;
    }
    Tensor hidden = allocate_hidden(workspace, weight, tokens);
    rmsnorm(x, norm_weight, eps, unit_offset, hidden, stream);
    attn_input_proj(hidden, weight, q, gate, k, v, policy, workspace, stream);
}

} // namespace ninfer::ops
