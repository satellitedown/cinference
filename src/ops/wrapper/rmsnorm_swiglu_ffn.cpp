#include "ninfer/ops/rmsnorm_swiglu_ffn.h"

#include "core/layout.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/rmsnorm_swiglu_ffn/nvfp4_rmsnorm_quantize.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

struct FfnProfile {
    const Weight& gate_up;
    LinearPolicy gate_up_policy;
    const Weight& down;
    LinearPolicy down_policy;
};

bool aligned4(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 3U) == 0;
}

void validate_profile(const FfnProfile& profile) {
    if (profile.gate_up.n <= 0 || (profile.gate_up.n % 2) != 0 ||
        profile.down.k != profile.gate_up.n / 2 || profile.down.n != profile.gate_up.k) {
        throw std::invalid_argument("rmsnorm_swiglu_ffn: gate/up and down shapes do not chain");
    }
}

// Both projections quantize their inputs to NVFP4 at this width, and the SwiGLU route can write its
// activation as the down projection's operand. The norm then writes the SwiGLU operand directly,
// so neither BF16 seam is materialized: the component quantize passes read exactly those seams.
bool quantized_route(const FfnProfile& profile, std::int32_t tokens) {
    return profile.gate_up.qtype == QType::NVFP4 && profile.down.qtype == QType::NVFP4 &&
           profile.gate_up.k == detail::kNvfp4RmsNormQuantizeWidth &&
           detail::nvfp4_linear_swiglu_quantizes_activation(profile.gate_up_policy, tokens) &&
           detail::nvfp4_linear_add_takes_quantized(profile.down.n, profile.down.k,
                                                    profile.down_policy, tokens);
}

// The FP8 SwiGLU quantizes its input to E4M3 rows at this width, so the norm writes those codes
// directly; the SwiGLU activation and the down projection stay as in the composition.
bool fp8_quantized_route(const FfnProfile& profile, std::int32_t tokens) {
    return profile.gate_up.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
           profile.gate_up.k == detail::kFp8RmsNormQuantizeWidth &&
           detail::fp8_linear_swiglu_uses_a8(profile.gate_up_policy, tokens);
}

struct Fp8QuantizedWorkspace {
    detail::Fp8A8Workspace hidden;
    Tensor activation;
};

template <class Allocator>
Fp8QuantizedWorkspace allocate_fp8_quantized(Allocator& allocator, const FfnProfile& profile,
                                             std::int32_t tokens) {
    Fp8QuantizedWorkspace out;
    out.hidden     = detail::allocate_fp8_a8_workspace(allocator, tokens, profile.gate_up.k);
    out.activation = allocator.alloc(DType::BF16, {profile.down.k, tokens}, 256);
    return out;
}

std::size_t fp8_quantized_bytes(const FfnProfile& profile, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fp8_quantized(layout, profile, tokens);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            linear_add_workspace_capacity_bytes(profile.down.qtype, profile.down.n, profile.down.k,
                                                profile.down_policy, tokens, tokens));
    }
    return layout.peak_bytes(1);
}

struct QuantizedWorkspace {
    detail::Nvfp4W4a4Workspace hidden;
    detail::Nvfp4W4a4Workspace activation;
};

template <class Allocator>
QuantizedWorkspace allocate_quantized(Allocator& allocator, const FfnProfile& profile,
                                      std::int32_t tokens) {
    QuantizedWorkspace out;
    out.hidden     = detail::allocate_nvfp4_w4a4_workspace(allocator, tokens, profile.gate_up.k);
    out.activation = detail::allocate_nvfp4_w4a4_workspace(allocator, tokens, profile.down.k);
    return out;
}

struct ComposedWorkspace {
    Tensor hidden;
    Tensor activation;
};

template <class Allocator>
ComposedWorkspace allocate_composed(Allocator& allocator, const FfnProfile& profile,
                                    std::int32_t tokens) {
    return {allocator.alloc(DType::BF16, {profile.gate_up.k, tokens}, 256),
            allocator.alloc(DType::BF16, {profile.down.k, tokens}, 256)};
}

std::size_t quantized_bytes(const FfnProfile& profile, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_quantized(layout, profile, tokens);
    return layout.peak_bytes(1);
}

std::size_t composed_bytes(const FfnProfile& profile, std::int32_t min_tokens,
                           std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_composed(layout, profile, max_tokens);
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(linear_swiglu_workspace_capacity_bytes(
            profile.gate_up.qtype, profile.gate_up.n, profile.gate_up.k, profile.gate_up_policy,
            min_tokens, max_tokens));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            linear_add_workspace_capacity_bytes(profile.down.qtype, profile.down.n, profile.down.k,
                                                profile.down_policy, min_tokens, max_tokens));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::size_t rmsnorm_swiglu_ffn_workspace_capacity_bytes(
    const Weight& gate_up_weight, LinearPolicy gate_up_policy, const Weight& down_weight,
    LinearPolicy down_policy, std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("rmsnorm_swiglu_ffn workspace: invalid token interval");
    }
    const FfnProfile profile{gate_up_weight, gate_up_policy, down_weight, down_policy};
    validate_profile(profile);
    // Each width takes one route: the NVFP4 quantized route covers one contiguous run of short
    // widths, the FP8 one a run whose peak grows with the width, and every remaining run of widths
    // composes over its own interval.
    std::size_t maximum             = 0;
    std::int32_t composed_first     = 0;
    std::int32_t composed_last      = 0;
    std::int32_t last_fp8_quantized = 0;
    const auto close_composed_run   = [&] {
        if (composed_first != 0) {
            maximum = std::max(maximum, composed_bytes(profile, composed_first, composed_last));
            composed_first = 0;
        }
    };
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        if (tokens <= detail::kNvfp4LinearSwiGluQuantizedMaxTokens &&
            quantized_route(profile, tokens)) {
            close_composed_run();
            maximum = std::max(maximum, quantized_bytes(profile, tokens));
        } else if (fp8_quantized_route(profile, tokens)) {
            close_composed_run();
            last_fp8_quantized = tokens;
        } else {
            if (composed_first == 0) { composed_first = tokens; }
            composed_last = tokens;
        }
    }
    close_composed_run();
    if (last_fp8_quantized != 0) {
        maximum = std::max(maximum, fp8_quantized_bytes(profile, last_fp8_quantized));
    }
    return maximum;
}

void rmsnorm_swiglu_ffn(Tensor& residual, const Tensor& norm_weight, float eps, bool unit_offset,
                        const Weight& gate_up_weight, LinearPolicy gate_up_policy,
                        const Weight& down_weight, LinearPolicy down_policy,
                        WorkspaceArena& workspace, cudaStream_t stream) {
    const FfnProfile profile{gate_up_weight, gate_up_policy, down_weight, down_policy};
    validate_profile(profile);
    const std::int32_t tokens = residual.ne[1];
    if (residual.dtype != DType::BF16 || residual.ne[0] != gate_up_weight.k || tokens <= 0 ||
        residual.ne[2] != 1 || residual.ne[3] != 1 || !residual.is_contiguous() ||
        !aligned4(residual.data)) {
        throw std::invalid_argument(
            "rmsnorm_swiglu_ffn: residual must be contiguous 4-byte aligned BF16 [K,T]");
    }
    if (norm_weight.dtype != DType::BF16 || norm_weight.ne[0] != gate_up_weight.k ||
        norm_weight.ne[1] != 1 || !norm_weight.is_contiguous() || !aligned4(norm_weight.data)) {
        throw std::invalid_argument(
            "rmsnorm_swiglu_ffn: norm weight must be contiguous 4-byte aligned BF16 [K]");
    }
    auto scope = workspace.scope();
    if (quantized_route(profile, tokens)) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 rmsnorm_swiglu_ffn gate/up");
        (void)detail::validate_nvfp4_weight(down_weight, "nvfp4 rmsnorm_swiglu_ffn down");
        const QuantizedWorkspace scratch = allocate_quantized(workspace, profile, tokens);
        detail::nvfp4_rmsnorm_quantize_launch(residual, norm_weight, eps, unit_offset,
                                              gate_up_weight.input_scale_divisor, scratch.hidden,
                                              stream);
        detail::nvfp4_linear_swiglu_w4a4_quantized_launch(scratch.hidden, gate_up_weight, tokens,
                                                          scratch.activation,
                                                          down_weight.input_scale_divisor, stream);
        detail::nvfp4_linear_add_w4a4_quantized_launch(down_weight, residual, scratch.activation,
                                                       tokens, stream);
        return;
    }
    if (fp8_quantized_route(profile, tokens)) {
        Fp8QuantizedWorkspace scratch = allocate_fp8_quantized(workspace, profile, tokens);
        detail::fp8_rmsnorm_quantize_launch(residual, norm_weight, eps, unit_offset, scratch.hidden,
                                            stream);
        detail::fp8_linear_swiglu_a8_quantized_launch(gate_up_weight, scratch.activation,
                                                      scratch.hidden, tokens, stream);
        linear_add(scratch.activation, down_weight, residual, down_policy, workspace, stream);
        return;
    }
    ComposedWorkspace scratch = allocate_composed(workspace, profile, tokens);
    rmsnorm(residual, norm_weight, eps, unit_offset, scratch.hidden, stream);
    {
        auto call = workspace.scope();
        linear_swiglu(scratch.hidden, gate_up_weight, scratch.activation, gate_up_policy, workspace,
                      stream);
    }
    linear_add(scratch.activation, down_weight, residual, down_policy, workspace, stream);
}

} // namespace ninfer::ops
