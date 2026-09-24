#pragma once

// ninfer::ops - residual += Down(SwiGLU(GateUp(RMSNorm(residual)))).

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient capacity required by RMSNormSwiGluFfn for every T in the inclusive
 * [min_tokens,max_tokens] interval under the given weight profiles and policies. Invalid
 * profiles, policies or intervals throw.
 */
[[nodiscard]] std::size_t rmsnorm_swiglu_ffn_workspace_capacity_bytes(
    const Weight& gate_up_weight, LinearPolicy gate_up_policy, const Weight& down_weight,
    LinearPolicy down_policy, std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: rmsnorm_swiglu_ffn
 *
 * Math / indexing:
 *   h        = RMSNorm(residual; norm_weight, eps, unit_offset)    as ops::rmsnorm defines it;
 *   a        = LinearSwiGLU(h; gate_up_weight, gate_up_policy)     as ops::linear_swiglu does;
 *   residual = LinearAdd(a; down_weight, down_policy) + residual   as ops::linear_add does.
 *
 * Logical shapes / supported domain:
 *   residual is contiguous BF16 [K,T] for any positive T, norm_weight contiguous BF16 [K], both
 *   4-byte aligned, and the weights are any pair ops::linear_swiglu and ops::linear_add register
 *   with gate_up_weight [2M,K] and down_weight [K,M].
 *
 * Numeric:
 *   h and a are BF16 seams exactly as the component Ops round them. Every route produces the
 *   residual bits of rmsnorm, linear_swiglu and linear_add called in sequence with the same
 *   arguments, so each component's oracle and criterion apply unchanged. A route may keep h or a
 *   only in the quantized form its policy lets the following projection consume.
 *
 * Effects:
 *   Updates residual in place. Weights, norm_weight and workspace must not overlap residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by rmsnorm_swiglu_ffn_workspace_capacity_bytes(),
 *   scoped to the call. There is no persistent state side effect.
 */
void rmsnorm_swiglu_ffn(Tensor& residual, const Tensor& norm_weight, float eps, bool unit_offset,
                        const Weight& gate_up_weight, LinearPolicy gate_up_policy,
                        const Weight& down_weight, LinearPolicy down_policy,
                        WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
