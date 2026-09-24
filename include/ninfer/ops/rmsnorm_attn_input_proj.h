#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: rmsnorm_attn_input_proj
 *
 * attn_input_proj(rmsnorm(x, norm_weight, eps, unit_offset), weight, q, gate, k, v, policy) for
 * the single-parent Q/K/output-gate/V form. q, gate, k and v are bit-identical to that composition
 * run through the component Ops, whose contracts define the domain: x is contiguous 4-byte aligned
 * BF16 [K,T], norm_weight contiguous BF16 [K], and the outputs follow attn_input_proj. Where the
 * projection quantizes its activation to row-scaled E4M3 (FP8 parents on the A8 route), the norm
 * writes those codes directly and the normalized BF16 row is not materialized.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * rmsnorm_attn_input_proj_workspace_capacity_bytes(); it must not overlap any operand.
 */
[[nodiscard]] std::size_t
rmsnorm_attn_input_proj_workspace_capacity_bytes(const Weight& query_key_gate_value_weight,
                                                 LinearPolicy policy, std::int32_t min_tokens,
                                                 std::int32_t max_tokens);

void rmsnorm_attn_input_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                             bool unit_offset, const Weight& query_key_gate_value_weight, Tensor& q,
                             Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                             WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
