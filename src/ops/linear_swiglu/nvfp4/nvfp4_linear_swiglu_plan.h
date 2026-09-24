// Modified by satellitedown for Cinference: declare the quantized-activation W4A4 launcher.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);

// The W4A4 single-token-tile route can hand its activation straight to a following W4A4
// projection: given the quantized input, it writes the activation's codes and row-major scales
// (quantized with that projection's input divisor) instead of the BF16 activation.
inline constexpr std::int32_t kNvfp4LinearSwiGluQuantizedMaxTokens = 16;

// True when T takes that route under the policy.
[[nodiscard]] bool nvfp4_linear_swiglu_quantizes_activation(LinearPolicy policy,
                                                            std::int32_t tokens);

void nvfp4_linear_swiglu_w4a4_quantized_launch(Nvfp4W4a4Workspace input, const Weight& weight,
                                               std::int32_t tokens, Nvfp4W4a4Workspace activation,
                                               float activation_input_scale_divisor,
                                               cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
