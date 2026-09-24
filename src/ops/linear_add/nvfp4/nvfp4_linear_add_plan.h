// Modified by satellitedown for Cinference: declare the pre-quantized W4A4 launcher.
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

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream);

// The W4A4 route reads TMA-tiled activation scales from this width on and row-major ones below it,
// so whoever quantizes its input derives the scale layout from this same predicate.
inline constexpr bool nvfp4_linear_add_w4a4_tma_route(std::int32_t tokens) {
    return tokens >= 1024;
}

// True when T takes the row-major W4A4 route under the policy.
[[nodiscard]] bool nvfp4_linear_add_takes_quantized(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t tokens);

// W4A4 projection of an input the caller already quantized with this weight's input divisor into
// row-major codes and scales (the RowMajor quantize-pass layout, T below the TMA route).
void nvfp4_linear_add_w4a4_quantized_launch(const Weight& weight, Tensor& residual,
                                            Nvfp4W4a4Workspace activation, std::int32_t tokens,
                                            cudaStream_t stream);

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
