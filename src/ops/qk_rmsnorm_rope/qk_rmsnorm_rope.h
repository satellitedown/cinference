#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The fused route's geometry: 256-wide heads, 24 query and 4 key heads, 64 rotated dimensions at
// theta 1e7 with 1-D or three-axis positions.
inline constexpr int kQkRmsNormRopeQueryHeads = 24;
inline constexpr int kQkRmsNormRopeKeyHeads   = 4;

void qk_rmsnorm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                            const Tensor& k_norm_weight, float eps, bool unit_offset,
                            const Tensor& positions, Tensor& q_out, Tensor& k_out,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
