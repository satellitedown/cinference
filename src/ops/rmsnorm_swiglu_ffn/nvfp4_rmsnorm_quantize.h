#pragma once

#include "core/tensor.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Width of the rows this launcher normalizes.
inline constexpr std::int32_t kNvfp4RmsNormQuantizeWidth = 5120;

// Normalizes each BF16 [5120] row of x exactly as ops::rmsnorm does and writes the BF16 result as
// the W4A4 operand of the following NVFP4 projection: its codes and row-major group scales,
// quantized with input_scale_divisor exactly as launch_nvfp4_w4a4_quantize does.
void nvfp4_rmsnorm_quantize_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                   bool unit_offset, float input_scale_divisor,
                                   Nvfp4W4a4Workspace out, cudaStream_t stream);

} // namespace ninfer::ops::detail
