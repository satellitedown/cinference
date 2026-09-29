#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Applies the two-tap finish convolution of a materialized BF16 projection z [5120,W*B] to the
// residual: residual += (base[.,0,1] + delta[.,0]) * z[i] + I(i>0) * (base[.,1,1] + delta[.,1]) *
// z[i-1], with the tap index restarting at every request of width W. Shared by every projection
// format so that the conv arithmetic is one definition.
void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                            const Tensor& finish_delta, Tensor& residual,
                                            int width, int tokens, cudaStream_t stream);

// The same finish for a K-split projection: z[i] is the BF16 rounding of the FP32 partial sums
// partials[(s * tokens + i) * 5120 + row] added in split order s = 0..splits-1 (2 or 4).
void dynamic_grouped_conv_add_finish_partials_launch(const float* partials, int splits,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     int width, int tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
