#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kRows = 5120, kGroups = 320;

__device__ __forceinline__ void finish_value(int row, int col, int width, float current,
                                             float previous, const __nv_bfloat16* base,
                                             const __nv_bfloat16* delta, __nv_bfloat16* residual) {
    const int index = col * kRows + row, di = col * 2 * kGroups + row / 16;
    float value = fmaf(__bfloat162float(base[2 * kRows + row]) + __bfloat162float(delta[di]),
                       current, __bfloat162float(residual[index]));
    if (col % width != 0)
        value =
            fmaf(__bfloat162float(base[3 * kRows + row]) + __bfloat162float(delta[di + kGroups]),
                 previous, value);
    residual[index] = __float2bfloat16_rn(value);
}

__global__ void finish_kernel(const __nv_bfloat16* projected, const __nv_bfloat16* base,
                              const __nv_bfloat16* delta, __nv_bfloat16* residual, int width) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    const int index = col * kRows + row;
    finish_value(row, col, width, __bfloat162float(projected[index]),
                 col % width ? __bfloat162float(projected[index - kRows]) : 0.0f, base, delta,
                 residual);
}

template <int Splits>
__device__ __forceinline__ float rounded_projection(const float* partials, int tokens, int col,
                                                    int row) {
    float sum = 0.0f;
#pragma unroll
    for (int s = 0; s < Splits; ++s)
        sum += partials[(static_cast<std::int64_t>(s) * tokens + col) * kRows + row];
    return __bfloat162float(__float2bfloat16_rn(sum));
}

template <int Splits>
__global__ void finish_partials_kernel(const float* partials, const __nv_bfloat16* base,
                                       const __nv_bfloat16* delta, __nv_bfloat16* residual,
                                       int width, int tokens) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (row >= kRows) return;
    finish_value(row, col, width, rounded_projection<Splits>(partials, tokens, col, row),
                 col % width ? rounded_projection<Splits>(partials, tokens, col - 1, row) : 0.0f,
                 base, delta, residual);
}
} // namespace

void dynamic_grouped_conv_add_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                            const Tensor& finish_delta, Tensor& residual,
                                            int width, int tokens, cudaStream_t stream) {
    const dim3 grid((kRows + 255) / 256, tokens);
    finish_kernel<<<grid, 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(projected.data),
                                            static_cast<const __nv_bfloat16*>(base_kernel.data),
                                            static_cast<const __nv_bfloat16*>(finish_delta.data),
                                            static_cast<__nv_bfloat16*>(residual.data), width);
    CUDA_CHECK(cudaGetLastError());
}

void dynamic_grouped_conv_add_finish_partials_launch(const float* partials, int splits,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     int width, int tokens, cudaStream_t stream) {
    const dim3 grid((kRows + 255) / 256, tokens);
    const auto* base  = static_cast<const __nv_bfloat16*>(base_kernel.data);
    const auto* delta = static_cast<const __nv_bfloat16*>(finish_delta.data);
    auto* output      = static_cast<__nv_bfloat16*>(residual.data);
    if (splits == 2)
        finish_partials_kernel<2><<<grid, 256, 0, stream>>>(partials, base, delta, output, width,
                                                            tokens);
    else if (splits == 4)
        finish_partials_kernel<4><<<grid, 256, 0, stream>>>(partials, base, delta, output, width,
                                                            tokens);
    else
        throw std::invalid_argument("dynamic grouped conv add finish: splits must be 2 or 4");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
