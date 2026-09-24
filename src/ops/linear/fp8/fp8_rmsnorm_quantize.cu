#include "ops/linear/fp8/fp8_a8_plan.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/fp8/fp8_a8_quantize.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;
constexpr int kPairs = kFp8RmsNormQuantizeWidth / 2 / kBlock;
static_assert(kFp8RmsNormQuantizeWidth / 2 == kBlock * kPairs);

// Hands launch_fp8_a8_quantize's operand for each normalized row straight to the A8 GEMM: the
// normalized BF16 pairs wait in shared memory while the CTA forms the row maximum, then are encoded
// with the same scale and conversions. Each thread reads back only the pairs it stored.
struct RmsFp8RowOutput {
    static constexpr int kRowSlices = 1;

    std::uint8_t* codes;
    float* scales;
    float maximum = 0.0F;

    __device__ static __nv_bfloat162* row_pairs() {
        __shared__ __nv_bfloat162 pairs[kBlock * kPairs];
        return pairs;
    }

    __device__ __forceinline__ void store(std::int64_t, int pair, __nv_bfloat162 value) {
        row_pairs()[pair]   = value;
        const float2 values = __bfloat1622float2(value);
        maximum             = fmaxf(maximum, fabsf(values.x));
        maximum             = fmaxf(maximum, fabsf(values.y));
    }

    __device__ __forceinline__ void finish_row(std::int64_t row) {
        constexpr int kWarps = kBlock / kWarpSize;
        __shared__ float warp_maxima[kWarps];
        __shared__ float row_scale;
        const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
        const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
        float value    = warp_max(maximum);
        if (lane == 0) { warp_maxima[warp] = value; }
        __syncthreads();
        if (warp == 0) {
            value = lane < kWarps ? warp_maxima[lane] : 0.0F;
            value = warp_max(value);
            if (lane == 0) { row_scale = fp8_a8_token_scale(value); }
        }
        __syncthreads();
        const float scale   = row_scale;
        const float inverse = fp8_a8_inverse_scale(scale);
        auto* row_codes     = reinterpret_cast<std::uint16_t*>(codes) + row * (kBlock * kPairs);
#pragma unroll
        for (int k = 0; k < kPairs; ++k) {
            const int pair  = static_cast<int>(threadIdx.x) + k * kBlock;
            row_codes[pair] = fp8_a8_encode_pair(__bfloat1622float2(row_pairs()[pair]), inverse);
        }
        if (threadIdx.x == 0) { scales[row] = scale; }
    }
};

// The instantiation ops::rmsnorm launches for aligned 5120-wide rows, with the encoding output.
template <RmsEpilogue Epilogue>
void launch(const Tensor& x, const Tensor& norm_weight, float eps, RmsFp8RowOutput output,
            cudaStream_t stream) {
    rmsnorm_cta_bf16x2_kernel<Epilogue, kBlock, kPairs, true, kFp8RmsNormQuantizeWidth,
                              RmsFp8RowOutput>
        <<<static_cast<unsigned>(x.ne[1]), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(x.data),
            static_cast<const __nv_bfloat162*>(norm_weight.data), nullptr, output,
            kFp8RmsNormQuantizeWidth, x.ne[1], eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_rmsnorm_quantize_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                 bool unit_offset, Fp8A8Workspace out, cudaStream_t stream) {
    if (x.ne[0] != kFp8RmsNormQuantizeWidth || x.ne[1] <= 0 || out.codes == nullptr ||
        out.scales == nullptr) {
        throw std::invalid_argument("fp8 rmsnorm quantize: unsupported row width or workspace");
    }
    const RmsFp8RowOutput output{out.codes, out.scales};
    if (unit_offset) {
        launch<RmsEpilogue::Offset>(x, norm_weight, eps, output, stream);
    } else {
        launch<RmsEpilogue::Plain>(x, norm_weight, eps, output, stream);
    }
}

} // namespace ninfer::ops::detail
