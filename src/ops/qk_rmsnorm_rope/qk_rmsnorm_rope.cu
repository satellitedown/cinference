#include "ops/qk_rmsnorm_rope/qk_rmsnorm_rope.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim     = 256;
constexpr int kHeadPairs   = kHeadDim / 2;
constexpr int kRotaryPairs = 16;
constexpr int kRowsPerCta  = 4;

// One warp owns one head row of one token, query heads first. The normalization is
// rmsnorm_warp_bf16x2_kernel's for this width: lane l holds pairs l + 32k, the lane partials are
// reduced by warp_reduce_sum and lane 0 forms the factor. The rotated pairs [0, 32) are all k = 0,
// so lane p < 16 holds first-half pair p and lane 16 + p its partner, and each lane rotates its own
// pair from the rounded BF16 values with the coefficients rope_fixed_kernel gives pair p.
template <RmsEpilogue Epilogue, RopeKernelMode Mode, int QHeads, int KHeads>
__global__ __launch_bounds__(kRowsPerCta * 32) void qk_rmsnorm_rope_kernel(
    const __nv_bfloat162* q, const __nv_bfloat162* k, const __nv_bfloat162* q_weight,
    const __nv_bfloat162* k_weight, __nv_bfloat162* q_out, __nv_bfloat162* k_out,
    const std::int32_t* positions, std::int32_t tokens, float eps) {
    constexpr int kHeads = QHeads + KHeads;
    const int lane       = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int row =
        static_cast<int>(blockIdx.x) * kRowsPerCta + static_cast<int>(threadIdx.x) / kWarpSize;
    const int token = row / kHeads;
    const int head  = row - token * kHeads;
    if (token >= tokens) { return; }
    const bool query = head < QHeads;
    const std::int64_t row_base =
        query ? (static_cast<std::int64_t>(token) * QHeads + head) * kHeadPairs
              : (static_cast<std::int64_t>(token) * KHeads + head - QHeads) * kHeadPairs;
    const __nv_bfloat162* x      = (query ? q : k) + row_base;
    const __nv_bfloat162* weight = query ? q_weight : k_weight;

    __nv_bfloat162 values[4];
    __nv_bfloat162 weights[4];
    float sum = 0.0f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int pair  = lane + step * kWarpSize;
        values[step]    = x[pair];
        weights[step]   = weight[pair];
        const float2 xf = __bfloat1622float2(values[step]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kHeadDim) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

    __nv_bfloat162 out[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const float2 xf = __bfloat1622float2(values[step]);
        const float2 wf = __bfloat1622float2(weights[step]);
        out[step]       = __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
                                                rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
    }

    const int pair = 2 * (lane & (kRotaryPairs - 1));
    float s0;
    float c0;
    float s1;
    float c1;
    fixed_sincos<Mode>(positions, tokens, token, pair, &s0, &c0);
    fixed_sincos<Mode>(positions, tokens, token, pair + 1, &s1, &c1);
    const float2 own      = __bfloat1622float2(out[0]);
    const float2 partner  = make_float2(__shfl_xor_sync(kFullWarpMask, own.x, kRotaryPairs),
                                        __shfl_xor_sync(kFullWarpMask, own.y, kRotaryPairs));
    const bool first_half = lane < kRotaryPairs;
    const float2 rotated_x =
        rope_rotate_pair(first_half ? own.x : partner.x, first_half ? partner.x : own.x, c0, s0);
    const float2 rotated_y =
        rope_rotate_pair(first_half ? own.y : partner.y, first_half ? partner.y : own.y, c1, s1);
    out[0] = first_half ? __floats2bfloat162_rn(rotated_x.x, rotated_y.x)
                        : __floats2bfloat162_rn(rotated_x.y, rotated_y.y);

    __nv_bfloat162* destination = (query ? q_out : k_out) + row_base;
#pragma unroll
    for (int step = 0; step < 4; ++step) { destination[lane + step * kWarpSize] = out[step]; }
}

template <RmsEpilogue Epilogue, RopeKernelMode Mode>
void launch(const Tensor& q, const Tensor& k, const Tensor& q_weight, const Tensor& k_weight,
            float eps, const Tensor& positions, Tensor& q_out, Tensor& k_out, cudaStream_t stream) {
    constexpr int kHeads = kQkRmsNormRopeQueryHeads + kQkRmsNormRopeKeyHeads;
    const int tokens     = q.ne[2];
    const int rows       = tokens * kHeads;
    qk_rmsnorm_rope_kernel<Epilogue, Mode, kQkRmsNormRopeQueryHeads, kQkRmsNormRopeKeyHeads>
        <<<(rows + kRowsPerCta - 1) / kRowsPerCta, kRowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(q.data), static_cast<const __nv_bfloat162*>(k.data),
            static_cast<const __nv_bfloat162*>(q_weight.data),
            static_cast<const __nv_bfloat162*>(k_weight.data),
            static_cast<__nv_bfloat162*>(q_out.data), static_cast<__nv_bfloat162*>(k_out.data),
            static_cast<const std::int32_t*>(positions.data), tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void qk_rmsnorm_rope_launch(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                            const Tensor& k_norm_weight, float eps, bool unit_offset,
                            const Tensor& positions, Tensor& q_out, Tensor& k_out,
                            cudaStream_t stream) {
    const bool mrope = positions.ne[1] == 3;
    if (unit_offset) {
        if (mrope) {
            launch<RmsEpilogue::Offset, RopeKernelMode::TextMrope>(
                q, k, q_norm_weight, k_norm_weight, eps, positions, q_out, k_out, stream);
        } else {
            launch<RmsEpilogue::Offset, RopeKernelMode::Text1D>(
                q, k, q_norm_weight, k_norm_weight, eps, positions, q_out, k_out, stream);
        }
    } else if (mrope) {
        launch<RmsEpilogue::Plain, RopeKernelMode::TextMrope>(q, k, q_norm_weight, k_norm_weight,
                                                              eps, positions, q_out, k_out, stream);
    } else {
        launch<RmsEpilogue::Plain, RopeKernelMode::Text1D>(q, k, q_norm_weight, k_norm_weight, eps,
                                                           positions, q_out, k_out, stream);
    }
}

} // namespace ninfer::ops::detail
