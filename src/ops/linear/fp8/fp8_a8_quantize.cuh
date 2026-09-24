#pragma once

// ninfer::ops - per-token E4M3 activation quantization shared by every A8 activation producer.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The scale of a token whose largest magnitude is `maximum` (zero for an all-zero token).
__device__ __forceinline__ float fp8_a8_token_scale(float maximum) {
    return maximum > 0.0F ? maximum / 448.0F : 0.0F;
}

__device__ __forceinline__ float fp8_a8_inverse_scale(float scale) {
    return scale > 0.0F ? 1.0F / scale : 0.0F;
}

// Two consecutive E4M3 codes of a token, low byte first.
__device__ __forceinline__ std::uint16_t fp8_a8_encode_pair(float2 values, float inverse) {
    const float2 scaled = make_float2(values.x * inverse, values.y * inverse);
    return __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
}

} // namespace ninfer::ops::detail
