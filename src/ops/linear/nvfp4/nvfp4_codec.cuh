// Modified by satellitedown for Cinference: share the group quantization steps with a lane form.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}

// A 16-element group's E4M3 scale: its largest magnitude times the input divisor, over 6 (the
// largest E2M1 magnitude). A zero scale means every code of the group is zero.
__device__ __forceinline__ std::uint8_t nvfp4_group_scale(float max_abs,
                                                          float input_scale_divisor) {
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    return __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
}

// The value an element is rounded from to E2M1 under a nonzero decoded group scale.
__device__ __forceinline__ float nvfp4_code_input(float value, float input_scale_divisor,
                                                  float decoded_scale) {
    return __fdiv_rn(value * input_scale_divisor, decoded_scale);
}

// One code byte in the low bits: the low nibble encodes lo, the high nibble hi.
__device__ __forceinline__ std::uint32_t pack_nvfp4_e2m1x2(float lo, float hi) {
    std::uint32_t result;
    asm("{\n"
        ".reg .b8 b0;\n"
        ".reg .b16 h0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "mov.b16 h0, {b0, b0};\n"
        "cvt.u32.u16 %0, h0;\n"
        "}\n"
        : "=r"(result)
        : "f"(lo), "f"(hi));
    return result & 0xffU;
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    result.scale = nvfp4_group_scale(max_abs, input_scale_divisor);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = nvfp4_code_input(values[pair].x, input_scale_divisor, decoded_scale);
        values[pair].y = nvfp4_code_input(values[pair].y, input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

struct Nvfp4LaneGroupCodes {
    std::uint32_t codes;
    std::uint8_t scale;
};

// quantize_nvfp4_k16 for a 16-element group spread over Lanes consecutive lanes, each holding
// 8 / Lanes consecutive pairs. The group maximum is exchanged between the lanes; every lane
// receives the scale and its code bytes (pair i in byte i). All 32 lanes must call it together.
template <int Lanes>
__device__ __forceinline__ Nvfp4LaneGroupCodes
quantize_nvfp4_lanes(const float2 (&values)[8 / Lanes], float input_scale_divisor) {
    static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8);
    constexpr int kPairs = 8 / Lanes;
    float max_abs        = 0.0F;
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
        max_abs = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs = fmaxf(max_abs, fabsf(values[pair].y));
    }
#pragma unroll
    for (int offset = 1; offset < Lanes; offset <<= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffU, max_abs, offset));
    }

    Nvfp4LaneGroupCodes result{0, nvfp4_group_scale(max_abs, input_scale_divisor)};
    if (result.scale == 0) { return result; }
    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
        result.codes |=
            pack_nvfp4_e2m1x2(nvfp4_code_input(values[pair].x, input_scale_divisor, decoded_scale),
                              nvfp4_code_input(values[pair].y, input_scale_divisor, decoded_scale))
            << (8 * pair);
    }
    return result;
}

} // namespace ninfer::ops::detail
