// Modified by satellitedown for Cinference: add the residual-add output policy for the K-split MMA.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8AddResidualEpilogue {
    const __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return value + __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]);
    }
};

// Output policy for row-oriented Tensor Core mainloops that hand over the scaled FP32 projection.
// The BF16 residual is read and replaced in place, with the same FP32 sum and final BF16 rounding
// as Fp8AddResidualEpilogue followed by a contiguous BF16 store.
struct Fp8AddResidualOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float value) const {
        __nv_bfloat16* element = data + static_cast<std::int64_t>(token) * rows + row;
        *element               = __float2bfloat16_rn(value + __bfloat162float(*element));
    }
};

} // namespace ninfer::ops::detail
