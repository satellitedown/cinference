// Modified by satellitedown for Cinference: add and retune the small-token A8 MMA schedule.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once
#include "ops/linear/fp8/fp8_a8_mma.cuh"

namespace ninfer::ops::detail {
using Fp8A8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// Speculative verify widths (T<=16) are weight-bandwidth bound. A 16-token tile removes the idle
// token half of the default tile. 32-row tiles double the CTA count, so four resident CTAs per SM
// each keep a 256-wide K tile in flight behind the one they multiply. Every output keeps the same
// k32 MMA sequence under any row or K tiling.
inline constexpr int kFp8A8SmallTokenLimit = 16;
using Fp8A8SmallTokenSchedule =
    Fp8MmaSchedule<16, 32, 256, 1, 4, 2, 4, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;
} // namespace ninfer::ops::detail
