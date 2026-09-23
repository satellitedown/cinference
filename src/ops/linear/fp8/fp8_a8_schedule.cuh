#pragma once
#include "ops/linear/fp8/fp8_a8_mma.cuh"

namespace ninfer::ops::detail {
using Fp8A8DefaultSchedule =
    Fp8MmaSchedule<64, 128, 128, 2, 4, 2, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;

// Speculative verify widths (T<=16) are weight-bandwidth bound. A 16-token tile removes the idle
// token half of the default tile, and 64-row tiles with four stages keep enough weight bytes in
// flight for the small-N projections to stream near the device read bandwidth.
inline constexpr int kFp8A8SmallTokenLimit = 16;
using Fp8A8SmallTokenSchedule =
    Fp8MmaSchedule<16, 64, 128, 1, 4, 4, 2, Cache::cg, Cache::cg, Fp8MmaFragmentPipeline::PingPong,
                   Fp8MmaRaster::TokenFast>;
} // namespace ninfer::ops::detail
