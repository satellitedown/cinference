#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

// K=25600 is 400 groups per row, beyond the static per-warp ownership of the 8-warp GEMV, so T=1
// uses the dynamic 4-rows-per-CTA GEMV. Two staging buffers fit the static limit through 16 columns.
Q4Launch select_q4_n5120_k25600(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<5120, 25600, 4, 2>;
    if (tokens <= 8) return launch_q4_ksplit<5120, 25600, 8, 2>;
    if (tokens <= 16) return launch_q4_ksplit<5120, 25600, 16, 2>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 25600, 24>;
    if (tokens <= 32) return launch_q4_ksplit<5120, 25600, 32>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
