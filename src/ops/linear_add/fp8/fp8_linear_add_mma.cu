#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Sixteen-row CTAs split K over eight warps, so the 320 CTAs of a [5120,K] projection keep the
// weight stream in flight while the MMA reuses each widened code across the whole token tile.
template <class Geometry, int TileTokens>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = Fp8A16KSplitSchedule<8, TileTokens, 2>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8AddResidualOutput output{static_cast<__nv_bfloat16*>(residual.data),
                                      Geometry::kOutputRows};
    fp8_a16_ksplit_mma_kernel<Geometry, TileTokens, Schedule, Fp8AddResidualOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_problem(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (x.ne[1] <= 16) {
        launch_tile<Geometry, 16>(x, weight, residual, stream);
    } else {
        launch_tile<Geometry, 24>(x, weight, residual, stream);
    }
}

} // namespace

void fp8_linear_add_mma_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream) {
    if (x.ne[1] < kFp8LinearAddFirstMmaTokens || x.ne[1] > kFp8LinearAddChunkTokens) {
        throw std::invalid_argument("fp8 linear_add MMA small-T: unsupported T");
    }
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launch_problem<Fp8N5120K6144>(x, weight, residual, stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launch_problem<Fp8N5120K17408>(x, weight, residual, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add MMA small-T: unsupported problem");
}

} // namespace ninfer::ops::detail
