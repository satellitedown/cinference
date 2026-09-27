#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_add.h"

#include "core/device.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;
// Widest K-split tile; wider verify blocks use the row-split MMA route.
constexpr int kKSplitColumns = 32;

enum class Route { KSplit, Mma };

struct Plan {
    Route route;
    std::size_t workspace_bytes;
};

Plan resolve_plan(int input_rows, int width, int batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    return {columns <= kKSplitColumns ? Route::KSplit : Route::Mma,
            static_cast<std::size_t>(kRows) * columns * sizeof(std::uint16_t)};
}

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// One 16-row tile per CTA: 5120 rows give 320 CTAs, which keeps every SM streaming weights at
// verify widths (multi-tile CTAs would leave most SMs with a single CTA). Two staging buffers keep
// the next K group's copies in flight wherever they fit the static shared-memory limit.
template <int InputRows, int Capacity>
void ksplit_projection(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry        = Q4LinearGeometry<kRows, InputRows>;
    constexpr int kStages = Capacity <= 16 ? 2 : 1;
    q4_ksplit_mma_kernel<Geometry, Capacity, Capacity, Q4KSplitStoreEpilogue,
                         Q4KSplitIdentityRows, true, 1, kStages>
        <<<kRows / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            {}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int InputRows>
constexpr std::array<Launch, 4> kKSplit{
    &ksplit_projection<InputRows, 8>, &ksplit_projection<InputRows, 16>,
    &ksplit_projection<InputRows, 24>, &ksplit_projection<InputRows, 32>};

using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

std::size_t q4_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch, std::int32_t max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    // Every route shares one projected BF16 matrix; its capacity is monotonic in W and B.
    return resolve_plan(input_rows, max_width, max_batch).workspace_bytes;
}

const char* q4_linear_dynamic_grouped_conv_add_route_name(std::int32_t input_rows,
                                                          std::int32_t width,
                                                          std::int32_t batch_size) {
    switch (resolve_plan(input_rows, width, batch_size).route) {
    case Route::KSplit:
        return "dynamic_grouped_conv_add.q4.ksplit.materialized_bf16";
    case Route::Mma:
        return "dynamic_grouped_conv_add.q4.rowsplit_mma.materialized_bf16";
    }
    throw std::logic_error("linear dynamic grouped conv add: invalid Q4 route");
}

void q4_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base_kernel,
                                                 const Tensor& finish_delta, Tensor& residual,
                                                 WorkspaceArena& workspace, cudaStream_t stream) {
    const Plan plan          = resolve_plan(x.ne[0], x.ne[1], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    const int tokens         = x.ne[1] * x.ne[2];
    const Tensor flat        = x.view({x.ne[0], tokens});
    Tensor projected(storage.data, DType::BF16, {kRows, tokens});
    if (plan.route == Route::KSplit) {
        const auto& launchers = x.ne[0] == 4096 ? kKSplit<4096> : kKSplit<17408>;
        launchers[(tokens - 1) / 8](flat, weight, projected, stream);
    } else if (tokens <= 96) {
        launch_q4_mma<MmaR32C32>(flat, weight, projected, stream);
    } else {
        launch_q4_mma<MmaR32C64>(flat, weight, projected, stream);
    }
    dynamic_grouped_conv_add_finish_launch(projected, base_kernel, finish_delta, residual,
                                           x.ne[1], tokens, stream);
}

} // namespace ninfer::ops::detail
