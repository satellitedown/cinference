#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_add.h"

#include "core/device.h"
#include "ops/dynamic_grouped_conv/dynamic_grouped_conv_add_finish.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows = 5120;
// Widest K-split tile; wider verify blocks use the row-split MMA route.
constexpr int kKSplitColumns = 32;

enum class Route { KSplit, SplitK, Mma };

struct Plan {
    Route route;
    std::size_t workspace_bytes;
};

// Verify widths of one request split K across CTAs: 32-row, four-warp CTAs cut the staged
// activation re-reads in half and the K slices keep every SM streaming. Cold microbench (16
// columns): C=17408 48.2 -> 38.7 us (4 slices, 3 stages), C=4096 12.6 -> 11.4 us (2 slices).
constexpr int kSplitKColumns = 16;
constexpr int split_count(int input_rows) { return input_rows == 17408 ? 4 : 2; }

Plan resolve_plan(int input_rows, int width, int batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const int columns = width * batch;
    const std::size_t projected =
        static_cast<std::size_t>(kRows) * columns * sizeof(std::uint16_t);
    if (columns <= kSplitKColumns)
        return {Route::SplitK, static_cast<std::size_t>(split_count(input_rows)) * kRows *
                                   columns * sizeof(float)};
    return {columns <= kKSplitColumns ? Route::KSplit : Route::Mma, projected};
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

template <int InputRows, int Capacity>
void splitk_projection(const Tensor& x, const Weight& weight, float* partials,
                       cudaStream_t stream) {
    using Geometry            = Q4LinearGeometry<kRows, InputRows>;
    constexpr int kSplits     = split_count(InputRows);
    constexpr int kStages     = InputRows == 17408 ? 3 : 2;
    constexpr int kMinBlocks  = kStages == 3 ? 2 : 3;
    constexpr int kRowTiles   = 2;
    const int columns         = x.ne[1];
    const dim3 grid(kRows / (16 * kRowTiles), kSplits);
    q4_ksplit_mma_kernel<Geometry, Capacity, Capacity, Q4KSplitPartialEpilogue,
                         Q4KSplitIdentityRows, true, kRowTiles, kStages, 4, kSplits, kMinBlocks>
        <<<grid, 128, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                   static_cast<const std::uint8_t*>(weight.qdata),
                                   static_cast<const std::uint8_t*>(weight.scales), nullptr,
                                   Q4KSplitPartialEpilogue{partials, kRows, columns}, {}, columns);
    CUDA_CHECK(cudaGetLastError());
}

using SplitLaunch = void (*)(const Tensor&, const Weight&, float*, cudaStream_t);

template <int InputRows>
constexpr std::array<SplitLaunch, 2> kSplitK{&splitk_projection<InputRows, 8>,
                                             &splitk_projection<InputRows, 16>};

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
    // FP32 split partials at narrow widths outgrow the BF16 projection of wider ones, so the
    // capacity is the maximum over the whole W/B interval.
    std::size_t capacity = 0;
    for (std::int32_t width = min_width; width <= max_width; ++width)
        for (std::int32_t batch = min_batch; batch <= max_batch; ++batch)
            capacity = std::max(capacity, resolve_plan(input_rows, width, batch).workspace_bytes);
    return capacity;
}

const char* q4_linear_dynamic_grouped_conv_add_route_name(std::int32_t input_rows,
                                                          std::int32_t width,
                                                          std::int32_t batch_size) {
    switch (resolve_plan(input_rows, width, batch_size).route) {
    case Route::KSplit:
        return "dynamic_grouped_conv_add.q4.ksplit.materialized_bf16";
    case Route::SplitK:
        return "dynamic_grouped_conv_add.q4.splitk.fp32_partials";
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
    if (plan.route == Route::SplitK) {
        auto* partials        = static_cast<float*>(storage.data);
        const auto& launchers = x.ne[0] == 4096 ? kSplitK<4096> : kSplitK<17408>;
        launchers[(tokens - 1) / 8](flat, weight, partials, stream);
        dynamic_grouped_conv_add_finish_partials_launch(partials, split_count(x.ne[0]),
                                                        base_kernel, finish_delta, residual,
                                                        x.ne[1], tokens, stream);
        return;
    }
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
