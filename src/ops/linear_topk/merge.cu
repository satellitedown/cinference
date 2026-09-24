// Modified by satellitedown for Cinference: merge top-k groups with one warp per output group.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kMergeWarps         = 8;
inline constexpr unsigned kMergeFullMask = 0xffffffffU;

static_assert(kLinearTopKMergeFanIn == 32, "one input group per lane");

__device__ __forceinline__ std::int64_t partial_offset(std::int32_t column, std::int32_t group,
                                                       std::int32_t rank,
                                                       std::int32_t group_stride) {
    return (static_cast<std::int64_t>(column) * group_stride + group) * kLinearTopK + rank;
}

// Sorts one lane's keys into descending order (bitonic network).
__device__ __forceinline__ void sort_lane_keys(std::uint64_t (&keys)[kLinearTopK]) {
#pragma unroll
    for (int size = 2; size <= kLinearTopK; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = 0; i < kLinearTopK; ++i) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool descending = (i & size) == 0;
                    const std::uint64_t a = keys[i];
                    const std::uint64_t b = keys[j];
                    const bool swap       = descending ? a < b : a > b;
                    keys[i]               = swap ? b : a;
                    keys[j]               = swap ? a : b;
                }
            }
        }
    }
}

// Lane l holds a descending list. Returns on lane r < kLinearTopK the r-th largest key of the 32
// lists. Five butterfly rounds merge partner lists: max(A[i], B[15 - i]) is a bitonic sequence
// holding the top kLinearTopK of both lists, which four compare-exchange layers sort descending.
// Afterwards every lane holds the same merged list.
__device__ __forceinline__ std::uint64_t merge_lane_lists(std::uint64_t (&keys)[kLinearTopK],
                                                          int lane) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::uint64_t partner[kLinearTopK];
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) {
            partner[i] = __shfl_xor_sync(kMergeFullMask, keys[i], offset);
        }
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) {
            const std::uint64_t other = partner[kLinearTopK - 1 - i];
            keys[i]                   = keys[i] > other ? keys[i] : other;
        }
#pragma unroll
        for (int stride = kLinearTopK / 2; stride > 0; stride >>= 1) {
#pragma unroll
            for (int i = 0; i < kLinearTopK; ++i) {
                if ((i & stride) == 0) {
                    const std::uint64_t a = keys[i];
                    const std::uint64_t b = keys[i + stride];
                    keys[i]               = a > b ? a : b;
                    keys[i + stride]      = a > b ? b : a;
                }
            }
        }
    }
    std::uint64_t result = 0;
#pragma unroll
    for (int i = 0; i < kLinearTopK; ++i) {
        if (lane == i) { result = keys[i]; }
    }
    return result;
}

// One warp per (output group, column): lane l reads input group 32 * output_group + l. The kept
// keys are the top kLinearTopK of the fan-in in descending ScoreIdOrderGreater order, the same
// keys a full sort of the fan-in keeps because keys are totally ordered and equal keys are equal.
// With Finalize, the last warp of a column merges every output group into the candidates.
template <bool SortedInput, bool Finalize>
__global__ __launch_bounds__(kMergeWarps * 32) void linear_topk_merge_kernel(
    const std::uint64_t* __restrict__ input_keys, std::uint64_t* __restrict__ output_keys,
    std::int32_t* __restrict__ group_done, std::int32_t* __restrict__ candidate_ids,
    float* __restrict__ candidate_scores, std::int32_t input_groups, std::int32_t output_groups,
    std::int32_t columns) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp =
        static_cast<int>(blockIdx.x) * kMergeWarps + static_cast<int>(threadIdx.x) / 32;
    const int column       = warp / output_groups;
    const int output_group = warp - column * output_groups;
    if (column >= columns) { return; }

    std::uint64_t keys[kLinearTopK];
    const int input_group = output_group * kLinearTopKMergeFanIn + lane;
    if (input_group < input_groups) {
        const auto* source = reinterpret_cast<const ulonglong2*>(
            input_keys + partial_offset(column, input_group, 0, input_groups));
#pragma unroll
        for (int i = 0; i < kLinearTopK / 2; ++i) {
            const ulonglong2 pair = source[i];
            keys[2 * i]           = pair.x;
            keys[2 * i + 1]       = pair.y;
        }
        if constexpr (!SortedInput) {
            sort_lane_keys(keys);
            // The sort consumed the loaded keys. A producer list is one 128-byte line that no
            // later read needs before it is written again, so it leaves L2 without a write-back.
            discard_l2_line(source);
        }
    } else {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) { keys[i] = 0; }
    }
    const std::uint64_t top = merge_lane_lists(keys, lane);
    if (lane < kLinearTopK) {
        output_keys[partial_offset(column, output_group, lane, output_groups)] = top;
    }
    if constexpr (!Finalize) { return; }

    __threadfence();
    __syncwarp();
    int is_last = 0;
    if (lane == 0) { is_last = atomicAdd(group_done + column, 1) + 1 == output_groups; }
    if (__shfl_sync(kMergeFullMask, is_last, 0) == 0) { return; }
    __threadfence();

    if (lane < output_groups) {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) {
            keys[i] = __ldcg(output_keys + partial_offset(column, lane, i, output_groups));
        }
    } else {
#pragma unroll
        for (int i = 0; i < kLinearTopK; ++i) { keys[i] = 0; }
    }
    const std::uint64_t key = merge_lane_lists(keys, lane);
    if (lane < kLinearTopK) {
        const std::int64_t out = static_cast<std::int64_t>(column) * kLinearTopK + lane;
        candidate_ids[out]     = id_from_order_key(key);
        candidate_scores[out]  = score_from_order_key(key);
    }
    if (lane == 0) { group_done[column] = 0; }
}

template <bool SortedInput, bool Finalize>
void launch_merge(const std::uint64_t* input_keys, std::uint64_t* output_keys,
                  std::int32_t* group_done, std::int32_t* candidate_ids, float* candidate_scores,
                  std::int32_t input_groups, std::int32_t output_groups, std::int32_t columns,
                  cudaStream_t stream) {
    const int warps = output_groups * columns;
    linear_topk_merge_kernel<SortedInput, Finalize>
        <<<(warps + kMergeWarps - 1) / kMergeWarps, kMergeWarps * 32, 0, stream>>>(
            input_keys, output_keys, group_done, candidate_ids, candidate_scores, input_groups,
            output_groups, columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_topk_merge_launch(const LinearTopKWorkspace& workspace, Tensor& candidate_ids,
                              Tensor& candidate_scores, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(workspace.group_done.data, 0, workspace.group_done.bytes(), stream));
    auto* const ids           = static_cast<std::int32_t*>(candidate_ids.data);
    auto* const scores        = static_cast<float*>(candidate_scores.data);
    auto* const done          = static_cast<std::int32_t*>(workspace.group_done.data);
    const auto* const partial = static_cast<const std::uint64_t*>(workspace.partial_keys.data);
    auto* const group         = static_cast<std::uint64_t*>(workspace.group_keys.data);
    if (workspace.secondary_groups != 0) {
        launch_merge<false, false>(partial, group, nullptr, nullptr, nullptr,
                                   workspace.producer_groups, workspace.merge_groups,
                                   workspace.columns, stream);
        launch_merge<true, true>(group, static_cast<std::uint64_t*>(workspace.secondary_keys.data),
                                 done, ids, scores, workspace.merge_groups,
                                 workspace.secondary_groups, workspace.columns, stream);
        return;
    }
    launch_merge<false, true>(partial, group, done, ids, scores, workspace.producer_groups,
                              workspace.merge_groups, workspace.columns, stream);
}

} // namespace ninfer::ops::detail
