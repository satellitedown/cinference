// Modified by satellitedown for Cinference: staged lattice walk; verify trees with lookup chains.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kCandidates = 16, kRank = 256, kMaxSteps = 15;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                            a.successor + static_cast<std::int64_t>(token) * kRank + lane * 8);
    cp_commit();
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank)
        shared.product[tid] =
            __bfloat162float(
                a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]) *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    cp_wait<0>();
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

// Every step's edge block is staged in shared memory first, so each step's rank-dependent lookup
// reads shared memory instead of waiting on a dependent global load.
__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    __shared__ __align__(16) float staged[kMaxSteps * kCandidates * kCandidates];
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto* block =
        edges + static_cast<std::int64_t>(batch) * a.steps * kCandidates * kCandidates;
    for (int i = lane; i < a.steps * kCandidates * kCandidates / 4; i += 32) {
        cp_async<16>(&staged[i * 4], block + i * 4);
    }
    cp_commit();
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
    cp_wait<0>();
    __syncwarp();
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? staged[(step * kCandidates + predecessor_rank) * kCandidates + lane]
                : -CUDART_INF_F;
        int selected =
            draw_rank(edge, temperature, seed, position + step, a.q + column * kCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

// Verify trees grown best-first over the lattice. A node is one candidate at one step reached from
// its parent node; its score is the sum of the lattice's log-softmax transition probabilities
// (edges / kTreeTemperature) along its root path, and every node's score is at most its parent's,
// so taking the best frontier entry N times yields the N most likely root paths as a tree.
//
// An optional lookup chain (tokens proposed for steps 0.. from an earlier occurrence of the
// context's suffix) scores depth * lookup log-probability along its own root path. Where its token
// is the lattice candidate under a lattice parent on the chain, that candidate takes the better of
// both scores; elsewhere the chain continues as lookup-only nodes, whose only child is the next
// chain token. Both scores fall with depth, so best-first still yields a tree.
constexpr float kTreeInverseTemperature = 1.0F / 1.5F;
constexpr int kTreeFrontier             = (kCandidates + 1) * (kMaxSteps + 1);
constexpr int kTreeNodes                = kMaxSteps + 1;
// Frontier/node info packs flags | parent node << 8 | step << 4 | rank.
constexpr int kTreeLookupOnly = 1 << 16;
constexpr int kTreeOnLookup   = 1 << 17;

__global__ __launch_bounds__(32) void selector_tree_kernel(
    DeviceArgs a, const float* edges, const std::int32_t* extents, std::int32_t* tree_parents,
    std::int32_t* tree_masks, std::int32_t* rope_positions, float inverse_temperature,
    const std::int32_t* lookup_tokens, const std::int32_t* lookup_counts,
    const float* lookup_log_probability) {
    __shared__ __align__(16) float staged[kMaxSteps * kCandidates * kCandidates];
    __shared__ float frontier_score[kTreeFrontier];
    __shared__ std::int32_t frontier_info[kTreeFrontier];
    __shared__ std::int32_t node_info[kTreeNodes];
    __shared__ std::int32_t lookup_rank[kMaxSteps];
    const int lane = threadIdx.x, batch = blockIdx.x, steps = a.steps, width = steps + 1;
    const auto* block =
        edges + static_cast<std::int64_t>(batch) * steps * kCandidates * kCandidates;
    for (int i = lane; i < steps * kCandidates * kCandidates / 4; i += 32) {
        cp_async<16>(&staged[i * 4], block + i * 4);
    }
    cp_commit();
    int extent                 = extents[batch];
    extent                     = extent < 0 ? 0 : (extent > steps ? steps : extent);
    int lookup_count           = 0;
    float lookup_log_p         = 0.0F;
    const std::int32_t* lookup = nullptr;
    if (lookup_tokens != nullptr) {
        lookup_count = lookup_counts[batch];
        lookup_count = lookup_count < 0 ? 0 : (lookup_count > steps ? steps : lookup_count);
        lookup_log_p = lookup_log_probability[batch];
        lookup       = lookup_tokens + static_cast<std::int64_t>(batch) * steps;
    }
    // The lattice rank of each chain token at its step, -1 when the drafter did not propose it.
    if (lane < lookup_count) {
        const std::int32_t token = lookup[lane];
        const std::int32_t* ids =
            a.ids + (static_cast<std::int64_t>(batch) * steps + lane) * kCandidates;
        int rank = -1;
        for (int r = kCandidates - 1; r >= 0; --r) rank = ids[r] == token ? r : rank;
        lookup_rank[lane] = rank;
    }
    cp_wait<0>();
    __syncwarp();

    // Every lattice row's transition log-probabilities, in place: one row per lane at a time.
    for (int row = lane; row < steps * kCandidates; row += 32) {
        float* values = &staged[row * kCandidates];
        float e[kCandidates];
#pragma unroll
        for (int c = 0; c < kCandidates; c += 4) {
            const float4 v = *reinterpret_cast<const float4*>(values + c);
            e[c]           = v.x * inverse_temperature;
            e[c + 1]       = v.y * inverse_temperature;
            e[c + 2]       = v.z * inverse_temperature;
            e[c + 3]       = v.w * inverse_temperature;
        }
        float m = e[0];
#pragma unroll
        for (int c = 1; c < kCandidates; ++c) m = fmaxf(m, e[c]);
        float z = 0.0F;
#pragma unroll
        for (int c = 0; c < kCandidates; ++c) z += __expf(e[c] - m);
        const float shift = m + __logf(z);
#pragma unroll
        for (int c = 0; c < kCandidates; c += 4) {
            *reinterpret_cast<float4*>(values + c) =
                make_float4(e[c] - shift, e[c + 1] - shift, e[c + 2] - shift, e[c + 3] - shift);
        }
    }
    __syncwarp();

    // Best-first growth.
    int count = 0;
    // Chain-only child at `step` (depth step + 1) of `parent`.
    const auto push_lookup = [&](int parent, int step) {
        if (lane == 0) {
            frontier_score[count] = static_cast<float>(step + 1) * lookup_log_p;
            frontier_info[count]  = kTreeLookupOnly | kTreeOnLookup | parent << 8 | step << 4;
        }
        count += 1;
    };
    // The lattice children at `step` of a lattice node of rank `rank` (the anchor row for the
    // root), plus the chain's child when the parent is on the chain.
    const auto push = [&](int parent, int step, int rank, float parent_score, bool on_lookup) {
        const bool chain_child = on_lookup && step < lookup_count;
        const int chain_rank   = chain_child ? lookup_rank[step] : -1;
        if (lane < kCandidates) {
            float score = parent_score + staged[(step * kCandidates + rank) * kCandidates + lane];
            int flags   = 0;
            if (lane == chain_rank) {
                score = fmaxf(score, static_cast<float>(step + 1) * lookup_log_p);
                flags = kTreeOnLookup;
            }
            frontier_score[count + lane] = score;
            frontier_info[count + lane]  = flags | parent << 8 | step << 4 | lane;
        }
        count += kCandidates;
        if (chain_child && chain_rank < 0) push_lookup(parent, step);
        __syncwarp();
    };
    if (extent > 0) push(0, 0, 0, 0.0F, true);
#pragma unroll 1
    for (int node = 1; node <= extent; ++node) {
        float best_score = -CUDART_INF_F;
        int best         = count;
        for (int i = lane; i < count; i += 32) {
            const float score = frontier_score[i];
            if (score > best_score) {
                best_score = score;
                best       = i;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_xor_sync(kFullWarpMask, best_score, offset);
            const int other         = __shfl_xor_sync(kFullWarpMask, best, offset);
            if (other_score > best_score || (other_score == best_score && other < best)) {
                best_score = other_score;
                best       = other;
            }
        }
        const int info = frontier_info[best];
        __syncwarp();
        if (lane == 0) {
            frontier_score[best] = -CUDART_INF_F;
            node_info[node]      = info;
        }
        __syncwarp();
        const int step = (info >> 4) & 15, rank = info & 15;
        if (step + 1 < extent) {
            if ((info & kTreeLookupOnly) != 0) {
                if (step + 1 < lookup_count) push_lookup(node, step + 1);
                __syncwarp();
            } else {
                push(node, step + 1, rank, best_score, (info & kTreeOnLookup) != 0);
            }
        }
    }

    // DFS pre-order with each node's children by increasing subtree size (so its largest subtree
    // is visited last), ties by selection order; nodes were selected parent first. One lane per
    // node: ancestors, subtree size, offset past the earlier siblings' subtrees, then the column
    // as the sum of the offsets on the root path.
    const bool live  = lane <= extent;
    const int info   = live && lane > 0 ? node_info[lane] : 0;
    const int parent = live ? (lane > 0 ? (info >> 8) & 0xff : -1) : -2;
    int ancestors    = 0;
    if (live && lane > 0) {
        for (int up = parent;; up = (node_info[up] >> 8) & 0xff) {
            ancestors |= 1 << up;
            if (up == 0) break;
        }
    }
    int size = 1;
#pragma unroll
    for (int m = 1; m < kTreeNodes; ++m) {
        const int other = __shfl_sync(kFullWarpMask, ancestors, m);
        size += m <= extent && ((other >> lane) & 1) != 0 ? 1 : 0;
    }
    int offset = 1;
#pragma unroll
    for (int m = 1; m < kTreeNodes; ++m) {
        const int other_parent = __shfl_sync(kFullWarpMask, parent, m);
        const int other_size   = __shfl_sync(kFullWarpMask, size, m);
        const bool earlier     = m <= extent && m != lane && other_parent == parent &&
                                 (other_size < size || (other_size == size && m < lane));
        offset += earlier ? other_size : 0;
    }
    int column = 0;
    int node   = live ? lane : 0;
#pragma unroll
    for (int level = 0; level < kTreeNodes - 1; ++level) {
        const int node_offset = __shfl_sync(kFullWarpMask, offset, node);
        const int node_parent = __shfl_sync(kFullWarpMask, parent, node);
        if (node != 0) {
            column += node_offset;
            node = node_parent;
        }
    }
    int mask = 1 << column;
#pragma unroll
    for (int m = 0; m < kTreeNodes; ++m) {
        const int other_column = __shfl_sync(kFullWarpMask, column, m);
        mask |= m <= extent && ((ancestors >> m) & 1) != 0 ? 1 << other_column : 0;
    }

    const int parent_column = __shfl_sync(kFullWarpMask, column, parent < 0 ? 0 : parent);

    const std::int64_t row = static_cast<std::int64_t>(batch) * width;
    const int root_rope    = rope_positions[row];
    if (live) {
        if (lane == 0) {
            tree_parents[row] = -1;
            tree_masks[row]   = 1;
        } else {
            const int step = (info >> 4) & 15, rank = info & 15;
            tree_parents[row + column]   = parent_column;
            tree_masks[row + column]     = mask;
            rope_positions[row + column] = root_rope + step + 1;
            a.drafts[batch * steps + column - 1] =
                (info & kTreeLookupOnly) != 0
                    ? lookup[step]
                    : a.ids[(static_cast<std::int64_t>(batch) * steps + step) * kCandidates + rank];
        }
    } else if (lane < width) {
        tree_parents[row + lane] = lane - 1;
        tree_masks[row + lane]   = 1 << lane;
    }
}

} // namespace

void candidate_selector_tree_launch(
    const Tensor& candidate_ids, const Tensor& unary_scores, const Tensor& projected_hidden,
    const Tensor& anchors, const Tensor& predecessor_codebook, const Tensor& successor_codebook,
    const Tensor& current_extents, const Tensor& lookup_tokens, const Tensor& lookup_counts,
    const Tensor& lookup_log_probability, Tensor& drafts, Tensor& tree_parents, Tensor& tree_masks,
    Tensor& rope_positions, const Tensor& edges, cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          nullptr,
                          nullptr,
                          static_cast<std::int32_t*>(drafts.data),
                          nullptr,
                          candidate_ids.ne[1]};
    auto* edge_data = static_cast<float*>(edges.data);
    selector_lattice_kernel<<<dim3(args.steps * candidate_ids.ne[2], kCandidates), 512, 0,
                              stream>>>(args, edge_data);
    CUDA_CHECK(cudaGetLastError());
    selector_tree_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(
        args, edge_data, static_cast<const std::int32_t*>(current_extents.data),
        static_cast<std::int32_t*>(tree_parents.data), static_cast<std::int32_t*>(tree_masks.data),
        static_cast<std::int32_t*>(rope_positions.data), kTreeInverseTemperature,
        static_cast<const std::int32_t*>(lookup_tokens.data),
        static_cast<const std::int32_t*>(lookup_counts.data),
        static_cast<const float*>(lookup_log_probability.data));
    CUDA_CHECK(cudaGetLastError());
}

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          candidate_ids.ne[1]};
    if (route == SelectorRoute::Direct) {
        selector_walk_kernel<<<candidate_ids.ne[2], 512, 0, stream>>>(args);
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        selector_lattice_kernel<<<dim3(args.steps * candidate_ids.ne[2], kCandidates), 512, 0,
                                  stream>>>(args, edges);
        CUDA_CHECK(cudaGetLastError());
        selector_lattice_walk_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(args, edges);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
