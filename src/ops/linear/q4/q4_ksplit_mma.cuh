// Modified by satellitedown for Cinference: share staged activations across multiple 16-row weight tiles per CTA.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q4KSplitStoreEpilogue {};

struct Q4KSplitIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

template <int OutputRows, int InputRows>
struct Q4LinearGeometry {
    static constexpr int kOutputRows   = OutputRows;
    static constexpr int kInputRows    = InputRows;
    static constexpr int kGroupsPerRow = kInputRows / 64;
};

struct Q4KSplitMmaSchedule {
    static constexpr int kKWarps            = 8;
    static constexpr int kMinBlocksPerSm    = 6;
    static constexpr auto kCodeCache        = Cache::cg;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
};

__device__ __forceinline__ int q4_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q4KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q4_ksplit_bf16_pair(std::uint8_t packed) {
    const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
    const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
    Q4KSplitBf16PairBits result;
    result.pair = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return result.bits;
}

// A CTA owns RowTiles 16-row weight tiles and splits K across eight warps. The staged activation
// slice of each K group feeds every row tile of the CTA, so wide heads amortize the activation
// traffic over RowTiles weight tiles. Each output keeps the single-tile arithmetic: the same MMA
// sequence per 64-wide group, the same scale application and the same cross-warp reduction order.
template <class Geometry, int TileCols, int ActiveCols, class Epilogue = Q4KSplitStoreEpilogue,
          class RowPolicy = Q4KSplitIdentityRows, bool MaskedColumns = false, int RowTiles = 1>
__launch_bounds__(256, RowTiles == 1 ? 6 : 2) __global__
    void q4_ksplit_mma_kernel(const __nv_bfloat16* __restrict__ x,
                              const std::uint8_t* __restrict__ codes,
                              const std::uint8_t* __restrict__ scales,
                              __nv_bfloat16* __restrict__ out, Epilogue epilogue = {},
                              RowPolicy row_policy = {}, int columns = ActiveCols) {
    using Schedule              = Q4KSplitMmaSchedule;
    constexpr int kHidden       = Geometry::kInputRows;
    constexpr int kTileK        = Schedule::kTileKPerWarp;
    constexpr int kWarps        = Schedule::kKWarps;
    constexpr int kTileRows     = Schedule::kRowsPerCta;
    constexpr int kRows         = kTileRows * RowTiles;
    constexpr int kGroupK       = Schedule::kGroupK;
    constexpr int kGroups       = kHidden / kGroupK;
    constexpr int kCodeRowBytes = kHidden / 2;
    constexpr int kTileCols     = TileCols;
    constexpr int kNt           = kTileCols / 8;
    constexpr int kCodeChunks   = kGroupK / 32;
    static_assert(kTileCols >= 8 && kTileCols <= 32 && (kTileCols % 8) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols && ActiveCols > kTileCols - 8);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(RowPolicy::kOutputRowsPerCta <= kTileRows);
    static_assert(RowTiles >= 1);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRows][kGroupK / 2];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
            std::uint16_t scales[kRows][kWarps];
        } staging;

        float partial[kWarps * RowTiles * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int k_split      = warp;
    const int cta_row0     = static_cast<int>(blockIdx.x) * RowTiles * RowPolicy::kOutputRowsPerCta;
    const int live_columns = MaskedColumns ? columns : ActiveCols;
    const auto output_row0 = [&](int tile) {
        return cta_row0 + tile * RowPolicy::kOutputRowsPerCta;
    };

    const auto stage_x = [&](int group_k0) {
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &x_shared[warp][col * kTileK + q4_ksplit_swizzle_64(col, k8 * 8)];
            if constexpr (MaskedColumns) {
                const int source = col < live_columns ? col : 0;
                cp_async_zfill<16>(dst,
                                   &x[static_cast<std::int64_t>(source) * kHidden + group_k0 +
                                      warp * kTileK + k8 * 8],
                                   col < live_columns ? 16 : 0);
            } else {
                cp_async<16>(dst, &x[static_cast<std::int64_t>(col) * kHidden + group_k0 +
                                     warp * kTileK + k8 * 8]);
            }
        }
    };

    const auto stage_weight = [&](int group_k0) {
        for (int task = tid; task < kRows * kCodeChunks; task += kWarps * 32) {
            const int row        = task / kCodeChunks;
            const int chunk      = task - row * kCodeChunks;
            const int tile       = row / kTileRows;
            const int weight_row = row_policy.weight_row(output_row0(tile), row - tile * kTileRows);
            cp_async<16, Schedule::kCodeCache>(
                &code_shared[row][chunk * 16],
                codes + static_cast<std::int64_t>(weight_row) * kCodeRowBytes + group_k0 / 2 +
                    chunk * 16);
        }
        for (int row = tid; row < kRows; row += kWarps * 32) {
            const int tile       = row / kTileRows;
            const int weight_row = row_policy.weight_row(output_row0(tile), row - tile * kTileRows);
            cp_async<16>(&scale_shared[row][0],
                         scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                   group_k0 / 64) *
                                      2);
        }
    };

    const int b_rin             = lane & 7;
    const int b_koff            = ((lane >> 3) & 1) << 3;
    const int warp_koff         = k_split * kTileK;
    float acc[RowTiles][kNt][4] = {};

    stage_weight(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0                = group_index * kGroupK;
        float group_acc[RowTiles][kNt][4] = {};

#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned bf[kNt][2];
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const int br = nt * 8 + b_rin;
                ldmatrix_x2(bf[nt][0], bf[nt][1],
                            smem_addr(&x_shared[k_split][br * kTileK + q4_ksplit_swizzle_64(
                                                                           br, ks * 16 + b_koff)]));
            }
            const int byte_col = warp_koff / 2 + ks * 8 + lid;
#pragma unroll
            for (int tile = 0; tile < RowTiles; ++tile) {
                const int r0       = tile * kTileRows + gid;
                const unsigned af0 = q4_ksplit_bf16_pair(code_shared[r0][byte_col]);
                const unsigned af1 = q4_ksplit_bf16_pair(code_shared[r0 + 8][byte_col]);
                const unsigned af2 = q4_ksplit_bf16_pair(code_shared[r0][byte_col + 4]);
                const unsigned af3 = q4_ksplit_bf16_pair(code_shared[r0 + 8][byte_col + 4]);
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    mma_bf16(group_acc[tile][nt][0], group_acc[tile][nt][1], group_acc[tile][nt][2],
                             group_acc[tile][nt][3], af0, af1, af2, af3, bf[nt][0], bf[nt][1]);
                }
            }
        }

#pragma unroll
        for (int tile = 0; tile < RowTiles; ++tile) {
            const int r0          = tile * kTileRows + gid;
            const float top_scale = __half2float(__ushort_as_half(scale_shared[r0][k_split]));
            const float bot_scale = __half2float(__ushort_as_half(scale_shared[r0 + 8][k_split]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[tile][nt][0] = fmaf(group_acc[tile][nt][0], top_scale, acc[tile][nt][0]);
                acc[tile][nt][1] = fmaf(group_acc[tile][nt][1], top_scale, acc[tile][nt][1]);
                acc[tile][nt][2] = fmaf(group_acc[tile][nt][2], bot_scale, acc[tile][nt][2]);
                acc[tile][nt][3] = fmaf(group_acc[tile][nt][3], bot_scale, acc[tile][nt][3]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            stage_weight(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial      = shared.partial;
    const auto slot_of = [&](int split, int tile, int nt) {
        return partial + (((split * RowTiles + tile) * kNt + nt) * 32 + lane) * 4;
    };
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int tile = 0; tile < RowTiles; ++tile) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                store_vec(slot_of(k_split, tile, nt),
                          make_float4(acc[tile][nt][0], acc[tile][nt][1], acc[tile][nt][2],
                                      acc[tile][nt][3]));
            }
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int tile = 0; tile < RowTiles; ++tile) {
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                const float4 partner = load_vec<float4>(slot_of(k_split + 1, tile, nt));
                acc[tile][nt][0] += partner.x;
                acc[tile][nt][1] += partner.y;
                acc[tile][nt][2] += partner.z;
                acc[tile][nt][3] += partner.w;
                if (k_split != 0) {
                    store_vec(slot_of(k_split, tile, nt),
                              make_float4(acc[tile][nt][0], acc[tile][nt][1], acc[tile][nt][2],
                                          acc[tile][nt][3]));
                }
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
#pragma unroll
        for (int tile = 0; tile < RowTiles; ++tile) {
            const int row0 = output_row0(tile);
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                float4 sum = make_float4(acc[tile][nt][0], acc[tile][nt][1], acc[tile][nt][2],
                                         acc[tile][nt][3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value = load_vec<float4>(slot_of(split, tile, nt));
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int col0 = nt * 8 + 2 * lid;
                if constexpr (std::is_same_v<Epilogue, Q4KSplitStoreEpilogue>) {
                    if (col0 < live_columns) {
                        out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid] =
                            __float2bfloat16_rn(sum.x);
                        out[static_cast<std::int64_t>(col0) * Geometry::kOutputRows + row0 + gid +
                            8] = __float2bfloat16_rn(sum.z);
                    }
                    if (col0 + 1 < live_columns) {
                        out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 +
                            gid]     = __float2bfloat16_rn(sum.y);
                        out[static_cast<std::int64_t>(col0 + 1) * Geometry::kOutputRows + row0 +
                            gid + 8] = __float2bfloat16_rn(sum.w);
                    }
                } else {
                    epilogue.template store<ActiveCols>(row0 + gid, col0, sum);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
