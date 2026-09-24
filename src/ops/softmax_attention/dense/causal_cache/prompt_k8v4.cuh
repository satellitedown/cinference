// Modified by satellitedown for Cinference: pipelined score and PV warps with staged copies.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

// Asymmetric K8V4 causal prompt kernel. Q and cached K use the existing row-scaled E4M3
// rotation and native FP8 Tensor Core path. Rotated V uses group-16 NVFP4, widens exactly to
// FP16 for FP16/FP32 PV MMA, and the normalized result receives the FP32 inverse rotation.
//
// The CTA is two pipelined warp groups. Score warps run QK and the online softmax of key tile j
// into a double-buffered P tile; PV warps meanwhile run the PV product of tile j - 1 and widen
// tile j's V. Every row, key and value dimension sees the same operations in the same order as a
// tile-at-a-time schedule.

#include "ops/common/mbarrier.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptK8V4Warps    = 16;
inline constexpr int kCausalPromptK8V4Threads  = kCausalPromptK8V4Warps * 32;
inline constexpr int kCausalPromptK8V4Br       = 64;
inline constexpr int kCausalPromptK8V4Bc       = 64;
inline constexpr int kCausalPromptK8V4DB16     = kCausalPromptHeadDim / 2;
inline constexpr int kCausalPromptK8V4RowTiles = kCausalPromptK8V4Br / 16;
// Score warp w owns row tile w / 2 and the 32-key column half w % 2; PV warp 8 + p owns row tile
// p % 4 and the 128-dimension half p / 4.
inline constexpr int kCausalPromptK8V4ScoreWarps = 2 * kCausalPromptK8V4RowTiles;
inline constexpr int kCausalPromptK8V4PvWarps =
    kCausalPromptK8V4Warps - kCausalPromptK8V4ScoreWarps;
inline constexpr int kCausalPromptK8V4GroupThreads = kCausalPromptK8V4ScoreWarps * 32;

inline constexpr int kCausalPromptK8V4KBytes = kCausalPromptK8V4Bc * kCausalPromptHeadDim;
inline constexpr int kCausalPromptK8V4KScaleBytes =
    kCausalPromptK8V4Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptK8V4VBytes = kCausalPromptK8V4Bc * (kCausalPromptHeadDim / 2);
inline constexpr int kCausalPromptK8V4VScaleBytes = kCausalPromptK8V4Bc * kKVCacheNvfp4Groups;
inline constexpr int kCausalPromptK8V4VStageBytes =
    kCausalPromptK8V4Bc * kCausalPromptHeadDim * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptK8V4PBytes =
    kCausalPromptK8V4Br * kCausalPromptK8V4Bc * static_cast<int>(sizeof(__half));
inline constexpr int kCausalPromptK8V4RowBytes =
    kCausalPromptK8V4Br * static_cast<int>(sizeof(float));

// Double-buffered K tiles and P tiles, single packed-V and widened-V stages; row statistics last,
// behind the FP32 output staging the epilogue lays over the dead tiles.
inline constexpr int kCausalPromptK8V4KOffset      = 0;
inline constexpr int kCausalPromptK8V4KScaleOffset = 2 * kCausalPromptK8V4KBytes;
inline constexpr int kCausalPromptK8V4VOffset =
    kCausalPromptK8V4KScaleOffset + 2 * kCausalPromptK8V4KScaleBytes;
inline constexpr int kCausalPromptK8V4VScaleOffset =
    kCausalPromptK8V4VOffset + kCausalPromptK8V4VBytes;
inline constexpr int kCausalPromptK8V4VStageOffset =
    kCausalPromptK8V4VScaleOffset + kCausalPromptK8V4VScaleBytes;
inline constexpr int kCausalPromptK8V4POffset =
    kCausalPromptK8V4VStageOffset + kCausalPromptK8V4VStageBytes;
inline constexpr int kCausalPromptK8V4StatsOffset =
    kCausalPromptK8V4POffset + 2 * kCausalPromptK8V4PBytes;
// q scale, two alpha buffers, two partial maxima and sums, running maximum and sum.
inline constexpr int kCausalPromptK8V4StatsRows = 1 + 2 + 2 + 2 + 2;
inline constexpr int kCausalPromptK8V4BarrierOffset =
    kCausalPromptK8V4StatsOffset + kCausalPromptK8V4StatsRows * kCausalPromptK8V4RowBytes;
inline constexpr int kCausalPromptK8V4SmemBytes =
    kCausalPromptK8V4BarrierOffset + 4 * static_cast<int>(sizeof(std::uint64_t));

static_assert(kCausalPromptK8V4ScoreWarps == 8 && kCausalPromptK8V4PvWarps == 8);
static_assert(kCausalPromptK8V4Bc == kPagedKVPageSize);
// The quantized query tile lives in the widened-V stage until the pipeline starts.
static_assert(kCausalPromptK8V4Br * kCausalPromptHeadDim <= kCausalPromptK8V4VStageBytes);
// The epilogue's FP32 output staging must end before the row statistics.
static_assert(kCausalPromptK8V4Br * kCausalPromptHeadDim * static_cast<int>(sizeof(float)) <=
              kCausalPromptK8V4StatsOffset);
static_assert(kCausalPromptK8V4SmemBytes == 93728);

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(128) void causal_attention_prompt_k8v4_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const __half* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int Br            = kCausalPromptK8V4Br;
    constexpr int Bc            = kCausalPromptK8V4Bc;
    constexpr int DB16          = kCausalPromptK8V4DB16;
    constexpr int QKKs          = D / 32;
    constexpr int QKNt          = (Bc / 2) / 8;
    constexpr int PVNt          = D / (2 * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int GroupThreads  = kCausalPromptK8V4GroupThreads;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffU;
    static_assert(QKKs == 8 && QKNt == 4 && PVNt == 16);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    const auto k_fp8 = [&](int stage) {
        return smem_raw + kCausalPromptK8V4KOffset + stage * kCausalPromptK8V4KBytes;
    };
    const auto k_scale_s = [&](int stage) {
        return reinterpret_cast<__half*>(smem_raw + kCausalPromptK8V4KScaleOffset) + stage * Bc;
    };
    std::uint8_t* const v_nvfp4   = smem_raw + kCausalPromptK8V4VOffset;
    std::uint8_t* const v_scale_s = smem_raw + kCausalPromptK8V4VScaleOffset;
    __half* const v_f16       = reinterpret_cast<__half*>(smem_raw + kCausalPromptK8V4VStageOffset);
    std::uint8_t* const q_fp8 = smem_raw + kCausalPromptK8V4VStageOffset;
    const auto p_s            = [&](int buffer) {
        return reinterpret_cast<__half*>(smem_raw + kCausalPromptK8V4POffset) + buffer * Br * Bc;
    };
    float* const stats       = reinterpret_cast<float*>(smem_raw + kCausalPromptK8V4StatsOffset);
    float* const q_scale     = stats;
    float* const alpha_s     = stats + Br;
    float* const partial_m_s = alpha_s + 2 * Br;
    float* const partial_l_s = partial_m_s + 2 * Br;
    float* const running_m_s = partial_l_s + 2 * Br;
    float* const running_l_s = running_m_s + Br;
    // p_ready[b]: the score warps published a P tile and its rescale factors in buffer b;
    // p_free[b]: the PV warps finished reading them.
    auto* const barriers =
        reinterpret_cast<std::uint64_t*>(smem_raw + kCausalPromptK8V4BarrierOffset);
    std::uint64_t* const p_ready = barriers;
    std::uint64_t* const p_free  = barriers + 2;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) return;
    if (q0 >= tokens) {
        causal_prompt_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                 kCausalPromptK8V4Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int tile_rows             = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + tile_rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    const bool scores               = warp < kCausalPromptK8V4ScoreWarps;
    const int group_tid             = scores ? tid : tid - GroupThreads;

    // Score warps stage K tiles (codes and scales), PV warps V tiles. Tiles are page-aligned, so
    // one physical page id addresses a tile; a tile whose keys are all visible copies whole
    // page-head blocks, and only the causal tail tests keys and zero-fills.
    const auto issue_k_tile = [&](int kb) {
        const int tile_k0       = kb * Bc;
        const int physical_page = block_table[kb];
        std::uint8_t* const k   = k_fp8(kb & 1);
        __half* const k_scale   = k_scale_s(kb & 1);
        if (tile_k0 + Bc - 1 <= max_query_abs) {
            const std::uint8_t* block =
                cache_k + kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Bc * (D / 16) / GroupThreads; ++i) {
                const int chunk = group_tid + i * GroupThreads;
                const int key_l = chunk / (D / 16);
                const int dc    = chunk % (D / 16);
                cp_async<16, Cache::cg>(&k[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2],
                                        block + key_l * D + dc * 16);
            }
        } else {
#pragma unroll 1
            for (int chunk = group_tid; chunk < Bc * (D / 16); chunk += GroupThreads) {
                const int key_l  = chunk / (D / 16);
                const int dc     = chunk - key_l * (D / 16);
                std::uint8_t* kd = &k[(key_l * DB16 + causal_prompt_swz(key_l, dc * 8)) * 2];
                if (tile_k0 + key_l <= max_query_abs) {
                    cp_async<16, Cache::cg>(kd, &cache_k[kv_cache_fp8_code_index<Geometry>(
                                                    physical_page, kv_head, dc * 16, key_l)]);
                } else {
                    store_vec(kd, make_int4(0, 0, 0, 0));
                }
            }
        }
        // Eight 2-byte scales per 16-byte copy; keys past the causal limit read as zero.
        if (group_tid < Bc / 8) {
            const int key_l = group_tid * 8;
            const int live  = min(8, max(0, max_query_abs + 1 - (tile_k0 + key_l)));
            const std::int64_t offset =
                kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, key_l);
            cp_async_zfill<16>(k_scale + key_l, cache_k_scale + (live > 0 ? offset : 0),
                               live * static_cast<int>(sizeof(__half)));
        }
    };
    const auto issue_v_tile = [&](int kb) {
        const int tile_k0       = kb * Bc;
        const int physical_page = block_table[kb];
        if (tile_k0 + Bc - 1 <= max_query_abs) {
            const std::uint8_t* block =
                cache_v + kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Bc * (D / 32) / GroupThreads; ++i) {
                const int chunk = group_tid + i * GroupThreads;
                cp_async<16, Cache::cg>(&v_nvfp4[chunk * 16], block + chunk * 16);
            }
        } else {
#pragma unroll 1
            for (int chunk = group_tid; chunk < Bc * (D / 32); chunk += GroupThreads) {
                const int key_l  = chunk / (D / 32);
                const int dc     = chunk - key_l * (D / 32);
                std::uint8_t* vd = &v_nvfp4[chunk * 16];
                if (tile_k0 + key_l <= max_query_abs) {
                    cp_async<16, Cache::cg>(vd, &cache_v[kv_cache_nvfp4_code_index<Geometry>(
                                                    physical_page, kv_head, dc * 32, key_l)]);
                } else {
                    store_vec(vd, make_int4(0, 0, 0, 0));
                }
            }
        }
        if (group_tid < Bc) {
            std::uint8_t* destination = v_scale_s + group_tid * kKVCacheNvfp4Groups;
            if (tile_k0 + group_tid <= max_query_abs) {
                cp_async<16>(destination,
                             cache_v_scale + kv_cache_nvfp4_scale_index<Geometry>(
                                                 physical_page, kv_head, 0, group_tid));
            } else {
                store_vec(destination, make_int4(0, 0, 0, 0));
            }
        }
    };

    for (int row = warp; row < Br; row += kCausalPromptK8V4Warps) {
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                row < tile_rows
                    ? __bfloat162float(q[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)])
                    : 0.0F;
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) local_absmax = fmaxf(local_absmax, fabsf(values[r]));
        const float absmax = warp_max(local_absmax, FullMask);
        const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
        const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            causal_prompt_store_byte_swizzled(q_fp8, row, d,
                                              kv_cache_fp8_quant_code(values[r], inv));
        }
        if (lane == 0) q_scale[row] = qs;
    }
    if (tid < Br) {
        running_m_s[tid] = -CUDART_INF_F;
        running_l_s[tid] = 0.0F;
    }
    if (tid == 0) {
        for (int buffer = 0; buffer < 2; ++buffer) {
            cta_mbarrier_init(&p_ready[buffer], GroupThreads);
            cta_mbarrier_init(&p_free[buffer], GroupThreads);
        }
        cta_mbarrier_fence_init();
    }
    // Score warps keep K tiles j and j + 1 in flight, one commit group each; PV warps stage V.
    if (scores) {
        issue_k_tile(0);
        ninfer::ops::cp_commit();
        if (key_blocks > 1) issue_k_tile(1);
        ninfer::ops::cp_commit();
    } else {
        issue_v_tile(0);
        ninfer::ops::cp_commit();
    }
    __syncthreads();

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    // FP32 output staging for the inverse rotation, laid over the K, V and P stages once dead.
    float* const rotated_out = reinterpret_cast<float*>(smem_raw);

    if (scores) {
        const int row_base     = (warp >> 1) * 16;
        const int col_half     = warp & 1;
        const int col_base     = col_half * (Bc / 2);
        const int row0         = row_base + gid;
        const int row1         = row0 + 8;
        const int qabs0        = row0 < tile_rows ? base_pos + q0 + row0 : -1;
        const int qabs1        = row1 < tile_rows ? base_pos + q0 + row1 : -1;
        const float q_scale_r0 = __shfl_sync(FullMask, lid == 0 ? q_scale[row0] : 0.0F, gid * 4);
        const float q_scale_r1 = __shfl_sync(FullMask, lid == 0 ? q_scale[row1] : 0.0F, gid * 4);
        unsigned qf[QKKs][4];
        {
            const auto* q_b16 = reinterpret_cast<const __nv_bfloat16*>(q_fp8);
#pragma unroll
            for (int kk = 0; kk < QKKs; ++kk) {
                const int acol = kk * 16 + a_coloff;
                ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
        }
        // The query tile is in registers; the PV warps may widen V over it.
        __syncthreads();
        const float scale_l2 = scale * Log2E;

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0     = kb * Bc;
            const int buffer = kb & 1;
            ninfer::ops::cp_wait<1>();
            asm volatile("bar.sync 1, %0;" ::"n"(GroupThreads) : "memory");

            const auto* k_b16 = reinterpret_cast<const __nv_bfloat16*>(k_fp8(buffer));
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt)
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;
#pragma unroll
            for (int kk = 0; kk < QKKs; ++kk) {
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow = col_base + nt * 8 + b_rin;
                    const int bcol = kk * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], qf[kk][0],
                                 qf[kk][1], qf[kk][2], qf[kk][3], bf[0], bf[1]);
                }
            }
            const __half* k_scale = k_scale_s(buffer);
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int keya = col_base + nt * 8 + 2 * lid;
                const int keyb = keya + 1;
                float ks0      = gid == 0 ? __half2float(k_scale[keya]) : 0.0F;
                float ks1      = gid == 0 ? __half2float(k_scale[keyb]) : 0.0F;
                ks0            = __shfl_sync(FullMask, ks0, lid);
                ks1            = __shfl_sync(FullMask, ks1, lid);
                score[nt][0] *= q_scale_r0 * ks0;
                score[nt][1] *= q_scale_r0 * ks1;
                score[nt][2] *= q_scale_r1 * ks0;
                score[nt][3] *= q_scale_r1 * ks1;
            }

            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + col_base + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);
            if (lid == 0) {
                partial_m_s[col_half * Br + row0] = bm0;
                partial_m_s[col_half * Br + row1] = bm1;
            }
            asm volatile("bar.sync 1, %0;" ::"n"(GroupThreads) : "memory");
            // Every score warp has consumed this K stage: refill it with tile kb + 2.
            if (kb + 2 < key_blocks) issue_k_tile(kb + 2);
            ninfer::ops::cp_commit();

            bm0                     = fmaxf(partial_m_s[row0], partial_m_s[Br + row0]);
            bm1                     = fmaxf(partial_m_s[row1], partial_m_s[Br + row1]);
            const float previous_m0 = running_m_s[row0];
            const float previous_m1 = running_m_s[row1];
            const float nm0         = fmaxf(previous_m0, bm0);
            const float nm1         = fmaxf(previous_m1, bm1);
            const float nm0_scaled  = nm0 * scale_l2;
            const float nm1_scaled  = nm1 * scale_l2;
            const float alpha0      = previous_m0 == -CUDART_INF_F
                                          ? 0.0F
                                          : exp2_approx(__fmaf_rn(previous_m0, scale_l2, -nm0_scaled));
            const float alpha1      = previous_m1 == -CUDART_INF_F
                                          ? 0.0F
                                          : exp2_approx(__fmaf_rn(previous_m1, scale_l2, -nm1_scaled));
            // P buffer kb & 1 last held tile kb - 2.
            if (kb >= 2) cta_mbarrier_wait(&p_free[buffer], ((kb - 2) >> 1) & 1);
            __half* const p = p_s(buffer);
            float bl0       = 0.0F;
            float bl1       = 0.0F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = col_base + nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0F;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0F;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0F;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col0)] = __float2half_rn(p00);
                p[row0 * Bc + causal_prompt_p_swz<Bc>(row0, col1)] = __float2half_rn(p01);
                p[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col0)] = __float2half_rn(p10);
                p[row1 * Bc + causal_prompt_p_swz<Bc>(row1, col1)] = __float2half_rn(p11);
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);
            if (lid == 0) {
                partial_l_s[col_half * Br + row0] = bl0;
                partial_l_s[col_half * Br + row1] = bl1;
            }
            asm volatile("bar.sync 1, %0;" ::"n"(GroupThreads) : "memory");
            if (col_half == 0 && lid == 0) {
                const float tile_l0         = partial_l_s[row0] + partial_l_s[Br + row0];
                const float tile_l1         = partial_l_s[row1] + partial_l_s[Br + row1];
                running_l_s[row0]           = __fmaf_rn(running_l_s[row0], alpha0, tile_l0);
                running_l_s[row1]           = __fmaf_rn(running_l_s[row1], alpha1, tile_l1);
                running_m_s[row0]           = nm0;
                running_m_s[row1]           = nm1;
                alpha_s[buffer * Br + row0] = alpha0;
                alpha_s[buffer * Br + row1] = alpha1;
            }
            cta_mbarrier_arrive(&p_ready[buffer]);
        }
        // Pairs with the PV warps' barrier before they read the final row sums.
        __syncthreads();
    } else {
        const int pv_warp  = warp - kCausalPromptK8V4ScoreWarps;
        const int row_base = (pv_warp % kCausalPromptK8V4RowTiles) * 16;
        const int d_half   = pv_warp / kCausalPromptK8V4RowTiles;
        float acc[PVNt][4];
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
        }
        // Widens tile kb's V into the stage once every PV warp has finished reading it.
        const auto widen = [&](int kb) {
            const int k0 = kb * Bc;
            ninfer::ops::cp_wait<0>();
            asm volatile("bar.sync 2, %0;" ::"n"(GroupThreads) : "memory");
            // Every code word and scale is requested before the first conversion so the
            // shared-memory latencies overlap.
            constexpr int kChunks = Bc * (D / 8) / GroupThreads;
            std::uint32_t packed[kChunks];
            std::uint8_t scale_codes[kChunks];
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                const int chunk = group_tid + i * GroupThreads;
                const int key_l = chunk / (D / 8);
                const int d     = (chunk % (D / 8)) * 8;
                packed[i]       = load_vec<std::uint32_t>(&v_nvfp4[key_l * (D / 2) + d / 2]);
                scale_codes[i]  = v_scale_s[key_l * kKVCacheNvfp4Groups + d / kKVCacheNvfp4Group];
            }
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                const int chunk = group_tid + i * GroupThreads;
                const int key_l = chunk / (D / 8);
                const int d     = (chunk % (D / 8)) * 8;
                __half* dst     = &v_f16[key_l * D + causal_prompt_swz(key_l, d)];
                if (k0 + key_l <= max_query_abs) {
                    store_vec(dst, kv_cache_nvfp4_dequant_f16x8(
                                       reinterpret_cast<const std::uint8_t*>(&packed[i]),
                                       scale_codes[i]));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            // The packed tile is consumed: stage the next one behind the widened stage.
            asm volatile("bar.sync 2, %0;" ::"n"(GroupThreads) : "memory");
            if (kb + 1 < key_blocks) issue_v_tile(kb + 1);
            ninfer::ops::cp_commit();
        };
        const auto accumulate = [&](int kb) {
            const int buffer = kb & 1;
            cta_mbarrier_wait(&p_ready[buffer], (kb >> 1) & 1);
            const float alpha0 = alpha_s[buffer * Br + row_base + gid];
            const float alpha1 = alpha_s[buffer * Br + row_base + gid + 8];
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
            const __half* p = p_s(buffer);
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                            smem_addr(&p[(row_base + a_rowoff) * Bc +
                                         causal_prompt_p_swz<Bc>(row_base + a_rowoff, pcol)]));
#pragma unroll
                for (int n = 0; n < PVNt; n += 2) {
                    // Lanes 16-31 address the second column tile, so one transposed x4 load
                    // carries the B fragments of column tiles n and n + 1.
                    unsigned vf[4];
                    const int vrow = k * 16 + b_koff + b_rin;
                    const int vcol = (d_half * PVNt + n + (lane >> 4)) * 8;
                    ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                                  smem_addr(&v_f16[vrow * D + causal_prompt_swz(vrow, vcol)]));
                    mma_f16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                            vf[0], vf[1]);
                    mma_f16(acc[n + 1][0], acc[n + 1][1], acc[n + 1][2], acc[n + 1][3], pf[0],
                            pf[1], pf[2], pf[3], vf[2], vf[3]);
                }
            }
            cta_mbarrier_arrive(&p_free[buffer]);
        };

        // The score warps load the query tile out of the V stage before it is widened into.
        __syncthreads();
        widen(0);
        for (int kb = 1; kb < key_blocks; ++kb) {
            accumulate(kb - 1);
            widen(kb);
        }
        accumulate(key_blocks - 1);

        const int row0 = row_base + gid;
        const int row1 = row0 + 8;
        // Every score and PV warp is done: the final row sums are published and the K, V and P
        // stages under the output staging are dead.
        __syncthreads();
        const float inv_l0 = running_l_s[row0] > 0.0F ? __frcp_rn(running_l_s[row0]) : 0.0F;
        const float inv_l1 = running_l_s[row1] > 0.0F ? __frcp_rn(running_l_s[row1]) : 0.0F;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0 = (d_half * PVNt + n) * 8 + 2 * lid;
            if (row0 < tile_rows) {
                *reinterpret_cast<float2*>(&rotated_out[row0 * D + d0]) =
                    make_float2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
            }
            if (row1 < tile_rows) {
                *reinterpret_cast<float2*>(&rotated_out[row1 * D + d0]) =
                    make_float2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
            }
        }
    }
    // The FP32 output staging is complete.
    __syncthreads();

    // R is symmetric, but this application is the inverse/transpose semantic boundary.
    for (int row = warp; row < tile_rows; row += kCausalPromptK8V4Warps) {
        float values[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) values[r] = rotated_out[row * D + lane + 32 * r];
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d                                               = lane + 32 * r;
            out[causal_prompt_q_index<Geometry>(q_head, d, q0 + row)] = __float2bfloat16(values[r]);
        }
    }
    __syncthreads();

    causal_prompt_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                             kCausalPromptK8V4Threads);
}

} // namespace ninfer::ops
