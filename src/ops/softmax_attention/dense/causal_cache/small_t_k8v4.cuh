// Modified by satellitedown for Cinference: wide kernel, split PV tails, verify-tree masks.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

// Asymmetric K8V4 split-KV causal attention for up to 96 query rows. A CTA owns one KV head and all
// GQA query heads, so each persistent K/V byte is streamed once. Q/K use the existing rotated,
// row-scaled E4M3 native Tensor Core path. Rotated V uses group-16 packed NVFP4 and widens exactly
// to FP16 at half scale for the PV MMA, which accumulates each 32-key product in FP16 before adding
// it to the FP32 accumulator. Split numerators and inverse rotation remain FP32.
//
// Narrow query blocks (up to three 16-row tiles) use the tiled kernel: producer warps score a key
// tile and publish P through shared memory to all warps for PV. Wide blocks (a complete 27B verify
// width) use the warp-specialized kernel: each compute warp owns one 16-row tile for the whole
// split and keeps its scores in registers as the PV operand, while two loader warps stream and
// widen the K/V tiles, so the Tensor Core work of one warp overlaps the softmax of another. Its
// query tile is rotated and quantized once per block by a preceding kernel and copied by every
// split, instead of being derived again in each split.

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// ----------------------------------------------------------------------------------------------
// Pieces shared by the tiled and warp-specialized kernels.

template <typename Geometry, int TokenTile, int Threads>
__device__ __forceinline__ void k8v4_write_neutral(int tid, int kv_head, int split,
                                                   float* partial_acc, float* partial_m,
                                                   float* partial_l) {
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int D        = kCausalHeadDim;
    for (int row = tid; row < RowCount; row += Threads) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                -CUDART_INF_F;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0F;
        }
    }
    for (int index = tid; index < RowCount * D; index += Threads) {
        const int row = index / D;
        const int d   = index - row * D;
        int q_head    = 0;
        int token     = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        if (causal_valid_q_head<Geometry>(kv_head, q_head)) {
            partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                0.0F;
        }
    }
}

// Keys [split_start, split_end) of one split, walked in Bc-key tiles from first_tile.
struct K8V4SplitRange {
    int split_start;
    int split_end;
    int first_tile;
    int key_blocks;
};

template <int TokenTile, int Bc>
__device__ __forceinline__ K8V4SplitRange k8v4_split_range(int window, int split,
                                                           int active_split_count) {
    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    int split_start         = 0;
    int split_end           = 0;
    if constexpr (TokenTile == 1 && Bc == 32) {
        const int first_owned_tile = split * logical_tiles / active_split_count;
        const int end_owned_tile   = (split + 1) * logical_tiles / active_split_count;
        split_start                = first_owned_tile * Bc;
        split_end                  = min(end_owned_tile * Bc, window);
    } else {
        const int units_per_split = tile_split ? div_up(logical_tiles, active_split_count)
                                               : div_up(window, active_split_count);
        split_start               = split * units_per_split * (tile_split ? Bc : 1);
        split_end = min(split_start + units_per_split * (tile_split ? Bc : 1), window);
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = split_start < split_end ? div_up(split_end - first_tile, Bc) : 0;
    return {split_start, split_end, first_tile, key_blocks};
}

// Appends this split's share of the new columns to the cache. One warp owns the complete D256 K
// row and then the complete V row; these are the same FP8-K and group-16 NVFP4-V operations as
// standalone K8V4 append. `page_of(position)` returns the physical page holding `position`.
template <typename Geometry, int Warps, typename CacheInput, typename PageOf>
__device__ __forceinline__ void
k8v4_append_split_columns(const CacheInput& input, const std::int32_t* positions, int valid_tokens,
                          const K8V4SplitRange& range, PageOf page_of, int kv_head,
                          std::uint8_t* cache_k, std::uint8_t* cache_v, __half* cache_k_scale,
                          std::uint8_t* cache_v_scale, float* scratch, int warp, int lane) {
    constexpr int D             = kCausalHeadDim;
    constexpr unsigned FullMask = 0xffffffffU;
    for (int token = warp; token < valid_tokens; token += Warps) {
        const int position = positions[token];
        if (position < range.split_start || position >= range.split_end) continue;
        const int physical_page = page_of(position);
        const int page_offset   = position & kPagedKVPageMask;
        float values[8];
        float local_absmax = 0.0F;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                __bfloat162float(input.k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
        const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_offset)] =
                kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
        }
        if (lane == 0) {
            cache_k_scale[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_offset)] =
                k_quant.scale;
        }

#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            values[r] =
                __bfloat162float(input.v[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
        }
        normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) scratch[warp * D + lane + 32 * r] = values[r];
        __syncwarp();
        if (lane < kKVCacheNvfp4Groups) {
            const auto quantized =
                kv_cache_nvfp4_quantize_group16(scratch + warp * D + lane * kKVCacheNvfp4Group);
            const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
            store_vec(cache_v + code_offset, make_uint2(quantized.codes_lo, quantized.codes_hi));
            cache_v_scale[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane,
                                                               page_offset)] = quantized.scale;
        }
        __syncwarp();
    }
}

// Rotates one query row, lane l holding dimension l + 32r in values[r], and row-quantizes it to
// E4M3: codes[r] encodes dimension l + 32r and the return value is the row scale. The whole warp
// must call it.
__device__ __forceinline__ float k8v4_quantize_query_row(float (&values)[8], int lane,
                                                         std::uint8_t (&codes)[8]) {
    constexpr unsigned FullMask = 0xffffffffU;
    normalized_hadamard_d256_inplace(values, lane);
    float local_absmax = 0.0F;
#pragma unroll
    for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
    const float absmax = warp_max(local_absmax, FullMask);
    const float qs     = absmax > 0.0F ? absmax / kKVCacheFp8MaxFinite : 0.0F;
    const float inv    = qs > 0.0F ? 1.0F / qs : 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) codes[r] = kv_cache_fp8_quant_code(values[r], inv);
    return qs;
}

__device__ __forceinline__ void k8v4_load_query_row(const __nv_bfloat16* row, int lane,
                                                    float (&values)[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = __bfloat162float(row[lane + 32 * r]);
}

// Rotates and row-quantizes the query rows of this KV head into the swizzled E4M3 tile q_s
// (Br x D bytes, rows past RowCount zero) and their scales into q_scale. Every thread of the CTA
// must call it; it returns after a CTA barrier.
template <typename Geometry, int TokenTile, int Threads>
__device__ __forceinline__ void k8v4_quantize_query(const __nv_bfloat16* q, int kv_head,
                                                    std::uint8_t* q_s, float* q_scale, int tid) {
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int Br       = ((RowCount + 15) / 16) * 16;
    constexpr int D        = kCausalHeadDim;
    constexpr int Warps    = Threads / 32;
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    for (int index = tid; index < Br * D; index += Threads) q_s[index] = 0;
    for (int row = tid; row < Br; row += Threads) q_scale[row] = 0.0F;
    __syncthreads();

    for (int row = warp; row < RowCount; row += Warps) {
        int q_head = 0;
        int token  = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float values[8];
        k8v4_load_query_row(q + causal_q_index<Geometry>(q_head, 0, token), lane, values);
        std::uint8_t codes[8];
        const float qs = k8v4_quantize_query_row(values, lane, codes);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            causal_small_t_store_byte_swizzled(q_s, row, lane + 32 * r, D / 2, codes[r]);
        }
        if (lane == 0) q_scale[row] = qs;
    }
    __syncthreads();
}

// The wide kernel's query tile, prepared once per verify block instead of by every split CTA:
// codes[((batch * KVHeads + kv_head) * RowCount + row) * D + d] and scales at the same row index,
// with the kernel's row order (row = token * GroupSize + local query head).
template <typename Geometry>
__device__ __forceinline__ std::int64_t k8v4_prepared_row(int batch, int kv_head, int row,
                                                          int row_count) {
    return (static_cast<std::int64_t>(batch) * Geometry::KVHeads + kv_head) * row_count + row;
}

// One warp per prepared row: the rows k8v4_quantize_query would derive in every CTA of the block.
template <typename Geometry, int TokenTile, bool MultiBatch>
__launch_bounds__(128) __global__
    void causal_attention_small_t_k8v4_prepare_query_kernel(const __nv_bfloat16* q,
                                                            std::int32_t full_width,
                                                            std::int32_t column_begin,
                                                            std::int32_t batch_size,
                                                            std::uint8_t* codes, float* scales) {
    constexpr int RowCount = TokenTile * Geometry::GroupSize;
    constexpr int D        = kCausalHeadDim;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int flat_row     = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x) / 32;
    const int row          = flat_row % RowCount;
    const int kv_head      = flat_row / RowCount % Geometry::KVHeads;
    const int batch        = flat_row / (RowCount * Geometry::KVHeads);
    if (batch >= batch_size) return;
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    int q_head = 0;
    int token  = 0;
    causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
    float values[8];
    k8v4_load_query_row(q + static_cast<std::int64_t>(D) * Geometry::QHeads * column_base +
                            causal_q_index<Geometry>(q_head, 0, token),
                        lane, values);
    std::uint8_t row_codes[8];
    const float qs          = k8v4_quantize_query_row(values, lane, row_codes);
    const std::int64_t base = k8v4_prepared_row<Geometry>(batch, kv_head, row, RowCount);
#pragma unroll
    for (int r = 0; r < 8; ++r) codes[base * D + lane + 32 * r] = row_codes[r];
    if (lane == 0) scales[base] = qs;
}

// Copies this KV head's prepared query tile into q_s/q_scale with k8v4_quantize_query's layout
// (rows past RowCount zero) and commits it as one cp.async group. Every thread must call it.
template <typename Geometry, int TokenTile, int Threads>
__device__ __forceinline__ void
k8v4_copy_prepared_query(const std::uint8_t* codes, const float* scales, int batch, int kv_head,
                         std::uint8_t* q_s, float* q_scale, int tid) {
    constexpr int RowCount   = TokenTile * Geometry::GroupSize;
    constexpr int Br         = ((RowCount + 15) / 16) * 16;
    constexpr int D          = kCausalHeadDim;
    constexpr int Chunks     = RowCount * (D / 16);
    const std::int64_t first = k8v4_prepared_row<Geometry>(batch, kv_head, 0, RowCount);
    for (int chunk = tid; chunk < Chunks; chunk += Threads) {
        const int row = chunk / (D / 16);
        const int c   = chunk - row * (D / 16);
        cp_async<16, Cache::cg>(q_s + row * D + ((c ^ (row & 7)) << 4),
                                codes + (first + row) * D + c * 16);
    }
    for (int row = tid; row < RowCount; row += Threads) {
        cp_async<4>(&q_scale[row], scales + first + row);
    }
    for (int index = RowCount * D + tid; index < Br * D; index += Threads) q_s[index] = 0;
    for (int row = RowCount + tid; row < Br; row += Threads) q_scale[row] = 0.0F;
    cp_commit();
}

// Issues the copies of one packed Bc-key tile: swizzled E4M3 K codes, packed NVFP4 V codes and
// both scale planes. Keys outside [split_start, split_end) are zero filled, so their codes and
// scales widen and score to zero. The caller commits.
//
// A tile never crosses a page (Bc divides the page), and within a page and head every plane is
// key-major, so a tile wholly inside the split is four contiguous spans copied with fixed
// per-thread work. Split-boundary tiles take the per-key path; there, scales for eight consecutive
// in-split keys are one aligned 16-byte span copied asynchronously and the remaining keys are
// filled per key.
template <typename Geometry, int Bc, int Threads>
__device__ __forceinline__ void
k8v4_issue_tile(int thread, std::uint8_t* k_codes, std::uint8_t* v_codes, __half* k_scale,
                std::uint8_t* v_scale, int tile_k0, int physical_page, const K8V4SplitRange& range,
                int kv_head, const std::uint8_t* cache_k, const std::uint8_t* cache_v,
                const __half* cache_k_scale, const std::uint8_t* cache_v_scale) {
    constexpr int D       = kCausalHeadDim;
    constexpr int DB16    = D / 2;
    constexpr int KChunks = Bc * (D / 16);
    constexpr int VChunks = Bc * (D / 32);
    static_assert(kPagedKVPageSize % Bc == 0);
    const auto k_destination = [&](int chunk) {
        const int key_l = chunk / (D / 16);
        const int dc    = chunk - key_l * (D / 16);
        return &k_codes[(key_l * DB16 + causal_small_t_tc_swz(key_l, dc * 8)) * 2];
    };

    if (tile_k0 >= range.split_start && tile_k0 + Bc <= range.split_end) {
        const int offset0 = tile_k0 & kPagedKVPageMask;
        const std::uint8_t* k_source =
            cache_k + kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, 0, offset0);
        const std::uint8_t* v_source =
            cache_v + kv_cache_nvfp4_code_index<Geometry>(physical_page, kv_head, 0, offset0);
#pragma unroll
        for (int i = 0; i < (KChunks + Threads - 1) / Threads; ++i) {
            const int chunk = thread + i * Threads;
            if (KChunks % Threads == 0 || chunk < KChunks) {
                cp_async<16, Cache::cg>(k_destination(chunk), k_source + chunk * 16);
            }
        }
#pragma unroll
        for (int i = 0; i < (VChunks + Threads - 1) / Threads; ++i) {
            const int chunk = thread + i * Threads;
            if (VChunks % Threads == 0 || chunk < VChunks) {
                cp_async<16, Cache::cg>(v_codes + chunk * 16, v_source + chunk * 16);
            }
        }
        for (int span = thread; span < Bc / 8; span += Threads) {
            cp_async<16>(&k_scale[span * 8],
                         cache_k_scale +
                             kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, offset0) +
                             span * 8);
        }
        for (int key_l = thread; key_l < Bc; key_l += Threads) {
            cp_async<16>(&v_scale[key_l * kKVCacheNvfp4Groups],
                         cache_v_scale + kv_cache_nvfp4_scale_index<Geometry>(
                                             physical_page, kv_head, 0, offset0 + key_l));
        }
        return;
    }

    for (int group = thread; group < Bc / 8; group += Threads) {
        const int key0 = tile_k0 + group * 8;
        if (key0 >= range.split_start && key0 + 8 <= range.split_end) {
            const std::int64_t k_scale_offset =
                kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, key0 & kPagedKVPageMask);
            cp_async<16>(&k_scale[group * 8], cache_k_scale + k_scale_offset);
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key          = key0 + i;
                k_scale[group * 8 + i] = key >= range.split_start && key < range.split_end
                                             ? cache_k_scale[kv_cache_fp8_scale_index<Geometry>(
                                                   physical_page, kv_head, key & kPagedKVPageMask)]
                                             : __float2half_rn(0.0F);
            }
        }
    }
    for (int key_l = thread; key_l < Bc; key_l += Threads) {
        const int key             = tile_k0 + key_l;
        std::uint8_t* v_scale_dst = &v_scale[key_l * kKVCacheNvfp4Groups];
        if (key >= range.split_start && key < range.split_end) {
            const std::int64_t v_scale_offset = kv_cache_nvfp4_scale_index<Geometry>(
                physical_page, kv_head, 0, key & kPagedKVPageMask);
            cp_async<16>(v_scale_dst, cache_v_scale + v_scale_offset);
        } else {
            store_vec(v_scale_dst, make_int4(0, 0, 0, 0));
        }
    }
#pragma unroll 1
    for (int chunk = thread; chunk < KChunks; chunk += Threads) {
        const int key_l = chunk / (D / 16);
        const int d     = (chunk - key_l * (D / 16)) * 16;
        const int key   = tile_k0 + key_l;
        if (key >= range.split_start && key < range.split_end) {
            const std::int64_t code_offset = kv_cache_fp8_code_index<Geometry>(
                physical_page, kv_head, d, key & kPagedKVPageMask);
            cp_async<16, Cache::cg>(k_destination(chunk), &cache_k[code_offset]);
        } else {
            store_vec(k_destination(chunk), make_int4(0, 0, 0, 0));
        }
    }
#pragma unroll 1
    for (int chunk = thread; chunk < VChunks; chunk += Threads) {
        const int key_l     = chunk / (D / 32);
        const int d         = (chunk - key_l * (D / 32)) * 32;
        const int key       = tile_k0 + key_l;
        std::uint8_t* v_dst = &v_codes[key_l * (D / 2) + d / 2];
        if (key >= range.split_start && key < range.split_end) {
            const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
                physical_page, kv_head, d, key & kPagedKVPageMask);
            cp_async<16, Cache::cg>(v_dst, &cache_v[code_offset]);
        } else {
            store_vec(v_dst, make_int4(0, 0, 0, 0));
        }
    }
}

// V is widened at half scale: every E2M1 code times half of a legal E4M3 group scale is exact in
// FP16 (at most four product fraction bits, magnitude <= 1344, smallest nonzero 2^-11). With P <=
// 1, a 32-key PV partial then stays below 43008 and accumulates in FP16 without overflow; the FP32
// accumulator is doubled once when the split's numerator is written.
__device__ __forceinline__ int4 k8v4_widen_half_f16x8(std::uint32_t packed,
                                                      std::uint8_t scale_code) {
    __nv_fp8_e4m3 encoded_scale;
    encoded_scale.__x    = scale_code;
    const __half scale   = __hmul(static_cast<__half>(encoded_scale), __float2half_rn(0.5F));
    const __half2 scale2 = __halves2half2(scale, scale);
    unsigned half_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        __nv_fp4x2_e2m1 encoded;
        encoded.__x         = static_cast<std::uint8_t>(packed >> (8 * pair));
        const __half2 value = __hmul2(static_cast<__half2>(encoded), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                     static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
}

// Widens one packed NVFP4 V tile at half scale to the swizzled FP16 [key][d] image read by
// ldmatrix.trans. Keys outside the split were copied as zero codes with zero scales, so they widen
// to zero. All codes and scales are loaded before the first conversion so their shared-memory
// latencies overlap.
template <int Bc, int Threads>
__device__ __forceinline__ void k8v4_widen_v_tile(int thread, const std::uint8_t* v_codes,
                                                  const std::uint8_t* v_scale, __half* v_f16) {
    constexpr int D         = kCausalHeadDim;
    constexpr int Chunks    = Bc * (D / 8);
    constexpr int PerThread = (Chunks + Threads - 1) / Threads;
    std::uint32_t packed[PerThread];
    std::uint8_t scales[PerThread];
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int chunk = thread + i * Threads;
        if (Chunks % Threads == 0 || chunk < Chunks) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            packed[i]       = load_vec<std::uint32_t>(&v_codes[key_l * (D / 2) + d / 2]);
            scales[i]       = v_scale[key_l * kKVCacheNvfp4Groups + d / kKVCacheNvfp4Group];
        }
    }
#pragma unroll
    for (int i = 0; i < PerThread; ++i) {
        const int chunk = thread + i * Threads;
        if (Chunks % Threads == 0 || chunk < Chunks) {
            const int key_l = chunk / (D / 8);
            const int d     = (chunk - key_l * (D / 8)) * 8;
            store_vec(&v_f16[key_l * D + causal_small_t_tc_swz(key_l, d)],
                      k8v4_widen_half_f16x8(packed[i], scales[i]));
        }
    }
}

// Adds one 32-key PV product (two 16-key steps chained in FP16 at twice the FP32-accumulating MMA
// rate) to an FP32 accumulator fragment holding half the numerator.
__device__ __forceinline__ void k8v4_pv_pair(float (&acc)[4], const unsigned (&p0)[4],
                                             const unsigned (&p1)[4], unsigned v0, unsigned v1,
                                             unsigned v2, unsigned v3) {
    unsigned top_bits;
    unsigned bottom_bits;
    mma_f16_f16acc(top_bits, bottom_bits, p0[0], p0[1], p0[2], p0[3], v0, v1);
    mma_f16_f16acc(top_bits, bottom_bits, p1[0], p1[1], p1[2], p1[3], v2, v3, top_bits,
                   bottom_bits);
    const float2 top    = __half22float2(*reinterpret_cast<const __half2*>(&top_bits));
    const float2 bottom = __half22float2(*reinterpret_cast<const __half2*>(&bottom_bits));
    acc[0] += top.x;
    acc[1] += top.y;
    acc[2] += bottom.x;
    acc[3] += bottom.y;
}

// Key visibility of one query row: every key before `base`, then the keys at or before the query
// position qabs whose offset from base is set in the row's ancestor-or-self column mask. A causal
// chain passes base = qabs + 1, which leaves exactly the keys at or before qabs.
__device__ __forceinline__ bool k8v4_key_visible(int key, int qabs, int base, unsigned mask) {
    return key < base || (key <= qabs && ((mask >> (key - base)) & 1U) != 0U);
}

// Scores one 16-row x Bc-key tile: scales, causal/split masking and the online-softmax update of
// the row statistics. Leaves P (unnormalized, <= 1) in `score` and the row rescale factors.
//
// Most tiles lie inside their split and wholly before every query row's position. There no key is
// masked and every score and row maximum is finite, so the same scaling and exponentials run
// without the per-element mask and infinity selects.
template <typename Geometry, int QKNt>
__device__ __forceinline__ void
k8v4_softmax_tile(float (&score)[QKNt][4], const __half* k_scale, float q_scale_r0,
                  float q_scale_r1, bool row0_valid, bool row1_valid, int qabs0, int qabs1,
                  int base0, int base1, unsigned mask0, unsigned mask1, int k0,
                  const K8V4SplitRange& range, float attention_scale, int lid, float& m0, float& m1,
                  float& l0, float& l1, float& alpha0, float& alpha1) {
    constexpr float Log2E       = 1.4426950408889634074F;
    constexpr unsigned FullMask = 0xffffffffU;
#pragma unroll
    for (int nt = 0; nt < QKNt; ++nt) {
        const int keya  = nt * 8 + 2 * lid;
        const float ks0 = __half2float(k_scale[keya]);
        const float ks1 = __half2float(k_scale[keya + 1]);
        score[nt][0] *= q_scale_r0 * ks0;
        score[nt][1] *= q_scale_r0 * ks1;
        score[nt][2] *= q_scale_r1 * ks0;
        score[nt][3] *= q_scale_r1 * ks1;
    }

    const int last_key  = k0 + QKNt * 8 - 1;
    const bool unmasked = __all_sync(
        FullMask, row0_valid && row1_valid && k0 >= range.split_start &&
                      last_key < range.split_end && last_key < base0 && last_key < base1);
    float bm0 = -CUDART_INF_F;
    float bm1 = -CUDART_INF_F;
    if (unmasked) {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) score[nt][i] *= attention_scale;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
    } else {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int key0 = k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
            const bool in0 = key0 >= range.split_start && key0 < range.split_end;
            const bool in1 = key1 >= range.split_start && key1 < range.split_end;
            score[nt][0]   = row0_valid && in0 && k8v4_key_visible(key0, qabs0, base0, mask0)
                                 ? score[nt][0] * attention_scale
                                 : -CUDART_INF_F;
            score[nt][1]   = row0_valid && in1 && k8v4_key_visible(key1, qabs0, base0, mask0)
                                 ? score[nt][1] * attention_scale
                                 : -CUDART_INF_F;
            score[nt][2]   = row1_valid && in0 && k8v4_key_visible(key0, qabs1, base1, mask1)
                                 ? score[nt][2] * attention_scale
                                 : -CUDART_INF_F;
            score[nt][3]   = row1_valid && in1 && k8v4_key_visible(key1, qabs1, base1, mask1)
                                 ? score[nt][3] * attention_scale
                                 : -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
    }
    bm0             = warp_max<4>(bm0, FullMask);
    bm1             = warp_max<4>(bm1, FullMask);
    const float nm0 = fmaxf(m0, bm0);
    const float nm1 = fmaxf(m1, bm1);
    alpha0          = m0 == -CUDART_INF_F ? 0.0F : exp2_approx((m0 - nm0) * Log2E);
    alpha1          = m1 == -CUDART_INF_F ? 0.0F : exp2_approx((m1 - nm1) * Log2E);
    float bl0       = 0.0F;
    float bl1       = 0.0F;
    if (unmasked) {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = exp2_approx((score[nt][0] - nm0) * Log2E);
            score[nt][1] = exp2_approx((score[nt][1] - nm0) * Log2E);
            score[nt][2] = exp2_approx((score[nt][2] - nm1) * Log2E);
            score[nt][3] = exp2_approx((score[nt][3] - nm1) * Log2E);
            bl0 += score[nt][0] + score[nt][1];
            bl1 += score[nt][2] + score[nt][3];
        }
    } else {
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F
                               ? exp2_approx((score[nt][0] - nm0) * Log2E)
                               : 0.0F;
            score[nt][1] = nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F
                               ? exp2_approx((score[nt][1] - nm0) * Log2E)
                               : 0.0F;
            score[nt][2] = nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F
                               ? exp2_approx((score[nt][2] - nm1) * Log2E)
                               : 0.0F;
            score[nt][3] = nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F
                               ? exp2_approx((score[nt][3] - nm1) * Log2E)
                               : 0.0F;
            bl0 += score[nt][0] + score[nt][1];
            bl1 += score[nt][2] + score[nt][3];
        }
    }
    bl0 = warp_sum<4>(bl0, FullMask);
    bl1 = warp_sum<4>(bl1, FullMask);
    l0  = __fmaf_rn(l0, alpha0, bl0);
    l1  = __fmaf_rn(l1, alpha1, bl1);
    m0  = nm0;
    m1  = nm1;
}

__device__ __forceinline__ unsigned k8v4_half2_bits(float lo, float hi) {
    const __half2 pair = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&pair);
}

// ----------------------------------------------------------------------------------------------
// Narrow blocks: producer warps score, all warps multiply P by V.

// Tree: tree_masks (I32 [full_width, B], laid out like positions) gives every query column its
// ancestor-or-self columns of a verify tree whose columns occupy consecutive cache positions from
// the row's column 0; such a column sees the keys before column 0 and its ancestors' keys only.
template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, bool Tree, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void causal_attention_small_t_k8v4_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
        std::uint8_t* cache_k, std::uint8_t* cache_v, __half* cache_k_scale,
        std::uint8_t* cache_v_scale, const std::int32_t* block_tables,
        const std::int32_t* valid_columns, const std::int32_t* tree_masks,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kCausalHeadDim;
    constexpr int DB16                 = D / 2;
    constexpr int Threads              = Wc * 32;
    constexpr int QKKs                 = D / 32;
    constexpr int QKNt                 = Bc / 8;
    constexpr int PStride              = Bc == 32 ? 64 : Bc;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int PageIds              = 64;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;

    static_assert(TokenTile >= 1 && TokenTile * Geometry::GroupSize <= 48);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc > RowTiles && Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);
    static_assert(QKKs == 8);
    constexpr int KStageBytes = Bc * D;
    constexpr int StageBytes  = KStageBytes + Bc * D / 2;
    constexpr int ArenaBytes  = StageBytes + Bc * D * 2;
    static_assert(ArenaBytes >= Wc * D * static_cast<int>(sizeof(float)));
    __shared__ __align__(16) std::uint8_t q_s[Br * D];
    __shared__ __align__(16) std::uint8_t static_arena[DynamicArena ? 16 : ArenaBytes];
    extern __shared__ __align__(16) std::uint8_t dynamic_arena[];
    std::uint8_t* arena   = DynamicArena ? dynamic_arena : static_arena;
    std::uint8_t* k_fp8   = arena;
    std::uint8_t* v_nvfp4 = arena + KStageBytes;
    __half* v_f16         = reinterpret_cast<__half*>(arena + StageBytes);
    __nv_bfloat16* q_b16  = reinterpret_cast<__nv_bfloat16*>(q_s);
    __shared__ __align__(16) __half p_s[Br * PStride];
    __shared__ float alpha_s[Br];
    __shared__ float q_scale_s[Br];
    __shared__ __align__(16) __half k_scale_s[Bc];
    __shared__ __align__(16) std::uint8_t v_scale_s[Bc * kKVCacheNvfp4Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];
    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    positions += column_base;
    if constexpr (Tree) { tree_masks += column_base; }
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    const auto write_neutral = [&]() {
        k8v4_write_neutral<Geometry, TokenTile, Threads>(tid, kv_head, split, partial_acc,
                                                         partial_m, partial_l);
    };
    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) return;
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = positions[0];
    const std::int32_t last_pos  = positions[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, TokenTile);
    if (split >= active_split_count) return;

    const K8V4SplitRange range = k8v4_split_range<TokenTile, Bc>(window, split, active_split_count);
    if (range.key_blocks == 0) {
        write_neutral();
        return;
    }
    const int first_page = range.first_tile >> kPagedKVPageShift;
    const int page_count = ((range.split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    __syncthreads();
    const auto page_of = [&](int key) {
        return physical_pages_s[(key >> kPagedKVPageShift) - first_page];
    };

    if constexpr (CacheInput::writes_cache) {
        k8v4_append_split_columns<Geometry, Wc>(
            input, positions, valid_tokens, range, page_of, kv_head, cache_k, cache_v,
            cache_k_scale, cache_v_scale, reinterpret_cast<float*>(arena), warp, lane);
        __syncthreads();
    }

    k8v4_quantize_query<Geometry, TokenTile, Threads>(q, kv_head, q_s, q_scale_s, tid);

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    float q_scale_r0 = 0.0F;
    float q_scale_r1 = 0.0F;
    int qabs0        = -1;
    int qabs1        = -1;
    int base0        = 0;
    int base1        = 0;
    unsigned mask0   = 0U;
    unsigned mask1   = 0U;
    if (warp < RowTiles) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        q_scale_r0     = q_scale_s[row0];
        q_scale_r1     = q_scale_s[row1];
        int q_head = 0, token0 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token1);
        qabs0 = row0 < RowCount ? positions[token0] : -1;
        qabs1 = row1 < RowCount ? positions[token1] : -1;
        base0 = qabs0 + 1;
        base1 = qabs1 + 1;
        if constexpr (Tree) {
            base0 = row0 < RowCount ? positions[0] : 0;
            base1 = row1 < RowCount ? positions[0] : 0;
            mask0 = row0 < RowCount ? static_cast<unsigned>(tree_masks[token0]) : 0U;
            mask1 = row1 < RowCount ? static_cast<unsigned>(tree_masks[token1]) : 0U;
        }
    }

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0F;
    float l1 = 0.0F;

    k8v4_issue_tile<Geometry, Bc, Threads>(tid, k_fp8, v_nvfp4, k_scale_s, v_scale_s,
                                           range.first_tile, page_of(range.first_tile), range,
                                           kv_head, cache_k, cache_v, cache_k_scale, cache_v_scale);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < range.key_blocks; ++kb) {
        const int k0      = range.first_tile + kb * Bc;
        const auto* k_b16 = reinterpret_cast<const __nv_bfloat16*>(k_fp8);
        if (warp < RowTiles) {
            const int row_base = warp * 16;
            __half* p_sw       = &p_s[row_base * PStride];
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;
            }
#pragma unroll
            for (int kk = 0; kk < QKKs; ++kk) {
                const int acol = kk * 16 + a_coloff;
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_small_t_tc_swz(row_base + a_rowoff, acol)]));
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow = nt * 8 + b_rin;
                    const int bcol = kk * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_small_t_tc_swz(brow, bcol)]));
                    mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                 af[1], af[2], af[3], bf[0], bf[1]);
                }
            }
            const int row0 = row_base + gid;
            float alpha0   = 0.0F;
            float alpha1   = 0.0F;
            k8v4_softmax_tile<Geometry, QKNt>(score, k_scale_s, q_scale_r0, q_scale_r1,
                                              row0 < RowCount, row0 + 8 < RowCount, qabs0, qabs1,
                                              base0, base1, mask0, mask1, k0, range,
                                              attention_scale, lid, m0, m1, l0, l1, alpha0, alpha1);
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                *reinterpret_cast<__half2*>(
                    &p_sw[gid * PStride + causal_small_t_tc_swz(gid, col0)]) =
                    __floats2half2_rn(score[nt][0], score[nt][1]);
                *reinterpret_cast<__half2*>(
                    &p_sw[(gid + 8) * PStride + causal_small_t_tc_swz(gid + 8, col0)]) =
                    __floats2half2_rn(score[nt][2], score[nt][3]);
            }
            if (lid == 0) {
                alpha_s[row0]     = alpha0;
                alpha_s[row0 + 8] = alpha1;
            }
        } else {
            k8v4_widen_v_tile<Bc, VLoaderThreads>(tid - ProducerThreads, v_nvfp4, v_scale_s, v_f16);
        }
        __syncthreads();

        // The tile consumed above is no longer read; refill it while PV runs.
        if (kb + 1 < range.key_blocks) {
            const int next_k0 = k0 + Bc;
            k8v4_issue_tile<Geometry, Bc, Threads>(tid, k_fp8, v_nvfp4, k_scale_s, v_scale_s,
                                                   next_k0, page_of(next_k0), range, kv_head,
                                                   cache_k, cache_v, cache_k_scale, cache_v_scale);
            cp_commit();
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __half* p_consumer          = &p_s[consumer_row_base * PStride];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }
        // P fragments depend only on the key step; load them once for every output tile.
        unsigned pf[PVKs][4];
#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(
                pf[k][0], pf[k][1], pf[k][2], pf[k][3],
                smem_addr(&p_consumer[a_rowoff * PStride + causal_small_t_tc_swz(a_rowoff, pcol)]));
        }
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; k += 2) {
                unsigned vf[4];
                const int vrow = k * 16 + lane;
                const int vcol = global_n * 8;
                ldmatrix_x4_t(vf[0], vf[1], vf[2], vf[3],
                              smem_addr(&v_f16[vrow * D + causal_small_t_tc_swz(vrow, vcol)]));
                k8v4_pv_pair(acc[n], pf[k], pf[k + 1], vf[0], vf[1], vf[2], vf[3]);
            }
        }
        if (kb + 1 < range.key_blocks) cp_wait<0>();
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) =
                make_float2(2.0F * acc[n][0], 2.0F * acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                causal_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<float2*>(&partial_acc[dst]) =
                make_float2(2.0F * acc[n][2], 2.0F * acc[n][3]);
        }
    }
}

// ----------------------------------------------------------------------------------------------
// Wide blocks: warp-specialized. Compute warps each own one 16-row tile; loader warps stream K/V.

template <typename Geometry, int TokenTile>
struct K8V4WideSchedule {
    static constexpr int RowCount       = TokenTile * Geometry::GroupSize;
    static constexpr int RowTiles       = (RowCount + 15) / 16;
    static constexpr int ComputeWarps   = RowTiles;
    static constexpr int LoaderWarps    = 2;
    static constexpr int Warps          = ComputeWarps + LoaderWarps;
    static constexpr int Threads        = Warps * 32;
    static constexpr int ComputeThreads = ComputeWarps * 32;
    static constexpr int LoaderThreads  = LoaderWarps * 32;
    // Warp w issues on SM sub-partition w % 4, so compute warps 4 and 5 share the Tensor Cores of
    // sub-partitions 0 and 1 with compute warps 0 and 1 while the loader warps (6 and 7) sit on
    // sub-partitions 2 and 3. Loader warp j therefore runs the PV product of row tile 4 + j from
    // the P fragments that compute warp 4 + j hands over, which balances the Tensor Core work.
    static constexpr int AssistedTiles = RowTiles > 4 ? RowTiles - 4 : 0;
    static_assert(AssistedTiles <= LoaderWarps);
    // With two assisted tiles, compute warps 4 and 5 would otherwise wait for most of each key tile
    // while warps 0-3 run their whole PV products. Row tile r < 4 therefore hands the P fragments
    // of every tile to compute warp 4 + (r & 1), which accumulates the last TailColumnTiles column
    // tiles (8 value dimensions each) of that row tile; the owner keeps the rest.
    static constexpr int ColumnTiles     = kCausalHeadDim / 8;
    static constexpr int TailColumnTiles = AssistedTiles == 2 ? 10 : 0;
    static constexpr int OwnColumnTiles  = ColumnTiles - TailColumnTiles;
    static constexpr int Bc             = 32;
    // Copies run PackedStages - 2 tiles ahead of the widening, which runs one tile ahead of the
    // compute warps; three tiles in flight per SM keep the cache stream at DRAM bandwidth.
    static constexpr int PackedStages = 5;
    // Page ids of a split's first pages are staged in shared memory; later pages (splits longer
    // than PageIds * 64 keys) are read from the block table directly.
    static constexpr int PageIds          = 128;
    static constexpr int KBytes           = Bc * kCausalHeadDim;
    static constexpr int VBytes           = Bc * kCausalHeadDim / 2;
    static constexpr int KScaleBytes      = Bc * static_cast<int>(sizeof(__half));
    static constexpr int VScaleBytes      = Bc * kKVCacheNvfp4Groups;
    static constexpr int PackedStageBytes = KBytes + VBytes + KScaleBytes + VScaleBytes;
    static constexpr int WideSlotBytes    = Bc * kCausalHeadDim * static_cast<int>(sizeof(__half));
    static constexpr int PackedBytes      = PackedStages * PackedStageBytes;
    static constexpr int DynamicBytes     = PackedBytes + 2 * WideSlotBytes;
    static_assert(RowTiles >= 4 && RowTiles <= 6);
    static_assert(PackedStageBytes % 16 == 0);
    // The query tile lives in the widened-V slots and the append scratch in the packed stages
    // until the pipeline starts.
    static_assert(RowTiles * 16 * kCausalHeadDim <= 2 * WideSlotBytes);
    static_assert(Warps * kCausalHeadDim * static_cast<int>(sizeof(float)) <= PackedBytes);
    // Tail handoffs of a tile live in its stage's packed V region and V scales, which are dead once
    // the loaders have widened the tile (full) and are refilled only after it is released (empty).
    static_assert(TailColumnTiles == 0 ||
                  (4 * 2 * 32 * static_cast<int>(sizeof(uint4)) <= VBytes &&
                   4 * 16 * static_cast<int>(sizeof(float)) <= VScaleBytes));
};

// Named barrier ids of the wide kernel (0 is __syncthreads).
namespace k8v4_wide_barrier {
inline constexpr int kQueryReleased = 1;
inline constexpr int kLoaders       = 2;
} // namespace k8v4_wide_barrier

__device__ __forceinline__ void k8v4_bar_sync(int id, int threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ void k8v4_bar_arrive(int id, int threads) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, bool Tree,
          typename CacheInput>
__launch_bounds__(K8V4WideSchedule<Geometry, TokenTile>::Threads, 1) __global__
    void causal_attention_small_t_k8v4_wide_kernel(
        const std::uint8_t* query_codes, const float* query_scales, CacheInput input,
        const std::int32_t* positions, std::uint8_t* cache_k, std::uint8_t* cache_v,
        __half* cache_k_scale, std::uint8_t* cache_v_scale, const std::int32_t* block_tables,
        const std::int32_t* valid_columns, const std::int32_t* tree_masks,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float attention_scale,
        float* partial_acc, float* partial_m, float* partial_l) {
    using Schedule         = K8V4WideSchedule<Geometry, TokenTile>;
    namespace barrier      = k8v4_wide_barrier;
    constexpr int RowCount = Schedule::RowCount;
    constexpr int Threads  = Schedule::Threads;
    constexpr int Bc       = Schedule::Bc;
    constexpr int D        = kCausalHeadDim;
    constexpr int DB16     = D / 2;
    constexpr int QKKs     = D / 32;
    constexpr int QKNt     = Bc / 8;
    constexpr int PVNt     = D / 8;
    static_assert(QKKs == 8 && Bc == 32);

    extern __shared__ __align__(16) std::uint8_t dynamic_arena[];
    __shared__ float q_scale_s[Schedule::RowTiles * 16];
    // Tile handoff. full[slot] completes once every loader thread has widened a tile into the
    // slot; empty[slot] once every thread that reads it (compute warps, and loader warps running an
    // assisted PV product) has finished it. Compute warps wait only on the loaders, never on one
    // another, so their softmax and Tensor Core phases drift apart.
    __shared__ __align__(8) std::uint64_t full[2];
    __shared__ __align__(8) std::uint64_t empty[2];
    __shared__ std::int32_t physical_pages_s[Schedule::PageIds];
    // Assisted row tiles: compute warp 4 + j leaves its FP16 P fragments (lane-major, in the
    // m16n8k16 A-operand layout) and per-row rescale factors here for loader warp j. The buffer is
    // single: p_full[j] publishes a tile, p_empty[j] releases it once the loader holds it.
    constexpr int kAssistSlots = Schedule::AssistedTiles > 0 ? Schedule::AssistedTiles : 1;
    __shared__ __align__(16) uint4 p_handoff[kAssistSlots][2][32];
    __shared__ float alpha_handoff[kAssistSlots][16];
    __shared__ __align__(8) std::uint64_t p_full[kAssistSlots];
    __shared__ __align__(8) std::uint64_t p_empty[kAssistSlots];
    // Tail handoffs: row tile r < 4 publishes tile t's P fragments and rescale factors in the
    // tile's stage (packed V bytes, V scale bytes) through tail_full[stage][r]; the stage is not
    // refilled before the tile's empty barrier, which the tail warp arrives on after using them.
    constexpr int kTailStages = Schedule::TailColumnTiles > 0 ? Schedule::PackedStages : 1;
    __shared__ __align__(8) std::uint64_t tail_full[kTailStages][4];
    const auto packed = [&](int stage) {
        return dynamic_arena + stage * Schedule::PackedStageBytes;
    };
    const auto packed_k       = [&](int stage) { return packed(stage); };
    const auto packed_v       = [&](int stage) { return packed(stage) + Schedule::KBytes; };
    const auto packed_k_scale = [&](int stage) {
        return reinterpret_cast<__half*>(packed(stage) + Schedule::KBytes + Schedule::VBytes);
    };
    const auto packed_v_scale = [&](int stage) {
        return packed(stage) + Schedule::KBytes + Schedule::VBytes + Schedule::KScaleBytes;
    };
    __half* const wide_v    = reinterpret_cast<__half*>(dynamic_arena + Schedule::PackedBytes);
    const auto v_slot       = [&](int slot) { return wide_v + slot * Bc * D; };
    const auto tail_p       = [&](int stage, int row_tile) {
        return reinterpret_cast<uint4*>(packed_v(stage)) + row_tile * 2 * 32;
    };
    const auto tail_alpha = [&](int stage, int row_tile) {
        return reinterpret_cast<float*>(packed_v_scale(stage)) + row_tile * 16;
    };
    std::uint8_t* const q_s = reinterpret_cast<std::uint8_t*>(wide_v);

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : min(remaining, TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    positions += column_base;
    if constexpr (Tree) { tree_masks += column_base; }
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    const auto write_neutral = [&]() {
        k8v4_write_neutral<Geometry, TokenTile, Threads>(tid, kv_head, split, partial_acc,
                                                         partial_m, partial_l);
    };
    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) return;
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }
    const std::int32_t first_pos = positions[0];
    const std::int32_t last_pos  = positions[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }
    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, TokenTile);
    if (split >= active_split_count) return;
    const K8V4SplitRange range = k8v4_split_range<TokenTile, Bc>(window, split, active_split_count);
    if (range.key_blocks == 0) {
        write_neutral();
        return;
    }
    // The prepared query tile streams in first, while the block table and the first tiles load.
    k8v4_copy_prepared_query<Geometry, TokenTile, Threads>(query_codes, query_scales, batch,
                                                           kv_head, q_s, q_scale_s, tid);
    const int first_page = range.first_tile >> kPagedKVPageShift;
    const int page_count = ((range.split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < min(page_count, Schedule::PageIds); page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }
    __syncthreads();
    const auto page_of = [&](int key) {
        const int local = (key >> kPagedKVPageShift) - first_page;
        return local < Schedule::PageIds ? physical_pages_s[local]
                                         : block_table[key >> kPagedKVPageShift];
    };

    if constexpr (CacheInput::writes_cache) {
        k8v4_append_split_columns<Geometry, Schedule::Warps>(
            input, positions, valid_tokens, range, page_of, kv_head, cache_k, cache_v,
            cache_k_scale, cache_v_scale, reinterpret_cast<float*>(dynamic_arena), warp, lane);
        __syncthreads();
    }

    // Loader warps. Copies for tile j land in packed stage j % PackedStages; its widened V lands in
    // slot j % 2. The compute warps release both after finishing tile j.
    const bool is_loader = warp >= Schedule::ComputeWarps;
    const int loader     = tid - Schedule::ComputeThreads;
    const auto issue     = [&](int tile) {
        const int stage   = tile % Schedule::PackedStages;
        const int tile_k0 = range.first_tile + tile * Bc;
        k8v4_issue_tile<Geometry, Bc, Schedule::LoaderThreads>(
            loader, packed_k(stage), packed_v(stage), packed_k_scale(stage), packed_v_scale(stage),
            tile_k0, page_of(tile_k0), range, kv_head, cache_k, cache_v, cache_k_scale,
            cache_v_scale);
    };
    if (is_loader) {
#pragma unroll
        for (int tile = 0; tile < Schedule::PackedStages - 2; ++tile) {
            if (tile < range.key_blocks) issue(tile);
            cp_commit();
        }
        // The query group precedes the tile groups.
        cp_wait<Schedule::PackedStages - 2>();
    } else {
        cp_wait<0>();
    }
    if (tid == 0) {
        for (int slot = 0; slot < 2; ++slot) {
            cta_mbarrier_init(&full[slot], Schedule::LoaderThreads);
            cta_mbarrier_init(&empty[slot],
                              Schedule::ComputeThreads + Schedule::AssistedTiles * 32);
        }
        for (int j = 0; j < Schedule::AssistedTiles; ++j) {
            cta_mbarrier_init(&p_full[j], 32);
            cta_mbarrier_init(&p_empty[j], 32);
        }
        if constexpr (Schedule::TailColumnTiles > 0) {
            for (int stage = 0; stage < kTailStages; ++stage) {
                for (int row_tile = 0; row_tile < 4; ++row_tile) {
                    cta_mbarrier_init(&tail_full[stage][row_tile], 32);
                }
            }
        }
        cta_mbarrier_fence_init();
    }
    // Publishes the query tile and the initialized mbarriers.
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    // Adds one tile's PV product over column tiles [First, First + Count) (8 value dimensions each)
    // of a 16-row tile: rescale by the tile's row factors, then multiply the FP16 P fragments by
    // the widened V slot. The operations per element do not depend on which warp owns the column.
    const auto accumulate_pv = [&]<int First, int Count>(float (&acc)[Count][4],
                                                         const unsigned (&pf)[Bc / 16][4],
                                                         float alpha0, float alpha1, int slot) {
        // Unchanged row maxima give alpha == 1 exactly; skip the no-op rescale.
        if (__any_sync(0xffffffffU, alpha0 != 1.0F || alpha1 != 1.0F)) {
#pragma unroll
            for (int n = 0; n < Count; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }
        }
        const __half* v_f16 = v_slot(slot);
#pragma unroll
        for (int n = 0; n < Count; ++n) {
            unsigned vf[4];
            ldmatrix_x4_t(
                vf[0], vf[1], vf[2], vf[3],
                smem_addr(&v_f16[lane * D + causal_small_t_tc_swz(lane, (First + n) * 8)]));
            k8v4_pv_pair(acc[n], pf[0], pf[1], vf[0], vf[1], vf[2], vf[3]);
        }
    };

    // Writes column tiles [First, First + Count) of the split numerator of one 16-row tile
    // (doubling the half-scale accumulator).
    const auto write_numerator = [&]<int First, int Count>(const float (&acc)[Count][4],
                                                           int row_base) {
        const int row0 = row_base + gid;
        const int row1 = row0 + 8;
        int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
#pragma unroll
        for (int n = 0; n < Count; ++n) {
            const int d0 = (First + n) * 8 + 2 * lid;
            if (row0 < RowCount) {
                const std::int64_t dst =
                    causal_partial_acc_index<Geometry>(q_head0, d0, token0, split, TokenTile);
                *reinterpret_cast<float2*>(&partial_acc[dst]) =
                    make_float2(2.0F * acc[n][0], 2.0F * acc[n][1]);
            }
            if (row1 < RowCount) {
                const std::int64_t dst =
                    causal_partial_acc_index<Geometry>(q_head1, d0, token1, split, TokenTile);
                *reinterpret_cast<float2*>(&partial_acc[dst]) =
                    make_float2(2.0F * acc[n][2], 2.0F * acc[n][3]);
            }
        }
    };

    if (is_loader) {
        const int loader_warp = warp - Schedule::ComputeWarps;
        const bool assists    = loader_warp < Schedule::AssistedTiles;
        float acc[PVNt][4];
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
        }
        // PV product of tile t for row tile 4 + loader_warp, from the handed-over P fragments.
        const auto assist = [&](int t) {
            const int j = loader_warp;
            cta_mbarrier_wait(&p_full[j], t & 1);
            unsigned pf[Bc / 16][4];
#pragma unroll
            for (int k = 0; k < Bc / 16; ++k) {
                const uint4 bits = p_handoff[j][k][lane];
                pf[k][0]         = bits.x;
                pf[k][1]         = bits.y;
                pf[k][2]         = bits.z;
                pf[k][3]         = bits.w;
            }
            const float alpha0 = alpha_handoff[j][gid];
            const float alpha1 = alpha_handoff[j][gid + 8];
            cta_mbarrier_arrive(&p_empty[j]);
            accumulate_pv.template operator()<0, PVNt>(acc, pf, alpha0, alpha1, t & 1);
            cta_mbarrier_arrive(&empty[t & 1]);
        };

        // The first widened tiles overwrite the query tile.
        k8v4_bar_sync(barrier::kQueryReleased, Threads);

        for (int kb = 0; kb < range.key_blocks; ++kb) {
            const int slot = kb & 1;
            // Tile kb - 2 used this V slot and the packed stage refilled below. Its assisted PV
            // products ran in the previous iteration.
            if (kb >= 2) cta_mbarrier_wait(&empty[slot], ((kb - 2) >> 1) & 1);
            const int prefetch = kb + Schedule::PackedStages - 2;
            if (prefetch < range.key_blocks) issue(prefetch);
            cp_commit();
            cp_wait<Schedule::PackedStages - 2>();
            k8v4_bar_sync(barrier::kLoaders, Schedule::LoaderThreads);
            const int stage = kb % Schedule::PackedStages;
            k8v4_widen_v_tile<Bc, Schedule::LoaderThreads>(loader, packed_v(stage),
                                                           packed_v_scale(stage), v_slot(slot));
            cta_mbarrier_arrive(&full[slot]);
            if (assists && kb >= 1) assist(kb - 1);
        }
        if (assists) {
            assist(range.key_blocks - 1);
            write_numerator.template operator()<0, PVNt>(
                acc, (Schedule::ComputeWarps - Schedule::AssistedTiles + loader_warp) * 16);
        }
        return;
    }

    // Compute warps: warp w owns query rows [16w, 16w + 16). The last AssistedTiles of them score
    // and hand their P fragments to a loader warp instead of running the PV product.
    const int assisted_tile = warp - (Schedule::ComputeWarps - Schedule::AssistedTiles);
    const bool assisted     = assisted_tile >= 0;
    const int a_mat         = lane >> 3;
    const int a_rowoff      = (lane & 7) + ((a_mat & 1) << 3);
    const int a_coloff      = (a_mat >> 1) << 3;
    const int b_rin         = lane & 7;
    const int b_koff        = ((lane >> 3) & 1) << 3;
    const int row_base      = warp * 16;
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;

    unsigned qf[QKKs][4];
    {
        const auto* q_b16 = reinterpret_cast<const __nv_bfloat16*>(q_s);
#pragma unroll
        for (int kk = 0; kk < QKKs; ++kk) {
            const int acol = kk * 16 + a_coloff;
            ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                        smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                         causal_small_t_tc_swz(row_base + a_rowoff, acol)]));
        }
    }
    const float q_scale_r0 = q_scale_s[row0];
    const float q_scale_r1 = q_scale_s[row1];
    int qabs0              = -1;
    int qabs1              = -1;
    int base0              = 0;
    int base1              = 0;
    unsigned mask0         = 0U;
    unsigned mask1         = 0U;
    {
        int q_head = 0, token0 = 0, token1 = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token0);
        causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token1);
        if (row0 < RowCount) qabs0 = positions[token0];
        if (row1 < RowCount) qabs1 = positions[token1];
        base0 = qabs0 + 1;
        base1 = qabs1 + 1;
        if constexpr (Tree) {
            base0 = row0 < RowCount ? positions[0] : 0;
            base1 = row1 < RowCount ? positions[0] : 0;
            mask0 = row0 < RowCount ? static_cast<unsigned>(tree_masks[token0]) : 0U;
            mask1 = row1 < RowCount ? static_cast<unsigned>(tree_masks[token1]) : 0U;
        }
    }
    k8v4_bar_arrive(barrier::kQueryReleased, Threads);

    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0F;
    float l1 = 0.0F;

    // Scores key tile kb for this warp's rows once the loaders have widened it: QK, then the
    // online-softmax update. P lands in the m16n8k16 A-operand layout (key tiles 2k and 2k + 1 of
    // the score accumulator form the FP16 P fragment of key step k) with the row rescale factors.
    const auto score_tile = [&](int kb, unsigned (&pf)[Bc / 16][4], float& alpha0, float& alpha1) {
        const int slot  = kb & 1;
        const int stage = kb % Schedule::PackedStages;
        const int k0    = range.first_tile + kb * Bc;
        cta_mbarrier_wait(&full[slot], (kb >> 1) & 1);

        const auto* k_b16 = reinterpret_cast<const __nv_bfloat16*>(packed_k(stage));
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0F;
        }
#pragma unroll
        for (int kk = 0; kk < QKKs; ++kk) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int brow = nt * 8 + b_rin;
                const int bcol = kk * 16 + b_koff;
                unsigned bf[2];
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&k_b16[brow * DB16 + causal_small_t_tc_swz(brow, bcol)]));
                mma_fp8_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], qf[kk][0],
                             qf[kk][1], qf[kk][2], qf[kk][3], bf[0], bf[1]);
            }
        }
        k8v4_softmax_tile<Geometry, QKNt>(score, packed_k_scale(stage), q_scale_r0, q_scale_r1,
                                          row0 < RowCount, row1 < RowCount, qabs0, qabs1, base0,
                                          base1, mask0, mask1, k0, range, attention_scale, lid, m0,
                                          m1, l0, l1, alpha0, alpha1);
#pragma unroll
        for (int k = 0; k < Bc / 16; ++k) {
            pf[k][0] = k8v4_half2_bits(score[2 * k][0], score[2 * k][1]);
            pf[k][1] = k8v4_half2_bits(score[2 * k][2], score[2 * k][3]);
            pf[k][2] = k8v4_half2_bits(score[2 * k + 1][0], score[2 * k + 1][1]);
            pf[k][3] = k8v4_half2_bits(score[2 * k + 1][2], score[2 * k + 1][3]);
        }
    };

    // Hands tile kb's P fragments and row rescale factors to the assisting loader warp.
    const auto hand_to_loader = [&](int kb, const unsigned (&pf)[Bc / 16][4], float alpha0,
                                    float alpha1) {
        // The loader warp took the previous tile's fragments before this buffer is reused.
        if (kb >= 1) cta_mbarrier_wait(&p_empty[assisted_tile], (kb - 1) & 1);
#pragma unroll
        for (int k = 0; k < Bc / 16; ++k) {
            p_handoff[assisted_tile][k][lane] = make_uint4(pf[k][0], pf[k][1], pf[k][2], pf[k][3]);
        }
        if (lid == 0) {
            alpha_handoff[assisted_tile][gid]     = alpha0;
            alpha_handoff[assisted_tile][gid + 8] = alpha1;
        }
        cta_mbarrier_arrive(&p_full[assisted_tile]);
    };

    const auto write_statistics = [&]() {
        if (lid != 0) { return; }
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    };

    if constexpr (Schedule::TailColumnTiles > 0) {
        constexpr int Own  = Schedule::OwnColumnTiles;
        constexpr int Tail = Schedule::TailColumnTiles;
        if (assisted) {
            // Compute warp 4 + assisted_tile: a loader warp runs its PV product, and it runs the
            // tail column tiles of row tiles assisted_tile and assisted_tile + 2.
            float tail_acc[2][Tail][4];
#pragma unroll
            for (int source = 0; source < 2; ++source) {
#pragma unroll
                for (int n = 0; n < Tail; ++n) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) tail_acc[source][n][i] = 0.0F;
                }
            }
            for (int kb = 0; kb < range.key_blocks; ++kb) {
                unsigned pf[Bc / 16][4];
                float alpha0 = 0.0F;
                float alpha1 = 0.0F;
                score_tile(kb, pf, alpha0, alpha1);
                hand_to_loader(kb, pf, alpha0, alpha1);
                const int stage = kb % Schedule::PackedStages;
#pragma unroll
                for (int source = 0; source < 2; ++source) {
                    const int row_tile = assisted_tile + 2 * source;
                    cta_mbarrier_wait(&tail_full[stage][row_tile],
                                      (kb / Schedule::PackedStages) & 1);
                    const uint4* handed = tail_p(stage, row_tile);
                    unsigned tail_pf[Bc / 16][4];
#pragma unroll
                    for (int k = 0; k < Bc / 16; ++k) {
                        const uint4 bits = handed[k * 32 + lane];
                        tail_pf[k][0]    = bits.x;
                        tail_pf[k][1]    = bits.y;
                        tail_pf[k][2]    = bits.z;
                        tail_pf[k][3]    = bits.w;
                    }
                    const float* factors = tail_alpha(stage, row_tile);
                    accumulate_pv.template operator()<Own, Tail>(
                        tail_acc[source], tail_pf, factors[gid], factors[gid + 8], kb & 1);
                }
                cta_mbarrier_arrive(&empty[kb & 1]);
            }
            write_statistics();
            write_numerator.template operator()<Own, Tail>(tail_acc[0], assisted_tile * 16);
            write_numerator.template operator()<Own, Tail>(tail_acc[1], (assisted_tile + 2) * 16);
        } else {
            float acc[Own][4];
#pragma unroll
            for (int n = 0; n < Own; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
            }
            for (int kb = 0; kb < range.key_blocks; ++kb) {
                unsigned pf[Bc / 16][4];
                float alpha0 = 0.0F;
                float alpha1 = 0.0F;
                score_tile(kb, pf, alpha0, alpha1);
                // Publish the tail column tiles' operands before running the owned ones.
                const int stage = kb % Schedule::PackedStages;
                uint4* handed   = tail_p(stage, warp);
#pragma unroll
                for (int k = 0; k < Bc / 16; ++k) {
                    handed[k * 32 + lane] = make_uint4(pf[k][0], pf[k][1], pf[k][2], pf[k][3]);
                }
                if (lid == 0) {
                    tail_alpha(stage, warp)[gid]     = alpha0;
                    tail_alpha(stage, warp)[gid + 8] = alpha1;
                }
                cta_mbarrier_arrive(&tail_full[stage][warp]);
                accumulate_pv.template operator()<0, Own>(acc, pf, alpha0, alpha1, kb & 1);
                cta_mbarrier_arrive(&empty[kb & 1]);
            }
            write_statistics();
            write_numerator.template operator()<0, Own>(acc, row_base);
        }
    } else {
        float acc[PVNt][4];
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[n][i] = 0.0F;
        }
        for (int kb = 0; kb < range.key_blocks; ++kb) {
            unsigned pf[Bc / 16][4];
            float alpha0 = 0.0F;
            float alpha1 = 0.0F;
            score_tile(kb, pf, alpha0, alpha1);
            if (assisted) {
                hand_to_loader(kb, pf, alpha0, alpha1);
            } else {
                accumulate_pv.template operator()<0, PVNt>(acc, pf, alpha0, alpha1, kb & 1);
            }
            cta_mbarrier_arrive(&empty[kb & 1]);
        }
        write_statistics();
        if (!assisted) write_numerator.template operator()<0, PVNt>(acc, row_base);
    }
}

template <typename Geometry, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void causal_attention_small_t_k8v4_reduce_output_kernel(
    const float* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    const int q_head      = static_cast<int>(blockIdx.x);
    const int flat_column = static_cast<int>(blockIdx.y);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = static_cast<int>(threadIdx.x);
    if (q_head >= Geometry::QHeads || token >= tokens) return;
    if constexpr (MultiBatch) {
        if (batch >= batch_size) return;
    }
    if constexpr (Offset) positions += column_begin;
    if constexpr (MultiBatch) positions += static_cast<std::int64_t>(batch) * full_width;
    const int window  = positions[tokens - 1] + 1;
    int output_column = token;
    if constexpr (Offset) output_column += column_begin;
    if constexpr (MultiBatch) output_column += batch * full_width;
    if constexpr (Masked) {
        const int absolute_column = token + (Offset ? column_begin : 0);
        if (absolute_column >= valid_columns[batch]) {
            if (tid < kCausalHeadDim)
                out[causal_q_index<Geometry>(q_head, tid, output_column)] = __float2bfloat16(0.0f);
            return;
        }
    }


    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kCausalHeadDim * Geometry::QHeads *
                       tokens * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }
    const int active_splits =
        causal_small_t_quantized_active_splits<Geometry>(window, split_count, tokens);
    __shared__ float weights[256], warp_sums[8], scalars[2];
    __shared__ float normalized[256];
    const float head_l = causal_merge_split_statistics<Geometry>(
        partial_m, partial_l, q_head, token, tokens, active_splits, weights, warp_sums, scalars);

    float numerator = 0.0F;
    for (int split = 0; split < active_splits; ++split) {
        if (weights[split] != 0.0f)
            numerator +=
                partial_acc[causal_partial_acc_index<Geometry>(q_head, tid, token, split, tokens)] *
                weights[split];
    }
    normalized[tid] = head_l > 0.0F ? numerator / head_l : 0.0F;
    __syncthreads();

    // This (head, token)'s split numerators were consumed above and stay dead until the next split
    // pass rewrites them, so their lines leave L2 without a write-back.
    for (int line = tid; line < active_splits * (kCausalHeadDim / 32); line += 256) {
        discard_l2_line(&partial_acc[causal_partial_acc_index<Geometry>(q_head, (line & 7) * 32,
                                                                        token, line >> 3, tokens)]);
    }
    if (tid >= 32) return;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = normalized[tid + 32 * r];
    // R is symmetric, but this application is the inverse/transpose semantic boundary.
    normalized_hadamard_d256_inplace(values, tid);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d                                             = tid + 32 * r;
        out[causal_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(values[r]);
    }
}

} // namespace ninfer::ops
