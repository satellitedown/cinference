// Modified by satellitedown for Cinference: add the record-route convolution output policy.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "ops/common/memory.cuh"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8GdnInputOutput {
    static constexpr std::int32_t kQueryRows = 2048;
    static constexpr std::int32_t kKeyRows   = 2048;
    static constexpr std::int32_t kValueRows = 6144;
    static constexpr std::int32_t kQkvRows   = kQueryRows + kKeyRows + kValueRows;
    static constexpr std::int32_t kZRows     = 6144;
    static constexpr std::int32_t kRows      = kQkvRows + kZRows;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

static_assert(Fp8GdnInputOutput::kRows == 16384);
static_assert((Fp8GdnInputOutput::kQkvRows % 128) == 0);
static_assert((Fp8GdnInputOutput::kZRows % 128) == 0);

// Record route of one width-16 speculative block: a 16-token GEMM tile holds the whole sequence for
// its TileRows channels, so the channels' convolution runs on the staged BF16 tile rather than
// reading the record back in a second kernel. The projection still lands in the record exactly as
// the store-only policy writes it, and every token's output is GdnConvEpilogue::write_token on the
// same BF16 inputs. A token's taps are the three projected inputs before it, so the CTA's threads
// take independent runs of consecutive tokens; only the first run starts from the conv state.
template <int TileRows, int Threads>
struct Fp8GdnRecordConvOutput : Fp8GdnInputOutput {
    static constexpr int kWidth     = 16;
    static constexpr int kRuns      = Threads / TileRows;
    static constexpr int kRunTokens = kWidth / kRuns;
    static_assert((kQkvRows % TileRows) == 0 && (Threads % TileRows) == 0);
    static_assert((kWidth % kRuns) == 0 && kRunTokens >= 3);

    GdnConvEpilogue<NoHistoryPublish> conv;

    struct TileState {
        GdnConvChannel channel;
    };

    __device__ __forceinline__ TileState begin_tile(std::int32_t row_begin,
                                                    std::int32_t token_begin, std::int32_t) const {
        TileState state{};
        if (row_begin < kQkvRows) {
            const int channel = static_cast<int>(threadIdx.x) % TileRows;
            state.channel = conv.load_channel<kWidth>(row_begin + channel, token_begin / kWidth);
            // Left alone the compiler sinks these loads to their use after the main loop, where
            // their latency (two dependent round trips) would extend the tile. Consuming them here
            // overlaps it with the first weight stages instead.
            asm volatile(""
                         : "+f"(state.channel.s0), "+f"(state.channel.s1), "+f"(state.channel.s2),
                           "+f"(state.channel.w0), "+f"(state.channel.w1), "+f"(state.channel.w2),
                           "+f"(state.channel.w3), "+r"(state.channel.valid));
        }
        return state;
    }

    __device__ __forceinline__ void finish_tile(const TileState& state, const __nv_bfloat16* tile,
                                                int stride, std::int32_t row_begin,
                                                std::int32_t token_begin, std::int32_t) const {
        if (row_begin >= kQkvRows) { return; }
        const int channel  = static_cast<int>(threadIdx.x) % TileRows;
        const int first    = static_cast<int>(threadIdx.x) / TileRows * kRunTokens;
        const int row      = row_begin + channel;
        const auto batch   = static_cast<std::int64_t>(token_begin / kWidth);
        const auto project = [&](int token) {
            return __bfloat162float(tile[token * stride + channel]);
        };
        float s0 = first == 0 ? state.channel.s0 : project(first - 3);
        float s1 = first == 0 ? state.channel.s1 : project(first - 2);
        float s2 = first == 0 ? state.channel.s2 : project(first - 1);
#pragma unroll
        for (int step = 0; step < kRunTokens; ++step) {
            const int token           = first + step;
            const std::int64_t column = batch * kWidth + token;
            if (token >= state.channel.valid) {
                conv.write_output(row, column, __float2bfloat16_rn(0.0F));
                continue;
            }
            const float p = project(token);
            conv.write_token(row, column, state.channel, s0, s1, s2, p);
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
    }
};

} // namespace ninfer::ops::detail
