// Modified by satellitedown for Cinference: split the convolution into load and per-token steps.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct SnapshotHistoryPublish {
    __nv_bfloat16* state_write;
    const std::int32_t* snapshot_base_slots;
    std::int32_t channels;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t base =
            static_cast<std::int64_t>(snapshot_base_slots[batch] + token) * slot_stride;
        state_write[base + row]                  = __float2bfloat16_rn(s1);
        state_write[base + channels + row]       = __float2bfloat16_rn(s2);
        state_write[base + 2LL * channels + row] = __float2bfloat16_rn(p);
    }
};

struct RecordColumnPublish {
    __nv_bfloat16* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float, float, float p) const {
        const std::int64_t column       = static_cast<std::int64_t>(batch) * width + token;
        record[column * channels + row] = __float2bfloat16_rn(p);
    }
};

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(std::int32_t, std::int32_t, std::int32_t, float, float,
                                            float) const {}
};

// One channel's convolution inputs for one sequence: its three history taps, its four weights and
// the sequence's live width. Loading them apart from the scan lets a caller issue the loads early.
struct GdnConvChannel {
    float s0;
    float s1;
    float s2;
    float w0;
    float w1;
    float w2;
    float w3;
    std::int32_t valid;
};

// Device-side implementation detail shared by exact packed projection kernels. Projection
// accumulators stay in the route's existing private precision; Publish changes only the side
// effect after the convolution has consumed that accumulator.
template <class Publish>
struct GdnConvEpilogue {
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;
    std::int32_t width;
    std::int32_t batch_row;
    Publish publish;

    template <int Tokens>
    __device__ __forceinline__ GdnConvChannel load_channel(std::int32_t row,
                                                           std::int32_t batch) const {
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base =
            static_cast<std::int64_t>(initial_slots[batch]) * slot_stride;
        std::int32_t valid = valid_columns == nullptr ? Tokens : valid_columns[batch];
        valid              = valid < 0 ? 0 : (valid > Tokens ? Tokens : valid);
        return {
            __bfloat162float(state_read[initial_base + row]),
            __bfloat162float(state_read[initial_base + channels + row]),
            __bfloat162float(state_read[initial_base + 2LL * channels + row]),
            __bfloat162float(conv_weight[row]),
            __bfloat162float(conv_weight[channels + row]),
            __bfloat162float(conv_weight[2LL * channels + row]),
            __bfloat162float(conv_weight[3LL * channels + row]),
            valid,
        };
    }

    __device__ __forceinline__ void write_output(std::int32_t row, std::int64_t column,
                                                 __nv_bfloat16 output) const {
        if (row < query_rows) {
            query[column * query_rows + row] = output;
        } else if (row < query_rows + key_rows) {
            key[column * key_rows + row - query_rows] = output;
        } else {
            value[column * value_rows + row - query_rows - key_rows] = output;
        }
    }

    // One live token: taps s0..s2 are the three preceding projected inputs, p is its own.
    __device__ __forceinline__ void write_token(std::int32_t row, std::int64_t column,
                                                const GdnConvChannel& channel, float s0, float s1,
                                                float s2, float p) const {
        float conv = fmaf(channel.w0, s0, 0.0F);
        conv       = fmaf(channel.w1, s1, conv);
        conv       = fmaf(channel.w2, s2, conv);
        conv       = fmaf(channel.w3, p, conv);
        write_output(row, column, __float2bfloat16_rn(silu(conv)));
    }

    template <int Tokens>
    __device__ __forceinline__ void apply(std::int32_t row, std::int32_t batch,
                                          const GdnConvChannel& channel,
                                          const float (&projected)[Tokens]) const {
        float s0 = channel.s0;
        float s1 = channel.s1;
        float s2 = channel.s2;
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
            if (token >= channel.valid) {
                write_output(row, column, __float2bfloat16_rn(0.0F));
                continue;
            }

            const float p = projected[token];
            write_token(row, column, channel, s0, s1, s2, p);
            publish.publish(token, batch, row, s1, s2, p);
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
    }

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        const std::int32_t row = global_row_offset + local_row;
        apply<Tokens>(row, batch_row, load_channel<Tokens>(row, batch_row), projected);
    }
};

} // namespace ninfer::ops::detail
