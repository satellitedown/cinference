// Modified by satellitedown for Cinference: preload short-width projected inputs before the token loop.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "ops/gdn_input_proj/gdn_projected_conv.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// MaxWidth > 0 bounds the width at compile time: every projected input of the row is loaded up
// front, so the per-token loads overlap instead of serializing one global-load latency per token.
template <int Channels, int QueryRows, int KeyRows, int ValueRows, int MaxWidth, class Publish>
__global__ void gdn_projected_conv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ conv_weight,
    const __nv_bfloat16* __restrict__ state_read, const std::int32_t* __restrict__ valid_columns,
    const std::int32_t* __restrict__ initial_state_slots, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, std::int32_t width,
    Publish publish) {
    static_assert(Channels == QueryRows + KeyRows + ValueRows);
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= Channels) { return; }
    const std::int32_t batch = static_cast<std::int32_t>(blockIdx.y);

    std::int32_t valid                 = valid_columns == nullptr ? width : valid_columns[batch];
    valid                              = valid < 0 ? 0 : (valid > width ? width : valid);
    constexpr std::int64_t slot_stride = static_cast<std::int64_t>(Channels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_state_slots[batch]) * slot_stride;
    float s0       = __bfloat162float(state_read[initial_base + row]);
    float s1       = __bfloat162float(state_read[initial_base + Channels + row]);
    float s2       = __bfloat162float(state_read[initial_base + 2LL * Channels + row]);
    const float w0 = __bfloat162float(conv_weight[row]);
    const float w1 = __bfloat162float(conv_weight[Channels + row]);
    const float w2 = __bfloat162float(conv_weight[2LL * Channels + row]);
    const float w3 = __bfloat162float(conv_weight[3LL * Channels + row]);

    const auto input = [&](std::int32_t token) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        return __bfloat162float(projected[column * Channels + row]);
    };
    const auto step = [&](std::int32_t token, float p) {
        const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
        if (token >= valid) {
            if (row < QueryRows) {
                query[column * QueryRows + row] = __float2bfloat16_rn(0.0F);
            } else if (row < QueryRows + KeyRows) {
                key[column * KeyRows + row - QueryRows] = __float2bfloat16_rn(0.0F);
            } else {
                value[column * ValueRows + row - QueryRows - KeyRows] = __float2bfloat16_rn(0.0F);
            }
            return;
        }

        float conv                 = fmaf(w0, s0, 0.0F);
        conv                       = fmaf(w1, s1, conv);
        conv                       = fmaf(w2, s2, conv);
        conv                       = fmaf(w3, p, conv);
        const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
        if (row < QueryRows) {
            query[column * QueryRows + row] = output;
        } else if (row < QueryRows + KeyRows) {
            key[column * KeyRows + row - QueryRows] = output;
        } else {
            value[column * ValueRows + row - QueryRows - KeyRows] = output;
        }
        publish.publish(token, batch, row, s1, s2, p);
        s0 = s1;
        s1 = s2;
        s2 = p;
    };

    if constexpr (MaxWidth > 0) {
        float inputs[MaxWidth];
#pragma unroll
        for (std::int32_t token = 0; token < MaxWidth; ++token) {
            inputs[token] = token < valid ? input(token) : 0.0F;
        }
#pragma unroll
        for (std::int32_t token = 0; token < MaxWidth; ++token) {
            if (token >= width) { break; }
            step(token, inputs[token]);
        }
    } else {
        for (std::int32_t token = 0; token < width; ++token) {
            step(token, token < valid ? input(token) : 0.0F);
        }
    }
}

template <int Channels, int QueryRows, int KeyRows, int ValueRows, class Publish>
void launch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
            const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
            Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    const std::int32_t width = projected.ne[1];
    const std::int32_t batch = projected.ne[2];
    const auto run           = [&]<int MaxWidth, int Threads>() {
        const dim3 grid((Channels + Threads - 1) / Threads, static_cast<unsigned>(batch));
        gdn_projected_conv_kernel<Channels, QueryRows, KeyRows, ValueRows, MaxWidth>
            <<<grid, Threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projected.data),
                static_cast<const __nv_bfloat16*>(conv_weight.data),
                static_cast<const __nv_bfloat16*>(state_read.data),
                valid_columns.data == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(valid_columns.data),
                static_cast<const std::int32_t*>(initial_state_slots.data),
                static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
                static_cast<__nv_bfloat16*>(value.data), width, publish);
    };
    // Short speculative widths use narrow CTAs so a single row of work spreads over the SMs.
    if (width <= 4) {
        run.template operator()<4, 64>();
    } else if (width <= 16) {
        run.template operator()<16, 64>();
    } else {
        run.template operator()<0, 256>();
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class Publish>
void dispatch(const Tensor& projected, const Tensor& conv_weight, const Tensor& state_read,
              const Tensor& valid_columns, const Tensor& initial_state_slots, Tensor& query,
              Tensor& key, Tensor& value, Publish publish, cudaStream_t stream) {
    if (projected.ne[0] == 10240 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 6144) {
        launch<10240, 2048, 2048, 6144>(projected, conv_weight, state_read, valid_columns,
                                        initial_state_slots, query, key, value, publish, stream);
        return;
    }
    if (projected.ne[0] == 8192 && query.ne[0] == 2048 && key.ne[0] == 2048 &&
        value.ne[0] == 4096) {
        launch<8192, 2048, 2048, 4096>(projected, conv_weight, state_read, valid_columns,
                                       initial_state_slots, query, key, value, publish, stream);
        return;
    }
    throw std::invalid_argument("GDN projected-conv received an unregistered geometry");
}

} // namespace

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream) {
    dispatch(projected, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value,
             SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                    static_cast<const std::int32_t*>(snapshot_base_slots.data),
                                    projected.ne[0]},
             stream);
}

void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& query, Tensor& key,
                                      Tensor& value, cudaStream_t stream) {
    dispatch(conv_record, conv_weight, conv_states, valid_columns, initial_state_slots, query, key,
             value, NoHistoryPublish{}, stream);
}

} // namespace ninfer::ops::detail
