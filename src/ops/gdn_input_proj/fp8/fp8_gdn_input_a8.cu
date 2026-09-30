// Modified by satellitedown for Cinference: small-token A8; record convolution with verify trees;
// record projection halves from a caller-produced activation.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N16384K5120;

template <class Schedule, bool FullTokens, class Output>
void launch_mma(const Weight& weight, const Output& output, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    static_assert((Fp8GdnInputOutput::kQkvRows % Schedule::kBlockRows) == 0);
    static_assert((Fp8GdnInputOutput::kZRows % Schedule::kBlockRows) == 0);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue, Output>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void run(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
         std::int32_t tokens, cudaStream_t stream) {
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    if ((tokens % Schedule::kBlockTokens) == 0) {
        launch_mma<Schedule, true>(weight, output, workspace, tokens, stream);
    } else {
        launch_mma<Schedule, false>(weight, output, workspace, tokens, stream);
    }
}

using RecordSchedule = Fp8A8SmallTokenSchedule;
using RecordOutput   = Fp8GdnRecordConvOutput<RecordSchedule::kBlockRows, RecordSchedule::kThreads>;
static_assert(RecordSchedule::kBlockTokens == RecordOutput::kWidth);

RecordOutput record_output(const Tensor& conv_weight, const Tensor& conv_states,
                           const Tensor& valid_columns, const Tensor& initial_slot,
                           const Tensor& tree_parents, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z) {
    return {
        {static_cast<__nv_bfloat16*>(conv_record.data), static_cast<__nv_bfloat16*>(z.data)},
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<const __nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            Fp8GdnInputOutput::kQkvRows,
            Fp8GdnInputOutput::kQueryRows,
            Fp8GdnInputOutput::kKeyRows,
            Fp8GdnInputOutput::kValueRows,
            0,
            RecordOutput::kWidth,
            0,
            NoHistoryPublish{},
        },
        static_cast<const std::int32_t*>(tree_parents.data),
    };
}

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if (x.ne[1] <= kFp8A8SmallTokenLimit) {
        run<Fp8A8SmallTokenSchedule>(weight, qkv, z, workspace, x.ne[1], stream);
    } else {
        run<Fp8A8DefaultSchedule>(weight, qkv, z, workspace, x.ne[1], stream);
    }
}

void fp8_gdn_record_conv_a8_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& tree_parents,
                                   Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                   Tensor& z, Fp8A8Workspace workspace, cudaStream_t stream) {
    if (x.ne[1] != kFp8GdnRecordConvWidth) {
        throw std::invalid_argument("fp8 GDN record convolution: unsupported block width");
    }
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    launch_mma<RecordSchedule, true>(weight,
                                     record_output(conv_weight, conv_states, valid_columns,
                                                   initial_slot, tree_parents, conv_record, query,
                                                   key, value, z),
                                     workspace, x.ne[1], stream);
}

void fp8_gdn_record_conv_a8_rows_launch(Fp8A8Workspace activation, const Weight& weight,
                                        const Tensor& conv_weight, const Tensor& conv_states,
                                        const Tensor& valid_columns, const Tensor& initial_slot,
                                        const Tensor& tree_parents, Tensor& conv_record,
                                        Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                        GdnRecordRows rows, cudaStream_t stream) {
    // Each row tile is exactly the tile of the complete launch (RecordSchedule's TokenFast raster
    // gives one row tile per block at W=16), so the halves together equal it bit for bit.
    constexpr int kQkvTiles = Fp8GdnInputOutput::kQkvRows / RecordSchedule::kBlockRows;
    constexpr int kZTiles   = Fp8GdnInputOutput::kZRows / RecordSchedule::kBlockRows;
    static_assert(kQkvTiles * RecordSchedule::kBlockRows == Fp8GdnInputOutput::kQkvRows);
    static_assert(kZTiles * RecordSchedule::kBlockRows == Fp8GdnInputOutput::kZRows);
    const bool gate           = rows == GdnRecordRows::OutputGate;
    const RecordOutput output = record_output(conv_weight, conv_states, valid_columns, initial_slot,
                                              tree_parents, conv_record, query, key, value, z);
    fp8_mma_kernel<Geometry, RecordSchedule, true, Fp8IdentityEpilogue, RecordOutput,
                   Fp8MmaRowRange><<<gate ? kZTiles : kQkvTiles, RecordSchedule::kThreads,
                                     RecordSchedule::kSharedBytes, stream>>>(
        activation.codes, activation.scales, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), kFp8GdnRecordConvWidth,
        Fp8IdentityEpilogue{}, output, Fp8MmaRowRange{gate ? kQkvTiles : 0});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
