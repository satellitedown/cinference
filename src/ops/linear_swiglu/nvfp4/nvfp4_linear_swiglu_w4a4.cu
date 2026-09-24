// Modified by satellitedown for Cinference: single-token-tile schedule; quantized activation.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4N34816K5120;
// Column tiles amortize gate/up decode over the complete speculative block.
using M64N128  = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 4, 2, 1>;
using M128N128 = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 4, 2, 1>;
using M96N128  = Nvfp4W4a4MmaSchedule<96, 128, 256, 3, 4, 2, 1>;
// Speculative verify widths fit one 16-token MMA tile; the smaller shared image lets two CTAs share
// an SM, so the 272 row tiles stream in one resident wave instead of a partial second wave.
using M16N128 = Nvfp4W4a4MmaSchedule<16, 128, 256, 1, 8, 2, 2>;

constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <int RowsPerBranch>
struct Nvfp4SwiGluRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = RowsPerBranch;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kIntermediate : 0);
    }
};

union Nvfp4SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

struct Nvfp4SwiGluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ static unsigned combine(unsigned gate_bits, unsigned up_bits) {
        Nvfp4SwiGluBf16Pair gate{gate_bits};
        Nvfp4SwiGluBf16Pair up{up_bits};
        const float2 gate_values = __bfloat1622float2(gate.values);
        const float2 up_values   = __bfloat1622float2(up.values);
        Nvfp4SwiGluBf16Pair result;
        result.values = __floats2bfloat162_rn(silu(gate_values.x) * up_values.x,
                                              silu(gate_values.y) * up_values.y);
        return result.bits;
    }

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        const uint4 values = make_uint4(combine(gate.x, up.x), combine(gate.y, up.y),
                                        combine(gate.z, up.z), combine(gate.w, up.w));
        store_vec(data + static_cast<std::int64_t>(token) * kIntermediate + row, values);
    }
};

// Writes the activation as the W4A4 operand of the projection that consumes it: each group of 16
// intermediate rows of one token is the BF16 activation Nvfp4SwiGluOutput stores, quantized with
// quantize_nvfp4_k16's arithmetic into the row-major code and scale planes the RowMajor quantize
// pass would write. Four lanes share a group, so the CTA's threads cover its whole tile at once.
template <class Schedule>
struct Nvfp4SwiGluQuantizedOutput {
    static constexpr int kPairRows       = Schedule::kBlockN / 2;
    static constexpr int kGroupsPerToken = kPairRows / 16;
    static constexpr int kLanesPerGroup  = 4;
    static constexpr int kValuesPerLane  = 16 / kLanesPerGroup;
    static_assert(Schedule::kBlockM * kGroupsPerToken * kLanesPerGroup == Schedule::kThreads);

    std::uint8_t* codes;
    std::uint8_t* scales;
    float input_scale_divisor;

    __device__ __forceinline__ void finish_tile(const __nv_bfloat16* tile, int stride,
                                                int row_begin, int token_begin, int tokens) const {
        const int task        = static_cast<int>(threadIdx.x);
        const int part        = task % kLanesPerGroup;
        const int group_task  = task / kLanesPerGroup;
        const int token_local = group_task / kGroupsPerToken;
        const int group_local = group_task - token_local * kGroupsPerToken;
        const __nv_bfloat16* gate =
            tile + token_local * stride + group_local * 16 + part * kValuesPerLane;
        const uint2 gate_bits = load_vec<uint2>(gate);
        const uint2 up_bits   = load_vec<uint2>(gate + kPairRows);
        const float2 values[2]{
            bf16x2_bits_to_float2(Nvfp4SwiGluOutput::combine(gate_bits.x, up_bits.x)),
            bf16x2_bits_to_float2(Nvfp4SwiGluOutput::combine(gate_bits.y, up_bits.y)),
        };
        const Nvfp4LaneGroupCodes quantized =
            quantize_nvfp4_lanes<kLanesPerGroup>(values, input_scale_divisor);
        const int token = token_begin + token_local;
        if (token >= tokens) { return; }
        const int group = row_begin / 16 + group_local;
        auto* code_row  = codes + static_cast<std::int64_t>(token) * (kIntermediate / 2);
        store_vec(code_row + group * 8 + part * 2, static_cast<std::uint16_t>(quantized.codes));
        if (part == 0) {
            scales[static_cast<std::int64_t>(token) * (kIntermediate / 16) + group] =
                quantized.scale;
        }
    }
};

template <class Schedule, class Output>
void launch_gemm(const Weight& weight, const Output& output, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    constexpr int kPairRows = Schedule::kBlockN / 2;
    using Rows              = Nvfp4SwiGluRows<kPairRows>;
    const dim3 grid(kIntermediate / kPairRows,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Rows row_policy{};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule, Nvfp4IdentityEpilogue, Output, Rows, true>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            activation, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
            output, row_policy);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
            cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4W4a4Workspace scratch =
        allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::RowMajor, stream);
    launch_gemm<Schedule>(weight, Nvfp4SwiGluOutput{static_cast<__nv_bfloat16*>(out.data)}, scratch,
                          x.ne[1], stream);
}

} // namespace

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    if (x.ne[1] <= M16N128::kBlockM) {
        launch<M16N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M64N128::kBlockM) {
        launch<M64N128>(x, weight, out, workspace, stream);
    } else if (x.ne[1] <= M96N128::kBlockM) {
        launch<M96N128>(x, weight, out, workspace, stream);
    } else {
        launch<M128N128>(x, weight, out, workspace, stream);
    }
}

void nvfp4_linear_swiglu_w4a4_quantized_launch(Nvfp4W4a4Workspace input, const Weight& weight,
                                               std::int32_t tokens, Nvfp4W4a4Workspace activation,
                                               float activation_input_scale_divisor,
                                               cudaStream_t stream) {
    if (tokens <= 0 || tokens > kNvfp4LinearSwiGluQuantizedMaxTokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu quantized activation: unsupported T");
    }
    static_assert(M16N128::kBlockM == kNvfp4LinearSwiGluQuantizedMaxTokens);
    launch_gemm<M16N128>(weight,
                         Nvfp4SwiGluQuantizedOutput<M16N128>{activation.codes, activation.scales,
                                                             activation_input_scale_divisor},
                         input, tokens, stream);
}

} // namespace ninfer::ops::detail
