// Modified by satellitedown for Cinference: declare the record-route convolution launcher.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                 std::int32_t min_tokens,
                                                                 std::int32_t max_tokens);

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream);

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream);

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream);

// The A8 record route of a single width-16 block convolves each channel inside the projection's
// epilogue: the projection still lands in conv_record, and query/key/value are complete on return.
inline constexpr std::int32_t kFp8GdnRecordConvWidth = 16;

void fp8_gdn_record_conv_a8_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   const Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                   Tensor& key, Tensor& value, Tensor& z, Fp8A8Workspace workspace,
                                   cudaStream_t stream);

// Exact contraction mechanisms shared by G1/G2/G3. Semantic Ops own their route frontier and
// call one of these launchers after resolving their complete-form plan.
void fp8_gdn_input_a16_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

void fp8_gdn_input_a8_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               WorkspaceArena& workspace, cudaStream_t stream);

void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
