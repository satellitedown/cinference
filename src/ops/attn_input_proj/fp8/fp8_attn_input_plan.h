// Modified by satellitedown for Cinference: expose the pre-quantized A8 attention input route.
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

inline constexpr int kFp8AttnInputLastSimtT     = 5;
inline constexpr int kFp8AttnInputLastSmallMmaT = 33;

[[nodiscard]] std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);
void fp8_attn_input_a16_gemm_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream);

// The A8 route of fp8_attn_input_dispatch on an already quantized input (the codes and scales
// launch_fp8_a8_quantize writes for the BF16 input).
void fp8_attn_input_a8_quantized_launch(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                                        Tensor& v, Fp8A8Workspace input, std::int32_t tokens,
                                        cudaStream_t stream);

// Whether fp8_attn_input_dispatch takes its A8 route (quantizing the input) at this width.
[[nodiscard]] bool fp8_attn_input_uses_a8(LinearPolicy policy, std::int32_t tokens);

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
