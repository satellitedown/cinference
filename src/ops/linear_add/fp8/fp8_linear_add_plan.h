// Modified by satellitedown for Cinference: dispatch verify-width FP8 LinearAdd to the K-split MMA.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kFp8LinearAddChunkTokens = 24;
// A16 chunks from this width use the Tensor Core K-split mainloop; narrower chunks stay on the
// SIMT contraction, whose per-row activation reuse is cheaper below one MMA token tile.
inline constexpr std::int32_t kFp8LinearAddFirstMmaTokens = 9;

[[nodiscard]] std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                  std::int32_t input_rows,
                                                                  LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void fp8_linear_add_mma_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                       cudaStream_t stream);
void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream);

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
