// Modified by satellitedown for Cinference: leave only the MTP head dense FFN here.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

// Sparse MoE blocks and the MTP head's dense FFN on an already normalized hidden. A target dense
// block normalizes and projects through ops::rmsnorm_swiglu_ffn instead.
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
