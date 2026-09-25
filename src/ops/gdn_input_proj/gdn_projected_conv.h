// Modified by satellitedown for Cinference: verify-tree record convolution.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gdn_projected_conv_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                        Tensor& conv_states, const Tensor& valid_columns,
                                        const Tensor& initial_state_slots,
                                        const Tensor& snapshot_base_slots, Tensor& query,
                                        Tensor& key, Tensor& value, cudaStream_t stream);

// tree_parents is empty or I32 [T,B] DFS pre-order verify trees (see gdn_input_proj.h).
void gdn_projected_conv_record_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, const Tensor& tree_parents,
                                      Tensor& query, Tensor& key, Tensor& value,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
