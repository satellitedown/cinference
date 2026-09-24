#pragma once

// ninfer::ops - per-head query/key RMSNorm followed by RoPE.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: qk_rmsnorm_rope
 *
 * Math / indexing:
 *   q_out = RMSNorm(q; q_norm_weight, eps, unit_offset)   per head row, as ops::rmsnorm defines it;
 *   k_out = RMSNorm(k; k_norm_weight, eps, unit_offset)   per head row;
 *   then ops::rope(positions, rotary_dim, theta, q_out, k_out) in place.
 *
 * Logical shapes / supported domain:
 *   q and q_out are BF16 [D,Hq,T], k and k_out BF16 [D,Hk,T], all contiguous and 4-byte aligned;
 *   the norm weights are contiguous BF16 [D]; positions, rotary_dim and theta are any argument set
 *   ops::rope accepts for these heads. Inputs, outputs, weights and positions must not overlap,
 *   except that the read-only weights may overlap each other.
 *
 * Numeric:
 *   The normalized heads are BF16 seams exactly as ops::rmsnorm rounds them. Every route produces
 *   the bits of rmsnorm on q, rmsnorm on k and rope called in sequence, so those Ops' oracles and
 *   criteria apply unchanged.
 *
 * Effects:
 *   Writes q_out and k_out completely; q, k, weights and positions are unchanged. There is no
 *   workspace or persistent state side effect.
 */
void qk_rmsnorm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                     const Tensor& k_norm_weight, float eps, bool unit_offset,
                     const Tensor& positions, int rotary_dim, float theta, Tensor& q_out,
                     Tensor& k_out, cudaStream_t stream);

} // namespace ninfer::ops
