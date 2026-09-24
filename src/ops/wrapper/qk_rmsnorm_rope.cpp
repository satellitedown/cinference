#include "ninfer/ops/qk_rmsnorm_rope.h"

#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ops/qk_rmsnorm_rope/qk_rmsnorm_rope.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned4(const void* pointer) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & 3U) == 0;
}

bool heads_tensor(const Tensor& tensor, std::int32_t head_dim, std::int32_t heads,
                  std::int32_t tokens) {
    return tensor.dtype == DType::BF16 && tensor.ne[0] == head_dim && tensor.ne[1] == heads &&
           tensor.ne[2] == tokens && tensor.ne[3] == 1 && tensor.is_contiguous() &&
           aligned4(tensor.data);
}

// The registered fused geometry. Other shapes run the component Ops, which own their domains.
bool fused_route(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                 const Tensor& k_norm_weight, const Tensor& positions, int rotary_dim, float theta,
                 const Tensor& q_out, const Tensor& k_out) {
    constexpr std::int32_t kHeadDim = 256;
    const std::int32_t tokens       = q.ne[2];
    const bool weights = q_norm_weight.dtype == DType::BF16 && k_norm_weight.dtype == DType::BF16 &&
                         q_norm_weight.ne[0] == kHeadDim && k_norm_weight.ne[0] == kHeadDim &&
                         q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous() &&
                         aligned4(q_norm_weight.data) && aligned4(k_norm_weight.data);
    const bool rotation = rotary_dim == 64 && theta == 1.0e7F && positions.dtype == DType::I32 &&
                          positions.ne[0] == tokens && positions.is_contiguous() &&
                          (positions.ne[1] == 1 || positions.ne[1] == 3) && positions.ne[2] == 1;
    return tokens > 0 && weights && rotation &&
           heads_tensor(q, kHeadDim, detail::kQkRmsNormRopeQueryHeads, tokens) &&
           heads_tensor(k, kHeadDim, detail::kQkRmsNormRopeKeyHeads, tokens) &&
           heads_tensor(q_out, kHeadDim, detail::kQkRmsNormRopeQueryHeads, tokens) &&
           heads_tensor(k_out, kHeadDim, detail::kQkRmsNormRopeKeyHeads, tokens);
}

} // namespace

void qk_rmsnorm_rope(const Tensor& q, const Tensor& k, const Tensor& q_norm_weight,
                     const Tensor& k_norm_weight, float eps, bool unit_offset,
                     const Tensor& positions, int rotary_dim, float theta, Tensor& q_out,
                     Tensor& k_out, cudaStream_t stream) {
    if (fused_route(q, k, q_norm_weight, k_norm_weight, positions, rotary_dim, theta, q_out,
                    k_out)) {
        detail::qk_rmsnorm_rope_launch(q, k, q_norm_weight, k_norm_weight, eps, unit_offset,
                                       positions, q_out, k_out, stream);
        return;
    }
    rmsnorm(q, q_norm_weight, eps, unit_offset, q_out, stream);
    rmsnorm(k, k_norm_weight, eps, unit_offset, k_out, stream);
    rope(positions, rotary_dim, theta, q_out, k_out, stream);
}

} // namespace ninfer::ops
