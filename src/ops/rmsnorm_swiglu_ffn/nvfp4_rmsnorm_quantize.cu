#include "ops/rmsnorm_swiglu_ffn/nvfp4_rmsnorm_quantize.h"

#include "core/device.h"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// A 5120-wide row holds 2560 pairs and 2560 code bytes, so pair p is code byte p of the row and
// sits in group p / 8. The launch below gives every lane a pair on every step, with the eight pairs
// of a group on eight consecutive lanes, so the group exchanges its maximum over those lanes.
// Quantizing costs three IEEE divisions per lane and step, so five CTAs share each row, two of the
// ten steps apiece (T=16: 1, 2, 5 and 10 slices measured 2.94, 2.24, 1.82 and 2.14 us).
struct RmsNvfp4Output {
    static constexpr int kRowSlices = 5;

    std::uint8_t* codes;
    std::uint8_t* scales;
    float input_scale_divisor;

    __device__ __forceinline__ void store(std::int64_t row_base, int pair,
                                          __nv_bfloat162 value) const {
        const float2 values[1]{__bfloat1622float2(value)};
        const Nvfp4LaneGroupCodes quantized = quantize_nvfp4_lanes<8>(values, input_scale_divisor);
        const std::int64_t byte             = row_base + pair;
        codes[byte]                         = static_cast<std::uint8_t>(quantized.codes);
        if ((pair & 7) == 0) { scales[byte / 8] = quantized.scale; }
    }
};

// The instantiation ops::rmsnorm launches for aligned 5120-wide rows, with the quantizing output.
template <RmsEpilogue Epilogue>
void launch(const Tensor& x, const Tensor& norm_weight, float eps, RmsNvfp4Output output,
            cudaStream_t stream) {
    constexpr int kBlock = 256;
    static_assert(kNvfp4RmsNormQuantizeWidth / 2 == kBlock * 10);
    rmsnorm_cta_bf16x2_kernel<Epilogue, kBlock, 10, true, kNvfp4RmsNormQuantizeWidth,
                              RmsNvfp4Output>
        <<<static_cast<unsigned>(x.ne[1] * RmsNvfp4Output::kRowSlices), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(x.data),
            static_cast<const __nv_bfloat162*>(norm_weight.data), nullptr, output,
            kNvfp4RmsNormQuantizeWidth, x.ne[1], eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_rmsnorm_quantize_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                   bool unit_offset, float input_scale_divisor,
                                   Nvfp4W4a4Workspace out, cudaStream_t stream) {
    if (x.ne[0] != kNvfp4RmsNormQuantizeWidth || x.ne[1] <= 0 || out.codes == nullptr ||
        out.scales == nullptr) {
        throw std::invalid_argument("nvfp4 rmsnorm quantize: unsupported row width or workspace");
    }
    const RmsNvfp4Output output{out.codes, out.scales, input_scale_divisor};
    if (unit_offset) {
        launch<RmsEpilogue::Offset>(x, norm_weight, eps, output, stream);
    } else {
        launch<RmsEpilogue::Plain>(x, norm_weight, eps, output, stream);
    }
}

} // namespace ninfer::ops::detail
