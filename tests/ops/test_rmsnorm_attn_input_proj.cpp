#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_attn_input_proj.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kQueryRows = 6144;
constexpr std::int32_t kKeyRows   = 1024;
constexpr float kEps              = 1.0e-6F;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed, float scale) {
    std::vector<std::uint16_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        std::uint32_t value = seed ^ (static_cast<std::uint32_t>(index) * 0x9e3779b9U);
        value ^= value >> 16;
        value *= 0x7feb352dU;
        value ^= value >> 15;
        result[index] = f32_to_bf16(static_cast<float>(static_cast<int>(value & 0x3ffU) - 512) *
                                    (scale / 512.0F));
    }
    return result;
}

struct Case {
    std::int32_t tokens;
    ops::LinearPolicy policy;
    bool unit_offset;
};

struct Outputs {
    std::array<DeviceBuffer, 4> storage;
    std::array<Tensor, 4> tensors;

    explicit Outputs(std::int32_t tokens) {
        const std::array<std::int32_t, 4> rows{kQueryRows, kQueryRows, kKeyRows, kKeyRows};
        for (int i = 0; i < 4; ++i) {
            storage[i] = DeviceBuffer(static_cast<std::size_t>(rows[i]) * tokens * 2);
            storage[i].fill(0xff);
            tensors[i] = Tensor(storage[i].p, DType::BF16, {rows[i], tokens});
        }
    }
};

// The Op's contract is the exact result of rmsnorm then attn_input_proj, so every output is
// compared bit for bit with that composition run through the component Ops.
int run_case(const Weight& weight, const Case& c, const DeviceBuffer& norm_weight_storage,
             const std::vector<std::uint16_t>& x_bits, bool capture) {
    const std::string label   = "FP8 T=" + std::to_string(c.tokens) +
                                " policy=" + std::to_string(static_cast<int>(c.policy)) +
                                (c.unit_offset ? " offset" : " plain");
    const std::size_t x_bytes = static_cast<std::size_t>(kHidden) * c.tokens * 2;
    DeviceBuffer x_storage(x_bytes);
    CUDA_CHECK(cudaMemcpy(x_storage.p, x_bits.data(), x_bytes, cudaMemcpyHostToDevice));
    Tensor x(x_storage.p, DType::BF16, {kHidden, c.tokens});
    Tensor norm_weight(norm_weight_storage.p, DType::BF16, {kHidden});

    Outputs expected(c.tokens);
    {
        DeviceBuffer hidden_storage(x_bytes);
        Tensor hidden(hidden_storage.p, DType::BF16, {kHidden, c.tokens});
        WorkspaceArena workspace(std::max<std::size_t>(
            256, ops::attn_input_proj_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                               c.policy, c.tokens, c.tokens)));
        ops::rmsnorm(x, norm_weight, kEps, c.unit_offset, hidden, nullptr);
        ops::attn_input_proj(hidden, weight, expected.tensors[0], expected.tensors[1],
                             expected.tensors[2], expected.tensors[3], c.policy, workspace,
                             nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Outputs actual(c.tokens);
    const std::size_t capacity =
        ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(weight, c.policy, c.tokens, c.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    const auto launch = [&](cudaStream_t stream) {
        ops::rmsnorm_attn_input_proj(x, norm_weight, kEps, c.unit_offset, weight, actual.tensors[0],
                                     actual.tensors[1], actual.tensors[2], actual.tensors[3],
                                     c.policy, workspace, stream);
    };
    launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failures = 0;
    if (workspace.used() != 0 || workspace.peak_used() != capacity) {
        std::cerr << label << ": workspace query/execution mismatch\n";
        ++failures;
    }
    const auto compare = [&](const std::string& what) {
        const std::array<const char*, 4> names{"q", "gate", "k", "v"};
        int differing = 0;
        for (int i = 0; i < 4; ++i) {
            const std::size_t words = expected.storage[i].bytes / 2;
            if (from_device<std::uint16_t>(expected.storage[i], words) !=
                from_device<std::uint16_t>(actual.storage[i], words)) {
                std::cerr << label << what << ": " << names[i] << " differs from the composition\n";
                ++differing;
            }
        }
        return differing;
    };
    failures += compare("");

    if (capture) {
        for (auto& buffer : actual.storage) { buffer.fill(0xff); }
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        failures += compare(" (graph replay)");
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    using ops::LinearPolicy;
    const auto parent = quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_ROW_BF16, 2 * kQueryRows + 2 * kKeyRows, kHidden, 931U);
    DeviceBuffer parent_storage(parent.payload.size());
    CUDA_CHECK(cudaMemcpy(parent_storage.p, parent.payload.data(), parent.payload.size(),
                          cudaMemcpyHostToDevice));
    const Weight weight = parent.device_weight(parent_storage.p);

    constexpr std::int32_t kMaxTokens = 40;
    std::vector<std::uint16_t> x_bits =
        make_bf16(static_cast<std::size_t>(kHidden) * kMaxTokens, 933U, 4.0F);
    // A zero row exercises the zero-scale encoding; one large row the saturating side.
    std::fill(x_bits.begin() + kHidden, x_bits.begin() + 2 * kHidden, 0);
    for (std::int32_t i = 0; i < kHidden; ++i) {
        x_bits[2 * kHidden + i] =
            f32_to_bf16((i % 7 == 0 ? 3.0e4F : 1.0F) * (i % 2 ? -1.0F : 1.0F));
    }
    const std::vector<std::uint16_t> norm_bits = make_bf16(kHidden, 935U, 0.5F);
    DeviceBuffer norm_weight_storage(norm_bits.size() * 2);
    CUDA_CHECK(cudaMemcpy(norm_weight_storage.p, norm_bits.data(), norm_bits.size() * 2,
                          cudaMemcpyHostToDevice));

    int failures = 0;
    std::vector<Case> cases;
    for (const std::int32_t tokens : {1, 4, 5, 16, 17, 40}) {
        cases.push_back({tokens, LinearPolicy::AllowA8, true});
    }
    cases.push_back({16, LinearPolicy::AllowA8, false});
    cases.push_back({16, LinearPolicy::A16Only, true});
    for (const Case& c : cases) {
        failures += run_case(weight, c, norm_weight_storage, x_bits,
                             c.tokens == 16 && c.policy == LinearPolicy::AllowA8);
        const std::size_t interval =
            ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(weight, c.policy, 1, c.tokens);
        const std::size_t single = ops::rmsnorm_attn_input_proj_workspace_capacity_bytes(
            weight, c.policy, c.tokens, c.tokens);
        if (single > interval) {
            std::cerr << "T=" << c.tokens << ": interval capacity is smaller than a member's\n";
            ++failures;
        }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " rmsnorm_attn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
