#include "core/weight.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_swiglu_ffn.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kIntermediate = 17408;
constexpr float kEps                 = 1.0e-6F;

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

struct Profile {
    std::string label;
    Weight gate_up;
    Weight down;
};

struct Case {
    std::int32_t tokens;
    ops::LinearPolicy gate_up_policy;
    ops::LinearPolicy down_policy;
    bool unit_offset;
};

// The Op's contract is the exact result of rmsnorm, linear_swiglu and linear_add in sequence, so
// each route is compared bit for bit with that composition run through the component Ops.
int run_case(const Profile& profile, const Case& c, const DeviceBuffer& norm_weight_storage,
             const std::vector<std::uint16_t>& initial_residual, bool capture) {
    const std::string label = profile.label + " T=" + std::to_string(c.tokens) +
                              " policies=" + std::to_string(static_cast<int>(c.gate_up_policy)) +
                              "/" + std::to_string(static_cast<int>(c.down_policy)) +
                              (c.unit_offset ? " offset" : " plain");
    const std::size_t residual_words = static_cast<std::size_t>(kHidden) * c.tokens;
    const std::size_t residual_bytes = residual_words * sizeof(std::uint16_t);
    Tensor norm_weight(norm_weight_storage.p, DType::BF16, {kHidden});

    DeviceBuffer expected_storage(residual_bytes);
    CUDA_CHECK(cudaMemcpy(expected_storage.p, initial_residual.data(), residual_bytes,
                          cudaMemcpyHostToDevice));
    Tensor expected(expected_storage.p, DType::BF16, {kHidden, c.tokens});
    {
        DeviceBuffer hidden_storage(residual_bytes);
        DeviceBuffer activation_storage(static_cast<std::size_t>(kIntermediate) * c.tokens *
                                        sizeof(std::uint16_t));
        Tensor hidden(hidden_storage.p, DType::BF16, {kHidden, c.tokens});
        Tensor activation(activation_storage.p, DType::BF16, {kIntermediate, c.tokens});
        WorkspaceArena swiglu_workspace(std::max<std::size_t>(
            256, ops::linear_swiglu_workspace_capacity_bytes(
                     profile.gate_up.qtype, profile.gate_up.n, profile.gate_up.k, c.gate_up_policy,
                     c.tokens, c.tokens)));
        WorkspaceArena add_workspace(std::max<std::size_t>(
            256, ops::linear_add_workspace_capacity_bytes(profile.down.qtype, profile.down.n,
                                                          profile.down.k, c.down_policy, c.tokens,
                                                          c.tokens)));
        ops::rmsnorm(expected, norm_weight, kEps, c.unit_offset, hidden, nullptr);
        ops::linear_swiglu(hidden, profile.gate_up, activation, c.gate_up_policy, swiglu_workspace,
                           nullptr);
        ops::linear_add(activation, profile.down, expected, c.down_policy, add_workspace, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    GuardedDeviceBuffer actual_storage(residual_bytes);
    actual_storage.copy_from_host(initial_residual.data(), residual_bytes);
    Tensor actual(actual_storage.data(), DType::BF16, {kHidden, c.tokens});
    const std::size_t capacity = ops::rmsnorm_swiglu_ffn_workspace_capacity_bytes(
        profile.gate_up, c.gate_up_policy, profile.down, c.down_policy, c.tokens, c.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    const auto launch = [&](cudaStream_t stream) {
        ops::rmsnorm_swiglu_ffn(actual, norm_weight, kEps, c.unit_offset, profile.gate_up,
                                c.gate_up_policy, profile.down, c.down_policy, workspace, stream);
    };
    launch(nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failures = 0;
    if (workspace.used() != 0 || workspace.peak_used() != capacity) {
        std::cerr << label << ": workspace query/execution mismatch\n";
        ++failures;
    }
    const auto compare = [&](const std::string& what) {
        std::vector<std::uint16_t> expected_bits(residual_words);
        std::vector<std::uint16_t> actual_bits(residual_words);
        CUDA_CHECK(cudaMemcpy(expected_bits.data(), expected_storage.p, residual_bytes,
                              cudaMemcpyDeviceToHost));
        actual_storage.copy_to_host(actual_bits.data(), residual_bytes);
        const auto mismatch =
            std::mismatch(actual_bits.begin(), actual_bits.end(), expected_bits.begin());
        if (mismatch.first != actual_bits.end()) {
            const auto index = static_cast<std::size_t>(mismatch.first - actual_bits.begin());
            std::cerr << label << what << ": residual differs from the composition at row "
                      << index % kHidden << " token " << index / kHidden << "\n";
            return 1;
        }
        return 0;
    };
    failures += compare("");

    if (capture) {
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CUDA_CHECK(cudaMemcpyAsync(actual_storage.data(), initial_residual.data(),
                                       residual_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaStreamDestroy(stream));
        failures += compare(" (graph replay)");
    }
    failures += actual_storage.verify_guards(label);
    return failures;
}

int run_profile(const Profile& profile, const std::vector<Case>& cases, std::uint32_t seed) {
    const std::int32_t max_tokens =
        std::max_element(cases.begin(), cases.end(), [](const Case& a, const Case& b) {
            return a.tokens < b.tokens;
        })->tokens;
    const std::vector<std::uint16_t> residual =
        make_bf16(static_cast<std::size_t>(kHidden) * max_tokens, seed, 4.0F);
    const std::vector<std::uint16_t> norm_weight = make_bf16(kHidden, seed + 1U, 0.5F);
    DeviceBuffer norm_weight_storage(norm_weight.size() * sizeof(std::uint16_t));
    CUDA_CHECK(cudaMemcpy(norm_weight_storage.p, norm_weight.data(),
                          norm_weight.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    int failures = 0;
    for (const Case& c : cases) {
        failures += run_case(profile, c, norm_weight_storage, residual,
                             c.tokens == 16 && c.gate_up_policy == ops::LinearPolicy::AllowA4);
        // Every single-width requirement fits inside the capacity reported for an interval
        // containing it.
        const std::size_t interval = ops::rmsnorm_swiglu_ffn_workspace_capacity_bytes(
            profile.gate_up, c.gate_up_policy, profile.down, c.down_policy, 1, c.tokens);
        const std::size_t single = ops::rmsnorm_swiglu_ffn_workspace_capacity_bytes(
            profile.gate_up, c.gate_up_policy, profile.down, c.down_policy, c.tokens, c.tokens);
        if (single > interval) {
            std::cerr << profile.label << " T=" << c.tokens
                      << ": interval capacity is smaller than a member width's\n";
            ++failures;
        }
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
    int failures = 0;
    {
        quantized_weight::PatternedWeightOptions gate_up_options;
        gate_up_options.weight_scale_divisor = 0.125F;
        gate_up_options.input_scale_divisor  = 3.5F;
        quantized_weight::PatternedWeightOptions down_options;
        down_options.weight_scale_divisor = 0.25F;
        down_options.input_scale_divisor  = 5.0F;
        const auto gate_up                = quantized_weight::make_patterned_weight(
            QType::NVFP4, 2 * kIntermediate, kHidden, 911U, gate_up_options);
        const auto down = quantized_weight::make_patterned_weight(
            QType::NVFP4, kHidden, kIntermediate, 913U, down_options);
        DeviceBuffer gate_up_storage(gate_up.payload.size());
        DeviceBuffer down_storage(down.payload.size());
        CUDA_CHECK(cudaMemcpy(gate_up_storage.p, gate_up.payload.data(), gate_up.payload.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(down_storage.p, down.payload.data(), down.payload.size(),
                              cudaMemcpyHostToDevice));
        const Profile profile{"NVFP4", gate_up.device_weight(gate_up_storage.p),
                              down.device_weight(down_storage.p)};
        std::vector<Case> cases;
        for (const std::int32_t tokens : {1, 4, 5, 7, 8, 11, 16, 17, 40}) {
            cases.push_back({tokens, LinearPolicy::AllowA4, LinearPolicy::AllowA4, true});
        }
        cases.push_back({16, LinearPolicy::AllowA4, LinearPolicy::AllowA4, false});
        cases.push_back({12, LinearPolicy::AllowA4, LinearPolicy::A16Only, true});
        cases.push_back({12, LinearPolicy::A16Only, LinearPolicy::AllowA4, true});
        cases.push_back({16, LinearPolicy::A16Only, LinearPolicy::A16Only, true});
        failures += run_profile(profile, cases, 915U);
    }
    {
        const auto gate_up = quantized_weight::make_patterned_weight(
            QType::FP8_E4M3FN_ROW_BF16, 2 * kIntermediate, kHidden, 921U);
        const auto down = quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,
                                                                  kHidden, kIntermediate, 923U);
        DeviceBuffer gate_up_storage(gate_up.payload.size());
        DeviceBuffer down_storage(down.payload.size());
        CUDA_CHECK(cudaMemcpy(gate_up_storage.p, gate_up.payload.data(), gate_up.payload.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(down_storage.p, down.payload.data(), down.payload.size(),
                              cudaMemcpyHostToDevice));
        const Profile profile{"FP8", gate_up.device_weight(gate_up_storage.p),
                              down.device_weight(down_storage.p)};
        std::vector<Case> cases;
        for (const std::int32_t tokens : {4, 16, 17}) {
            cases.push_back({tokens, LinearPolicy::AllowA8, LinearPolicy::AllowA8, true});
        }
        failures += run_profile(profile, cases, 925U);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " rmsnorm_swiglu_ffn\n";
    return failures == 0 ? 0 : 1;
}
