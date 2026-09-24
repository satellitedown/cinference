#include "ninfer/ops/qk_rmsnorm_rope.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "core/device.h"

#include "ops/op_tester.h"

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

constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKeyHeads   = 4;
constexpr float kEps               = 1.0e-6F;

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

DeviceBuffer upload(const void* data, std::size_t bytes) {
    DeviceBuffer buffer(bytes);
    buffer.copy_from_host(data, bytes);
    return buffer;
}

// The Op's contract is the exact result of rmsnorm on q, rmsnorm on k and rope in sequence, so
// every route is compared bit for bit with that composition run through the component Ops.
int run_case(std::int32_t tokens, std::int32_t axes, bool unit_offset, std::uint32_t seed) {
    const std::string label   = "qk_rmsnorm_rope T=" + std::to_string(tokens) +
                                " axes=" + std::to_string(axes) +
                                (unit_offset ? " offset" : " plain");
    const std::size_t q_words = static_cast<std::size_t>(kHeadDim) * kQueryHeads * tokens;
    const std::size_t k_words = static_cast<std::size_t>(kHeadDim) * kKeyHeads * tokens;
    const auto q_host         = make_bf16(q_words, seed, 3.0F);
    const auto k_host         = make_bf16(k_words, seed + 1U, 3.0F);
    const auto q_weight_host  = make_bf16(kHeadDim, seed + 2U, 0.5F);
    const auto k_weight_host  = make_bf16(kHeadDim, seed + 3U, 0.5F);
    std::vector<std::int32_t> positions_host(static_cast<std::size_t>(tokens) * axes);
    for (std::size_t index = 0; index < positions_host.size(); ++index) {
        positions_host[index] = static_cast<std::int32_t>((index * 7919U + seed) % 262144U);
    }

    const DeviceBuffer q_storage = upload(q_host.data(), q_words * sizeof(std::uint16_t));
    const DeviceBuffer k_storage = upload(k_host.data(), k_words * sizeof(std::uint16_t));
    const DeviceBuffer q_weight_storage =
        upload(q_weight_host.data(), kHeadDim * sizeof(std::uint16_t));
    const DeviceBuffer k_weight_storage =
        upload(k_weight_host.data(), kHeadDim * sizeof(std::uint16_t));
    const DeviceBuffer positions_storage =
        upload(positions_host.data(), positions_host.size() * sizeof(std::int32_t));
    const Tensor q(q_storage.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    const Tensor k(k_storage.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    const Tensor q_weight(q_weight_storage.p, DType::BF16, {kHeadDim});
    const Tensor k_weight(k_weight_storage.p, DType::BF16, {kHeadDim});
    const Tensor positions = axes == 1 ? Tensor(positions_storage.p, DType::I32, {tokens})
                                       : Tensor(positions_storage.p, DType::I32, {tokens, axes});

    DeviceBuffer expected_q_storage(q_words * sizeof(std::uint16_t));
    DeviceBuffer expected_k_storage(k_words * sizeof(std::uint16_t));
    Tensor expected_q(expected_q_storage.p, DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor expected_k(expected_k_storage.p, DType::BF16, {kHeadDim, kKeyHeads, tokens});
    ops::rmsnorm(q, q_weight, kEps, unit_offset, expected_q, nullptr);
    ops::rmsnorm(k, k_weight, kEps, unit_offset, expected_k, nullptr);
    ops::rope(positions, 64, 1.0e7F, expected_q, expected_k, nullptr);

    GuardedDeviceBuffer actual_q_storage(q_words * sizeof(std::uint16_t));
    GuardedDeviceBuffer actual_k_storage(k_words * sizeof(std::uint16_t));
    Tensor actual_q(actual_q_storage.data(), DType::BF16, {kHeadDim, kQueryHeads, tokens});
    Tensor actual_k(actual_k_storage.data(), DType::BF16, {kHeadDim, kKeyHeads, tokens});
    ops::qk_rmsnorm_rope(q, k, q_weight, k_weight, kEps, unit_offset, positions, 64, 1.0e7F,
                         actual_q, actual_k, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    int failures       = 0;
    const auto compare = [&](const char* what, const DeviceBuffer& expected,
                             const GuardedDeviceBuffer& actual, std::size_t words) {
        std::vector<std::uint16_t> expected_bits(words);
        std::vector<std::uint16_t> actual_bits(words);
        expected.copy_to_host(expected_bits.data(), words * sizeof(std::uint16_t));
        actual.copy_to_host(actual_bits.data(), words * sizeof(std::uint16_t));
        const auto mismatch =
            std::mismatch(actual_bits.begin(), actual_bits.end(), expected_bits.begin());
        if (mismatch.first != actual_bits.end()) {
            const auto index  = static_cast<std::size_t>(mismatch.first - actual_bits.begin());
            std::size_t count = 0;
            for (std::size_t i = 0; i < words; ++i) count += actual_bits[i] != expected_bits[i];
            std::cerr << label << ": " << what << " differs from the composition at element "
                      << index << " (dim " << index % kHeadDim << ", expected "
                      << bf16_to_f32(expected_bits[index]) << ", actual "
                      << bf16_to_f32(actual_bits[index]) << ", " << count << " elements)\n";
            ++failures;
        }
    };
    compare("query", expected_q_storage, actual_q_storage, q_words);
    compare("key", expected_k_storage, actual_k_storage, k_words);
    failures += actual_q_storage.verify_guards(label + " query");
    failures += actual_k_storage.verify_guards(label + " key");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures       = 0;
    std::uint32_t seed = 31U;
    for (const std::int32_t tokens : {1, 3, 16, 17, 64, 1000}) {
        for (const std::int32_t axes : {1, 3}) {
            for (const bool unit_offset : {true, false}) {
                failures += run_case(tokens, axes, unit_offset, seed);
                seed += 11U;
            }
        }
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qk_rmsnorm_rope\n";
    return failures == 0 ? 0 : 1;
}
