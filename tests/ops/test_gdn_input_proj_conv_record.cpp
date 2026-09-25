// Modified by satellitedown for Cinference: verify-tree convolution taps against root-path chains.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "core/weight.h"
#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"
#include "ops/verify_tree_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t source_slots      = 8;
    const std::int32_t slots             = aggregate_columns + source_slots;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before = make_bf16_bits(
        static_cast<std::size_t>(channels) * 3 * source_slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::uint16_t> snapshot_before(static_cast<std::size_t>(channels) * 3 * slots);
    std::copy(state_before.begin(), state_before.end(),
              snapshot_before.begin() + static_cast<std::size_t>(channels) * 3 * aggregate_columns);
    std::vector<std::int32_t> snapshot_initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        // Read-only selectors may repeat; source capacity does not depend on W.
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 7 ? 7 : (batch_row * 3 + 7) % source_slots;
        snapshot_initial_slots[static_cast<std::size_t>(batch_row)] =
            aggregate_columns + initial_slots[static_cast<std::size_t>(batch_row)];
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(snapshot_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial          = to_device(initial_slots);
    DeviceBuffer device_snapshot_initial = to_device(snapshot_initial_slots);
    DeviceBuffer device_snapshot         = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.p, DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.p, DType::BF16, {channels, 3, slots});
    Tensor record_state_view(record_state.p, DType::BF16, {channels, 3, source_slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_initial(device_snapshot_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.p, DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.data(), DType::BF16, {channels, width, batch});

    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    const auto launch = [&] {
        snapshot_launch(x, conv_weight, snapshot_state_view, valid, snapshot_initial, snapshot_base,
                        snapshot_q, snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace,
                        stream);
        record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                      record_k, record_v, record_z_view, record_workspace, stream);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto activation_bits = bf16_bits(activation);
    if (width == 2 || width == 9 || width == 16) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // The same graph must consume changed inputs at the captured addresses.
        activation_bits = bf16_bits(make_bf16_activation(hidden, aggregate_columns, seed + 19));
        CUDA_CHECK(cudaMemcpyAsync(device_x.p, activation_bits.data(),
                                   activation_bits.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures = 0;
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, snapshot_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels, width,
                                   batch, valid_columns, snapshot_bases);
    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += verify_preserved(std::string(label) + " x", device_x, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures += verify_preserved(std::string(label) + " initial", device_initial, initial_slots);
    if (!dense) {
        failures += verify_preserved(std::string(label) + " valid", device_valid, valid_columns);
    }
    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

std::vector<std::int32_t> ragged(std::int32_t width, std::int32_t batch) {
    std::vector<std::int32_t> valid(batch);
    for (int b = 0; b < batch; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
    return valid;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, 4096, kHidden, 1401U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, 12288, kHidden, 1403U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q4/Q5 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, qk.view(), value_z.view(), conv, state,
                                                  valid_columns, initial, snapshot_base, q, k, v, z,
                                                  workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv, state,
                                                valid_columns, initial, record, q, k, v, z,
                                                workspace, stream);
            },
            seed);
    };
    for (int width = 2; width <= 16; ++width) {
        failures += run(width, 1, {}, 1400U + width);
        failures += run(width, 8, ragged(width, 8), 1450U + width);
    }
    failures += run(5, 3, {5, 3, 1}, 1491U);
    failures += run(4, 4, {4, 3, 2, 1}, 1492U);
    failures += qk.verify_preserved("Q4 record qk weight");
    failures += value_z.verify_preserved("Q5 record value/z weight");
    return failures;
}

int run_q8() {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, 12288, kHidden, 1501U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q8 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, workspace,
                                                  stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, workspace, stream);
            },
            seed);
    };
    failures += run(2, 1, {1}, 1511U);
    failures += run(16, 1, {}, 1521U);
    failures += run(16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531U);
    failures += parent.verify_preserved("Q8 record parent weight");
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1601U, options));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         ops::LinearPolicy policy, std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, policy,
                                                  workspace, stream);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, Tensor{}, record, q, k, v, z, policy,
                                                workspace, stream);
            },
            seed);
    };
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA4}) {
        for (int width = 2; width <= 16; ++width) {
            failures += run(width, 1, {}, policy, 1600U + width);
            failures += run(width, 8, ragged(width, 8), policy, 1650U + width);
        }
    }
    failures += parent.verify_preserved("NVFP4 record parent weight");
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t width, std::int32_t batch,
                 std::vector<std::int32_t> valid, ops::LinearPolicy policy, std::uint32_t seed) {
    const std::size_t snapshot_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, 16384, 5120, policy, batch, width, width);
    const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16, 16384, 5120, policy, batch, width, width);
    return run_case(
        "FP8 policy=" + std::to_string(static_cast<int>(policy)) + " B=" + std::to_string(batch) +
            " W=" + std::to_string(width),
        5120, 6144, 6144, width, batch, std::move(valid), snapshot_bytes, record_bytes,
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns, initial,
                                              snapshot_base, q, k, v, z, policy, workspace, stream);
        },
        [&](const Tensor& x, const Tensor& conv, const Tensor& state, const Tensor& valid_columns,
            const Tensor& initial, Tensor& record, Tensor& q, Tensor& k, Tensor& v, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
            ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns, initial,
                                            Tensor{}, record, q, k, v, z, policy, workspace,
                                            stream);
        },
        seed);
}

// Every column of a verify tree must convolve exactly like the last column of the chain over its
// root path (same projection, same initial history), and record its own projected input.
int run_fp8_tree_case(DevicePackedWeight& parent, std::int32_t batch,
                      const std::vector<std::int32_t>& valid, ops::LinearPolicy policy,
                      std::uint32_t seed) {
    constexpr std::int32_t kHidden          = 5120;
    constexpr std::int32_t kWidth           = 16;
    constexpr std::int32_t kChannels        = 10240;
    constexpr std::int32_t kZRows           = 6144;
    constexpr std::int32_t kSlots           = 8;
    const std::int32_t columns              = kWidth * batch;
    const std::vector<float> activation     = make_bf16_activation(kHidden, columns, seed);
    const std::vector<std::uint16_t> x_bits = bf16_bits(activation);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, seed + 2, -0.05F, 0.05F);

    std::vector<std::int32_t> initial(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> parents(static_cast<std::size_t>(columns), 0);
    std::vector<std::vector<std::int32_t>> trees(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial[static_cast<std::size_t>(row)] = (row * 3 + 5) % kSlots;
        trees[static_cast<std::size_t>(row)] =
            random_verify_tree(valid[static_cast<std::size_t>(row)], seed + 31U * row, 0.4);
        std::copy(trees[static_cast<std::size_t>(row)].begin(),
                  trees[static_cast<std::size_t>(row)].end(),
                  parents.begin() + static_cast<std::ptrdiff_t>(row) * kWidth);
    }

    const auto launch = [&](const std::vector<std::uint16_t>& x_host, std::int32_t rows,
                            const std::vector<std::int32_t>& valid_host,
                            const std::vector<std::int32_t>& initial_host,
                            const std::vector<std::int32_t>* parents_host,
                            std::vector<std::uint16_t>& qkv_out,
                            std::vector<std::uint16_t>& record_out) {
        const std::int32_t launch_columns = kWidth * rows;
        DeviceBuffer device_x             = to_device(x_host);
        DeviceBuffer device_conv          = to_device(conv_weight_bits);
        DeviceBuffer device_state         = to_device(state_bits);
        DeviceBuffer device_valid         = to_device(valid_host);
        DeviceBuffer device_initial       = to_device(initial_host);
        DeviceBuffer device_parents;
        if (parents_host != nullptr) { device_parents = to_device(*parents_host); }
        DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * launch_columns * 2);
        DeviceBuffer k(static_cast<std::size_t>(kKeyRows) * launch_columns * 2);
        DeviceBuffer v(static_cast<std::size_t>(6144) * launch_columns * 2);
        DeviceBuffer z(static_cast<std::size_t>(kZRows) * launch_columns * 2);
        DeviceBuffer record(static_cast<std::size_t>(kChannels) * launch_columns * 2);
        const std::size_t workspace_bytes =
            ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16, 16384, kHidden, policy, rows, kWidth, kWidth);
        WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
        Tensor x(device_x.p, DType::BF16, {kHidden, kWidth, rows});
        Tensor conv(device_conv.p, DType::BF16, {kChannels, 4});
        Tensor state(device_state.p, DType::BF16, {kChannels, 3, kSlots});
        Tensor valid_tensor(device_valid.p, DType::I32, {rows});
        Tensor initial_tensor(device_initial.p, DType::I32, {rows});
        Tensor parents_tensor;
        if (parents_host != nullptr) {
            parents_tensor = Tensor(device_parents.p, DType::I32, {kWidth, rows});
        }
        Tensor q_view(q.p, DType::BF16, {kQueryRows, kWidth, rows});
        Tensor k_view(k.p, DType::BF16, {kKeyRows, kWidth, rows});
        Tensor v_view(v.p, DType::BF16, {6144, kWidth, rows});
        Tensor z_view(z.p, DType::BF16, {kZRows, kWidth, rows});
        Tensor record_view(record.p, DType::BF16, {kChannels, kWidth, rows});
        ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_tensor, initial_tensor,
                                        parents_tensor, record_view, q_view, k_view, v_view, z_view,
                                        policy, workspace, nullptr);
        cuda_synchronize();
        const auto q_bits = from_device<std::uint16_t>(q, q.bytes / 2);
        const auto k_bits = from_device<std::uint16_t>(k, k.bytes / 2);
        const auto v_bits = from_device<std::uint16_t>(v, v.bytes / 2);
        qkv_out.clear();
        for (std::int32_t column = 0; column < launch_columns; ++column) {
            qkv_out.insert(qkv_out.end(), q_bits.begin() + column * kQueryRows,
                           q_bits.begin() + (column + 1) * kQueryRows);
            qkv_out.insert(qkv_out.end(), k_bits.begin() + column * kKeyRows,
                           k_bits.begin() + (column + 1) * kKeyRows);
            qkv_out.insert(qkv_out.end(), v_bits.begin() + column * 6144,
                           v_bits.begin() + (column + 1) * 6144);
        }
        record_out = from_device<std::uint16_t>(record, record.bytes / 2);
    };

    std::vector<std::uint16_t> tree_qkv;
    std::vector<std::uint16_t> tree_record;
    launch(x_bits, batch, valid, initial, &parents, tree_qkv, tree_record);
    const std::string label    = "FP8 tree policy=" + std::to_string(static_cast<int>(policy)) +
                                 " B=" + std::to_string(batch);
    constexpr std::size_t kQkv = kChannels;
    // The chain reference keeps the physical [16,B] block, so the projection arithmetic matches;
    // only the tested row carries the gathered root path.
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& tree = trees[static_cast<std::size_t>(row)];
        for (std::int32_t node = 0; node < static_cast<std::int32_t>(tree.size()); ++node) {
            const std::vector<std::int32_t> path = verify_tree_path(tree, node);
            std::vector<std::uint16_t> chain_x   = x_bits;
            for (std::size_t j = 0; j < path.size(); ++j) {
                std::copy_n(
                    x_bits.begin() + (static_cast<std::size_t>(row) * kWidth + path[j]) * kHidden,
                    kHidden,
                    chain_x.begin() + (static_cast<std::size_t>(row) * kWidth + j) * kHidden);
            }
            std::vector<std::int32_t> chain_valid      = valid;
            chain_valid[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(path.size());
            std::vector<std::uint16_t> chain_qkv;
            std::vector<std::uint16_t> chain_record;
            launch(chain_x, batch, chain_valid, initial, nullptr, chain_qkv, chain_record);
            const std::size_t chain_column =
                static_cast<std::size_t>(row) * kWidth + path.size() - 1;
            const std::size_t tree_column = static_cast<std::size_t>(row) * kWidth + node;
            if (!std::equal(chain_qkv.begin() + chain_column * kQkv,
                            chain_qkv.begin() + (chain_column + 1) * kQkv,
                            tree_qkv.begin() + tree_column * kQkv) ||
                !std::equal(chain_record.begin() + chain_column * kChannels,
                            chain_record.begin() + (chain_column + 1) * kChannels,
                            tree_record.begin() + tree_column * kChannels)) {
                std::cerr << label << ": node " << node << " of row " << row
                          << " differs from its root-path chain\n";
                return 1;
            }
        }
    }
    return 0;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, kRows, kHidden, 1701U));
    int failures = 0;
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        for (int width = 2; width <= 16; ++width) {
            failures += run_fp8_case(parent, width, 1, {}, policy, 1700U + width);
            failures += run_fp8_case(parent, width, 8, ragged(width, 8), policy, 1750U + width);
        }
        for (int batch : {2, 3, 4}) {
            failures += run_fp8_case(parent, 4, batch, ragged(4, batch), policy, 1810U + batch);
        }
    }
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        failures += run_fp8_tree_case(parent, 1, {16}, policy, 1830U);
        failures += run_fp8_tree_case(parent, 3, {16, 9, 16}, policy, 1840U);
    }
    failures += parent.verify_preserved("FP8 record parent weight");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_q8();
    failures += run_nvfp4();
    failures += run_fp8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
