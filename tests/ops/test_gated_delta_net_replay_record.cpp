// Modified by satellitedown for Cinference: exercise the next-layer state hint and verify trees.
// See NOTICE and upstream-provenance.json for upstream attribution.

#include "ninfer/ops/gated_delta_net.h"
#include "core/device.h"

#include "ops/op_tester.h"
#include "ops/verify_tree_test_common.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim    = 128;
constexpr std::uint16_t kBf16Poison = 0xffffU;
constexpr std::uint32_t kFp32Poison = 0xffffffffU;

std::vector<std::uint16_t> make_bf16(std::size_t count, std::uint32_t seed) {
    std::vector<float> values(count);
    fill_uniform(values, seed, -0.08F, 0.08F);
    round_to_bf16(values);
    std::vector<std::uint16_t> bits(count);
    for (std::size_t index = 0; index < count; ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

int verify_equal(const std::string& label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs == rhs) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int run_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
    constexpr std::int32_t kQkHeads = 16;
    const bool dense                = valid_columns.empty();
    if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
    const std::int32_t columns       = width * batch;
    const std::int32_t slots         = 8;
    const std::size_t qk_elements    = static_cast<std::size_t>(kStateDim) * kQkHeads * columns;
    const std::size_t value_elements = static_cast<std::size_t>(kStateDim) * value_heads * columns;
    const std::size_t gate_elements  = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    std::vector<std::uint16_t> k_bits       = make_bf16(qk_elements, seed + 1);
    std::vector<std::uint16_t> v_bits       = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    k_bits[0] = 0x8000U;
    k_bits[1] = 0x0001U;
    v_bits[0] = 0x8000U;
    v_bits[1] = 0x0001U;
    g[0]      = std::bit_cast<float>(0x80000000U);
    beta[0]   = std::bit_cast<float>(0x00000001U);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);
    state[0] = std::bit_cast<float>(0x80000000U);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = row == 7 ? 7 : (row * 3 + 7) % slots;
    }

    DeviceBuffer device_q        = to_device(q_bits);
    DeviceBuffer device_k        = to_device(k_bits);
    DeviceBuffer device_v        = to_device(v_bits);
    DeviceBuffer device_g        = to_device(g);
    DeviceBuffer device_beta     = to_device(beta);
    DeviceBuffer reference_state = to_device(state);
    DeviceBuffer reference_final(static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * batch * sizeof(float));
    DeviceBuffer record_state    = to_device(state);
    DeviceBuffer device_initial  = to_device(initial_slots);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }

    DeviceBuffer reference_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer record_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    reference_out.fill(0);
    record_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, batch});
    Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, batch});
    Tensor reference_states(reference_state.p, DType::FP32,
                            {kStateDim, kStateDim, value_heads, slots});
    Tensor record_states(record_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    Tensor reference_final_states(reference_final.p, DType::FP32, {kStateDim, kStateDim, value_heads, batch});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor reference_output(reference_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor key_record_tensor(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
    Tensor value_record_tensor(value_record.p, DType::BF16, {kStateDim, value_heads, width, batch});
    Tensor gate_record_tensor(gate_record.p, DType::FP32, {2, value_heads, width, batch});

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    WorkspaceArena reference_workspace(256);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_synchronize();
    const auto launch_reference = [&] {
        CUDA_CHECK(cudaMemsetAsync(reference_out.p, 0, reference_out.bytes, stream));
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        Tensor q_row =
            q.slice(3, row, 1).slice(2, 0, valid_extent).view({kStateDim, kQkHeads, valid_extent});
        Tensor k_row =
            k.slice(3, row, 1).slice(2, 0, valid_extent).view({kStateDim, kQkHeads, valid_extent});
        Tensor v_row = v.slice(3, row, 1)
                           .slice(2, 0, valid_extent)
                           .view({kStateDim, value_heads, valid_extent});
        Tensor g_row =
            g_tensor.slice(2, row, 1).slice(1, 0, valid_extent).view({value_heads, valid_extent});
        Tensor beta_row = beta_tensor.slice(2, row, 1)
                              .slice(1, 0, valid_extent)
                              .view({value_heads, valid_extent});
        Tensor state_row =
            reference_states.slice(3, initial_slots[static_cast<std::size_t>(row)], 1)
                .view({kStateDim, kStateDim, value_heads});
        Tensor out_row = reference_output.slice(3, row, 1)
                             .slice(2, 0, valid_extent)
                             .view({kStateDim, value_heads, valid_extent});
        Tensor final_row = reference_final_states.slice(3, row, 1).view({kStateDim, kStateDim, value_heads});
        ops::gated_delta_net(q_row, k_row, v_row, g_row, beta_row, kScale, true,
                             reference_workspace, state_row, final_row, out_row, stream);
    }
    };
    const auto launch_record = [&] {
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, record_states,
                                           valid, initial, key_record_tensor, value_record_tensor,
                                           gate_record_tensor, record_output, reference_states,
                                           Tensor{}, stream);
    };
    launch_reference();
    launch_record();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (width == 2 || width == 9 || width == 16) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_record();
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (auto& bits : q_bits) bits ^= 0x8000U;
        CUDA_CHECK(cudaMemcpyAsync(device_q.p, q_bits.data(), device_q.bytes, cudaMemcpyHostToDevice, stream));
        if (!dense) {
            for (auto& count : valid_columns) count = 1 + count % width;
            CUDA_CHECK(cudaMemcpyAsync(device_valid.p, valid_columns.data(), device_valid.bytes,
                                        cudaMemcpyHostToDevice, stream));
        }
        CUDA_CHECK(cudaMemsetAsync(key_record.p, 0xff, key_record.bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(value_record.p, 0xff, value_record.bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(gate_record.p, 0xff, gate_record.bytes, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        launch_reference();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    int failures             = 0;
    const std::string suffix = " Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> reference_output_bits =
        from_device<std::uint16_t>(reference_out, value_elements);
    const std::vector<std::uint16_t> record_output_bits =
        from_device<std::uint16_t>(record_out, value_elements);
    failures +=
        verify_equal("replay record output" + suffix, reference_output_bits, record_output_bits);

    const std::vector<std::uint16_t> key_bits_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_bits_after =
        from_device<std::uint16_t>(value_record, value_elements);
    const std::vector<std::uint32_t> gate_bits_after =
        from_device<std::uint32_t>(gate_record, gate_elements * 2);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid_extent = valid_columns[static_cast<std::size_t>(row)];
        for (std::int32_t token = 0; token < width; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(row) * width + token;
            const bool active         = token < valid_extent;
            for (std::int32_t head = 0; head < kQkHeads; ++head) {
                const std::size_t base =
                    static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? k_bits[base + dim] : kBf16Poison;
                    if (key_bits_after[base + dim] != expected) {
                        std::cerr << "key record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
            for (std::int32_t head = 0; head < value_heads; ++head) {
                const std::size_t vector_base =
                    static_cast<std::size_t>((column * value_heads + head) * kStateDim);
                for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                    const std::uint16_t expected = active ? v_bits[vector_base + dim] : kBf16Poison;
                    if (value_bits_after[vector_base + dim] != expected) {
                        std::cerr << "value record mismatch" << suffix << "\n";
                        return failures + 1;
                    }
                }
                const std::size_t gate_offset =
                    static_cast<std::size_t>((column * value_heads + head) * 2);
                const std::size_t source_offset =
                    static_cast<std::size_t>(column * value_heads + head);
                const std::uint32_t expected_g =
                    active ? std::bit_cast<std::uint32_t>(g[source_offset]) : kFp32Poison;
                const std::uint32_t expected_beta =
                    active ? std::bit_cast<std::uint32_t>(beta[source_offset]) : kFp32Poison;
                if (gate_bits_after[gate_offset] != expected_g ||
                    gate_bits_after[gate_offset + 1] != expected_beta) {
                    std::cerr << "gate record mismatch" << suffix << "\n";
                    return failures + 1;
                }
            }
            if (!active) {
                const std::size_t output_base =
                    static_cast<std::size_t>(column) * value_heads * kStateDim;
                for (std::int32_t index = 0; index < value_heads * kStateDim; ++index) {
                    if (record_output_bits[output_base + index] != 0) {
                        std::cerr << "record invalid output is not zero" << suffix << "\n";
                        return failures + 1;
                    }
                }
            }
        }
    }

    const std::vector<float> state_after = from_device<float>(record_state, state_elements);
    if (std::memcmp(state_after.data(), state.data(), state_elements * sizeof(float)) != 0) {
        std::cerr << "replay record modified source state" << suffix << "\n";
        ++failures;
    }
    if (from_device<std::uint16_t>(device_q, qk_elements) != q_bits ||
        from_device<std::uint16_t>(device_k, qk_elements) != k_bits ||
        from_device<std::uint16_t>(device_v, value_elements) != v_bits) {
        std::cerr << "replay record modified inputs" << suffix << "\n";
        ++failures;
    }
    return failures;
}

// Every node of a verify tree must produce exactly the output of the chain record over its root
// path, and every column must still record its own raw inputs.
int run_tree_case(std::int32_t value_heads, std::int32_t width, std::int32_t batch,
                  const std::vector<std::int32_t>& valid_columns, std::uint32_t seed,
                  double chain_bias) {
    constexpr std::int32_t kQkHeads   = 16;
    constexpr std::int32_t kChainRows = 8;
    const std::int32_t columns        = width * batch;
    const std::int32_t slots          = 8;
    const std::size_t qk_column       = static_cast<std::size_t>(kStateDim) * kQkHeads;
    const std::size_t value_column    = static_cast<std::size_t>(kStateDim) * value_heads;
    const std::size_t qk_elements     = qk_column * columns;
    const std::size_t value_elements  = value_column * columns;
    const std::size_t gate_elements   = static_cast<std::size_t>(value_heads) * columns;
    const std::size_t state_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * value_heads * slots;

    const std::vector<std::uint16_t> q_bits = make_bf16(qk_elements, seed);
    const std::vector<std::uint16_t> k_bits = make_bf16(qk_elements, seed + 1);
    const std::vector<std::uint16_t> v_bits = make_bf16(value_elements, seed + 2);
    std::vector<float> g(gate_elements);
    std::vector<float> beta(gate_elements);
    fill_uniform(g, seed + 3, -1.2F, -0.02F);
    fill_uniform(beta, seed + 4, 0.02F, 0.98F);
    std::vector<float> state(state_elements);
    fill_uniform(state, seed + 5, -0.03F, 0.03F);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> parents(static_cast<std::size_t>(columns), 0);
    std::vector<std::vector<std::int32_t>> trees(static_cast<std::size_t>(batch));
    for (std::int32_t row = 0; row < batch; ++row) {
        initial_slots[static_cast<std::size_t>(row)] = (row * 5 + 3) % slots;
        trees[static_cast<std::size_t>(row)]         = random_verify_tree(
            valid_columns[static_cast<std::size_t>(row)], seed + 97U * row, chain_bias);
        std::copy(trees[static_cast<std::size_t>(row)].begin(),
                  trees[static_cast<std::size_t>(row)].end(),
                  parents.begin() + static_cast<std::ptrdiff_t>(row) * width);
    }

    DeviceBuffer device_q       = to_device(q_bits);
    DeviceBuffer device_k       = to_device(k_bits);
    DeviceBuffer device_v       = to_device(v_bits);
    DeviceBuffer device_g       = to_device(g);
    DeviceBuffer device_beta    = to_device(beta);
    DeviceBuffer device_state   = to_device(state);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_valid   = to_device(valid_columns);
    DeviceBuffer device_parents = to_device(parents);
    DeviceBuffer tree_out(value_elements * sizeof(std::uint16_t));
    DeviceBuffer key_record(qk_elements * sizeof(std::uint16_t));
    DeviceBuffer value_record(value_elements * sizeof(std::uint16_t));
    DeviceBuffer gate_record(gate_elements * 2 * sizeof(std::uint32_t));
    tree_out.fill(0xff);
    key_record.fill(0xff);
    value_record.fill(0xff);
    gate_record.fill(0xff);

    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    Tensor states(device_state.p, DType::FP32, {kStateDim, kStateDim, value_heads, slots});
    {
        Tensor q(device_q.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
        Tensor k(device_k.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
        Tensor v(device_v.p, DType::BF16, {kStateDim, value_heads, width, batch});
        Tensor g_tensor(device_g.p, DType::FP32, {value_heads, width, batch});
        Tensor beta_tensor(device_beta.p, DType::FP32, {value_heads, width, batch});
        Tensor valid(device_valid.p, DType::I32, {batch});
        Tensor initial(device_initial.p, DType::I32, {batch});
        Tensor tree_parents(device_parents.p, DType::I32, {width, batch});
        Tensor out(tree_out.p, DType::BF16, {kStateDim, value_heads, width, batch});
        Tensor keys(key_record.p, DType::BF16, {kStateDim, kQkHeads, width, batch});
        Tensor values(value_record.p, DType::BF16, {kStateDim, value_heads, width, batch});
        Tensor gates(gate_record.p, DType::FP32, {2, value_heads, width, batch});
        ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states, valid,
                                           initial, keys, values, gates, out, Tensor{},
                                           tree_parents, nullptr);
        cuda_synchronize();
    }
    const std::vector<std::uint16_t> tree_bits =
        from_device<std::uint16_t>(tree_out, value_elements);

    int failures             = 0;
    const std::string suffix = " tree Hv=" + std::to_string(value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(batch);
    const std::vector<std::uint16_t> key_after =
        from_device<std::uint16_t>(key_record, qk_elements);
    const std::vector<std::uint16_t> value_after =
        from_device<std::uint16_t>(value_record, value_elements);
    for (std::int32_t row = 0; row < batch; ++row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(row)];
             ++token) {
            const std::size_t column = static_cast<std::size_t>(row) * width + token;
            if (!std::equal(k_bits.begin() + column * qk_column,
                            k_bits.begin() + (column + 1) * qk_column,
                            key_after.begin() + column * qk_column) ||
                !std::equal(v_bits.begin() + column * value_column,
                            v_bits.begin() + (column + 1) * value_column,
                            value_after.begin() + column * value_column)) {
                std::cerr << "tree record inputs differ" << suffix << "\n";
                return failures + 1;
            }
        }
    }

    // Chain references: node r of the tree becomes chain row r over its gathered root path.
    const std::int32_t chain_columns = width * kChainRows;
    DeviceBuffer chain_q(qk_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_k(qk_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_v(value_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_g(static_cast<std::size_t>(value_heads) * chain_columns * sizeof(float));
    DeviceBuffer chain_beta(static_cast<std::size_t>(value_heads) * chain_columns * sizeof(float));
    DeviceBuffer chain_out(value_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_keys(qk_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_values(value_column * chain_columns * sizeof(std::uint16_t));
    DeviceBuffer chain_gates(static_cast<std::size_t>(value_heads) * chain_columns * 2 *
                             sizeof(std::uint32_t));
    DeviceBuffer chain_valid(kChainRows * sizeof(std::int32_t));
    DeviceBuffer chain_initial(kChainRows * sizeof(std::int32_t));
    for (std::int32_t row = 0; row < batch; ++row) {
        const auto& tree      = trees[static_cast<std::size_t>(row)];
        const auto node_count = static_cast<std::int32_t>(tree.size());
        for (std::int32_t first = 0; first < node_count; first += kChainRows) {
            const std::int32_t count = std::min(kChainRows, node_count - first);
            std::vector<std::uint16_t> gq(qk_column * chain_columns, 0);
            std::vector<std::uint16_t> gk(qk_column * chain_columns, 0);
            std::vector<std::uint16_t> gv(value_column * chain_columns, 0);
            std::vector<float> gg(static_cast<std::size_t>(value_heads) * chain_columns, -0.5F);
            std::vector<float> gb(static_cast<std::size_t>(value_heads) * chain_columns, 0.5F);
            std::vector<std::int32_t> lengths(kChainRows, 1);
            std::vector<std::int32_t> chain_slots(kChainRows,
                                                  initial_slots[static_cast<std::size_t>(row)]);
            for (std::int32_t r = 0; r < count; ++r) {
                const std::vector<std::int32_t> path = verify_tree_path(tree, first + r);
                lengths[static_cast<std::size_t>(r)] = static_cast<std::int32_t>(path.size());
                for (std::size_t j = 0; j < path.size(); ++j) {
                    const std::size_t source = static_cast<std::size_t>(row) * width + path[j];
                    const std::size_t target = static_cast<std::size_t>(r) * width + j;
                    std::copy_n(q_bits.begin() + source * qk_column, qk_column,
                                gq.begin() + target * qk_column);
                    std::copy_n(k_bits.begin() + source * qk_column, qk_column,
                                gk.begin() + target * qk_column);
                    std::copy_n(v_bits.begin() + source * value_column, value_column,
                                gv.begin() + target * value_column);
                    std::copy_n(g.begin() + source * value_heads, value_heads,
                                gg.begin() + target * value_heads);
                    std::copy_n(beta.begin() + source * value_heads, value_heads,
                                gb.begin() + target * value_heads);
                }
            }
            CUDA_CHECK(cudaMemcpy(chain_q.p, gq.data(), chain_q.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(chain_k.p, gk.data(), chain_k.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(chain_v.p, gv.data(), chain_v.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(chain_g.p, gg.data(), chain_g.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(
                cudaMemcpy(chain_beta.p, gb.data(), chain_beta.bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(chain_valid.p, lengths.data(), chain_valid.bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(chain_initial.p, chain_slots.data(), chain_initial.bytes,
                                  cudaMemcpyHostToDevice));
            Tensor q(chain_q.p, DType::BF16, {kStateDim, kQkHeads, width, kChainRows});
            Tensor k(chain_k.p, DType::BF16, {kStateDim, kQkHeads, width, kChainRows});
            Tensor v(chain_v.p, DType::BF16, {kStateDim, value_heads, width, kChainRows});
            Tensor g_tensor(chain_g.p, DType::FP32, {value_heads, width, kChainRows});
            Tensor beta_tensor(chain_beta.p, DType::FP32, {value_heads, width, kChainRows});
            Tensor valid(chain_valid.p, DType::I32, {kChainRows});
            Tensor initial(chain_initial.p, DType::I32, {kChainRows});
            Tensor out(chain_out.p, DType::BF16, {kStateDim, value_heads, width, kChainRows});
            Tensor keys(chain_keys.p, DType::BF16, {kStateDim, kQkHeads, width, kChainRows});
            Tensor values(chain_values.p, DType::BF16, {kStateDim, value_heads, width, kChainRows});
            Tensor gates(chain_gates.p, DType::FP32, {2, value_heads, width, kChainRows});
            ops::gated_delta_net_replay_record(q, k, v, g_tensor, beta_tensor, kScale, states,
                                               valid, initial, keys, values, gates, out, Tensor{},
                                               Tensor{}, nullptr);
            cuda_synchronize();
            const std::vector<std::uint16_t> chain_bits =
                from_device<std::uint16_t>(chain_out, value_column * chain_columns);
            for (std::int32_t r = 0; r < count; ++r) {
                const std::size_t chain_column =
                    static_cast<std::size_t>(r) * width + lengths[static_cast<std::size_t>(r)] - 1;
                const std::size_t tree_column = static_cast<std::size_t>(row) * width + first + r;
                if (!std::equal(chain_bits.begin() + chain_column * value_column,
                                chain_bits.begin() + (chain_column + 1) * value_column,
                                tree_bits.begin() + tree_column * value_column)) {
                    std::cerr << "tree output differs from its root-path chain at row " << row
                              << " node " << first + r << suffix << "\n";
                    return failures + 1;
                }
            }
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

    int failures = 0;
    failures += run_case(32, 2, 1, {}, 1701U);
    failures += run_case(32, 16, 1, {7}, 1711U);
    failures += run_case(32, 6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1721U);
    for (int width = 2; width <= 16; ++width) {
        failures += run_case(48, width, 1, {}, 1730U + width);
        std::vector<std::int32_t> valid(8);
        for (int b = 0; b < 8; ++b) valid[b] = b == 0 ? width : 1 + (3 * b) % width;
        failures += run_case(48, width, 8, valid, 1760U + width);
    }
    failures += run_case(48, 5, 3, {5, 3, 1}, 1791U);
    for (int width = 2; width <= 16; ++width) {
        failures += run_tree_case(48, width, 1, {width}, 1800U + width, 0.3);
    }
    failures += run_tree_case(48, 16, 8, {16, 16, 16, 16, 9, 12, 3, 16}, 1830U, 0.5);
    failures += run_tree_case(48, 16, 4, {16, 16, 16, 16}, 1840U, 0.0);
    failures += run_tree_case(32, 16, 3, {16, 11, 16}, 1850U, 0.8);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net_replay_record\n";
    return failures == 0 ? 0 : 1;
}
