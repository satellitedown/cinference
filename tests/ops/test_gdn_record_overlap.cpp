// Exactness of the overlapped GDN record route. gdn_norm_gating_fp8_hidden must equal the A8
// quantization of gdn_norm_gating_proj's hidden output and gdn_norm_gating_control its g and beta,
// for every T of the fused route; the two activation-form halves of the FP8 record projection, run
// concurrently on two streams beside the control Op, must equal the complete record Op on that
// hidden output.

#include "core/weight.h"
#include "core/device.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"
#include "ops/verify_tree_test_common.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kHeads      = 48;
constexpr std::int32_t kQueryRows  = 2048;
constexpr std::int32_t kKeyRows    = 2048;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kParentRows = kChannels + kZRows;
constexpr std::int32_t kWidth      = 16;
constexpr std::int32_t kSlots      = 8;
constexpr float kEps               = 1.0e-6F;

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

std::vector<std::uint16_t> random_bf16(std::size_t elements, std::uint32_t seed, float low,
                                       float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

// The residual, norm weight and control parameters of one norm-gating problem on the device.
struct NormGating {
    DeviceBuffer x, norm, ab, a_log, dt_bias;
    Tensor tx, tn, ta, td;
    Weight parent, a, b;

    NormGating(std::int32_t tokens, std::uint32_t seed) {
        std::vector<std::uint16_t> norm_bits = random_bf16(kHidden, seed + 1, -0.2F, 0.2F);
        norm_bits[0]                         = f32_to_bf16(-1.0F);
        std::vector<float> log_decay(kHeads), bias(kHeads);
        fill_uniform(log_decay, seed + 3, -2.0F, 1.0F);
        fill_uniform(bias, seed + 4, -1.0F, 1.0F);
        x    = to_device(random_bf16(std::size_t(kHidden) * tokens, seed, -1.0F, 1.0F));
        norm = to_device(norm_bits);
        ab   = to_device(random_bf16(std::size_t(2 * kHeads) * kHidden, seed + 2, -0.015F, 0.015F));
        a_log   = to_device(log_decay);
        dt_bias = to_device(bias);
        tx      = Tensor(x.p, DType::BF16, {kHidden, tokens});
        tn      = Tensor(norm.p, DType::BF16, {kHidden});
        ta      = Tensor(a_log.p, DType::FP32, {kHeads});
        td      = Tensor(dt_bias.p, DType::FP32, {kHeads});
        parent  = bf16_weight(ab.p, 2 * kHeads, kHidden);
        a       = bf16_weight(ab.p, kHeads, kHidden);
        b       = bf16_weight(static_cast<std::uint8_t*>(ab.p) +
                                  std::size_t(kHeads) * kHidden * sizeof(std::uint16_t),
                              kHeads, kHidden);
    }

    // The complete Op's h, g and beta.
    void complete(Tensor& h, Tensor& g, Tensor& beta, cudaStream_t stream) {
        const std::size_t capacity =
            ops::gdn_norm_gating_proj_workspace_capacity_bytes(kHeads, kHidden, tx.ne[1], tx.ne[1]);
        WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::gdn_norm_gating_proj(tx, tn, kEps, parent, ta, td, workspace, h, g, beta,
                                  {.stream = stream, .multiprocessor_count = sm_count()});
    }

    static std::int32_t sm_count() {
        int device = 0;
        int count  = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device),
                   "cudaDeviceGetAttribute");
        return count;
    }
};

// The FP8 A8 activation of BF16 columns as the A8 route forms it: scale = column maximum / 448,
// codes = E4M3 (saturating, nearest) of value * (1 / scale).
void quantize_columns(const std::vector<std::uint16_t>& hidden, std::int32_t tokens,
                      std::vector<std::uint8_t>& codes, std::vector<float>& scales) {
    codes.assign(hidden.size(), 0);
    scales.assign(static_cast<std::size_t>(tokens), 0.0F);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * kHidden;
        float maximum          = 0.0F;
        for (std::int32_t row = 0; row < kHidden; ++row) {
            maximum = std::max(maximum, std::fabs(bf16_to_f32(hidden[base + row])));
        }
        const float scale                       = maximum > 0.0F ? maximum / 448.0F : 0.0F;
        const float inverse                     = scale > 0.0F ? 1.0F / scale : 0.0F;
        scales[static_cast<std::size_t>(token)] = scale;
        for (std::int32_t row = 0; row < kHidden; ++row) {
            codes[base + row] = __nv_cvt_float_to_fp8(bf16_to_f32(hidden[base + row]) * inverse,
                                                      __NV_SATFINITE, __NV_E4M3);
        }
    }
}

int check(bool equal, const std::string& label) {
    if (equal) { return 0; }
    std::cerr << label << ": differs from the complete Op\n";
    return 1;
}

int run_norm_gating_case(std::int32_t tokens, std::uint32_t seed) {
    const std::string label = "T=" + std::to_string(tokens);
    if (!ops::gdn_norm_gating_split_admits(kHeads, kHidden, tokens)) {
        std::cerr << label << ": split forms not admitted\n";
        return 1;
    }
    NormGating problem(tokens, seed);
    const std::size_t hidden_elements  = std::size_t(kHidden) * tokens;
    const std::size_t control_elements = std::size_t(kHeads) * tokens;
    DeviceBuffer h(hidden_elements * 2), g(control_elements * 4), beta(control_elements * 4);
    DeviceBuffer codes(hidden_elements), scales(std::size_t(tokens) * 4);
    DeviceBuffer split_g(control_elements * 4), split_beta(control_elements * 4);
    DeviceBuffer pair_g(control_elements * 4), pair_beta(control_elements * 4);
    Tensor th(h.p, DType::BF16, {kHidden, tokens});
    Tensor tg(g.p, DType::FP32, {kHeads, tokens});
    Tensor tbeta(beta.p, DType::FP32, {kHeads, tokens});
    Tensor tcodes(codes.p, DType::U8, {kHidden, tokens});
    Tensor tscales(scales.p, DType::FP32, {tokens});
    Tensor tsplit_g(split_g.p, DType::FP32, {kHeads, tokens});
    Tensor tsplit_beta(split_beta.p, DType::FP32, {kHeads, tokens});
    Tensor tpair_g(pair_g.p, DType::FP32, {kHeads, tokens});
    Tensor tpair_beta(pair_beta.p, DType::FP32, {kHeads, tokens});

    problem.complete(th, tg, tbeta, nullptr);
    ops::gdn_norm_gating_fp8_hidden(problem.tx, problem.tn, kEps, tcodes, tscales, nullptr);
    ops::gdn_norm_gating_control(problem.tx, problem.tn, kEps, problem.parent, problem.ta,
                                 problem.td, tsplit_g, tsplit_beta, nullptr);
    ops::gdn_norm_gating_control(problem.tx, problem.tn, kEps, problem.a, problem.b, problem.ta,
                                 problem.td, tpair_g, tpair_beta, nullptr);
    cuda_synchronize();

    std::vector<std::uint8_t> expected_codes;
    std::vector<float> expected_scales;
    quantize_columns(from_device<std::uint16_t>(h, hidden_elements), tokens, expected_codes,
                     expected_scales);
    const auto reference_g    = from_device<std::uint32_t>(g, control_elements);
    const auto reference_beta = from_device<std::uint32_t>(beta, control_elements);
    int failures              = 0;
    failures += check(from_device<std::uint8_t>(codes, hidden_elements) == expected_codes,
                      label + " fp8 hidden codes");
    std::vector<float> actual_scales = from_device<float>(scales, std::size_t(tokens));
    failures +=
        check(std::equal(
                  actual_scales.begin(), actual_scales.end(), expected_scales.begin(),
                  [](float lhs, float rhs) { return std::memcmp(&lhs, &rhs, sizeof(float)) == 0; }),
              label + " fp8 hidden scales");
    failures += check(from_device<std::uint32_t>(split_g, control_elements) == reference_g,
                      label + " control g");
    failures += check(from_device<std::uint32_t>(split_beta, control_elements) == reference_beta,
                      label + " control beta");
    failures += check(from_device<std::uint32_t>(pair_g, control_elements) == reference_g,
                      label + " two-weight control g");
    failures += check(from_device<std::uint32_t>(pair_beta, control_elements) == reference_beta,
                      label + " two-weight control beta");
    return failures;
}

// Outputs of one record block: query, key, value, z and the record, as downloaded bits.
struct RecordOutputs {
    std::vector<std::uint16_t> query, key, value, z, record;
    std::vector<std::uint32_t> g, beta;
};

int run_record_case(DevicePackedWeight& parent, std::int32_t valid, bool tree, std::uint32_t seed) {
    const std::string label = "record valid=" + std::to_string(valid) + (tree ? " tree" : " chain");
    const ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
    if (!ops::gdn_input_proj_conv_record_takes_activation(parent.view(), policy, 1, kWidth)) {
        std::cerr << label << ": activation form not admitted\n";
        return 1;
    }
    NormGating problem(kWidth, seed);
    DeviceBuffer conv_weight =
        to_device(random_bf16(std::size_t(kChannels) * 4, seed + 11, -0.02F, 0.02F));
    DeviceBuffer conv_state =
        to_device(random_bf16(std::size_t(kChannels) * 3 * kSlots, seed + 12, -0.05F, 0.05F));
    DeviceBuffer valid_columns = to_device(std::vector<std::int32_t>{valid});
    DeviceBuffer initial_slot  = to_device(std::vector<std::int32_t>{5});
    DeviceBuffer parents;
    if (tree) {
        std::vector<std::int32_t> columns(kWidth, 0);
        const std::vector<std::int32_t> nodes = random_verify_tree(valid, seed + 13, 0.4);
        std::copy(nodes.begin(), nodes.end(), columns.begin());
        parents = to_device(columns);
    }
    Tensor tconv(conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor tstate(conv_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor tvalid(valid_columns.p, DType::I32, {1});
    Tensor tinitial(initial_slot.p, DType::I32, {1});
    Tensor tparents = tree ? Tensor(parents.p, DType::I32, {kWidth, 1}) : Tensor{};

    const auto run = [&](bool overlapped) {
        DeviceBuffer h(std::size_t(kHidden) * kWidth * 2);
        DeviceBuffer g(std::size_t(kHeads) * kWidth * 4), beta(std::size_t(kHeads) * kWidth * 4);
        DeviceBuffer codes(std::size_t(kHidden) * kWidth), scales(std::size_t(kWidth) * 4);
        DeviceBuffer query(std::size_t(kQueryRows) * kWidth * 2);
        DeviceBuffer key(std::size_t(kKeyRows) * kWidth * 2);
        DeviceBuffer value(std::size_t(kValueRows) * kWidth * 2);
        DeviceBuffer z(std::size_t(kZRows) * kWidth * 2);
        DeviceBuffer record(std::size_t(kChannels) * kWidth * 2);
        Tensor th(h.p, DType::BF16, {kHidden, kWidth});
        Tensor tg(g.p, DType::FP32, {kHeads, kWidth});
        Tensor tbeta(beta.p, DType::FP32, {kHeads, kWidth});
        Tensor tq(query.p, DType::BF16, {kQueryRows, kWidth, 1});
        Tensor tk(key.p, DType::BF16, {kKeyRows, kWidth, 1});
        Tensor tv(value.p, DType::BF16, {kValueRows, kWidth, 1});
        Tensor tz(z.p, DType::BF16, {kZRows, kWidth, 1});
        Tensor trecord(record.p, DType::BF16, {kChannels, kWidth, 1});
        if (!overlapped) {
            problem.complete(th, tg, tbeta, nullptr);
            const std::size_t capacity = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16, kParentRows, kHidden, policy, 1, kWidth, kWidth);
            WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
            ops::gdn_input_proj_conv_record(th.view({kHidden, kWidth, 1}), parent.view(), tconv,
                                            tstate, tvalid, tinitial, tparents, trecord, tq, tk, tv,
                                            tz, policy, workspace, nullptr);
        } else {
            cudaStream_t main = nullptr;
            cudaStream_t side = nullptr;
            cudaEvent_t ready = nullptr;
            cuda_check(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking), "stream");
            cuda_check(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "stream");
            cuda_check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "event");
            Tensor tcodes(codes.p, DType::U8, {kHidden, kWidth});
            Tensor tscales(scales.p, DType::FP32, {kWidth});
            ops::gdn_norm_gating_fp8_hidden(problem.tx, problem.tn, kEps, tcodes, tscales, main);
            cuda_check(cudaEventRecord(ready, main), "cudaEventRecord");
            cuda_check(cudaStreamWaitEvent(side, ready, 0), "cudaStreamWaitEvent");
            ops::gdn_input_proj_conv_record(tcodes, tscales, parent.view(), tconv, tstate, tvalid,
                                            tinitial, tparents, trecord, tq, tk, tv, tz,
                                            ops::GdnRecordRows::QueryKeyValue, main);
            ops::gdn_norm_gating_control(problem.tx, problem.tn, kEps, problem.parent, problem.ta,
                                         problem.td, tg, tbeta, side);
            ops::gdn_input_proj_conv_record(tcodes, tscales, parent.view(), tconv, tstate, tvalid,
                                            tinitial, tparents, trecord, tq, tk, tv, tz,
                                            ops::GdnRecordRows::OutputGate, side);
            cuda_synchronize(main);
            cuda_synchronize(side);
            cuda_check(cudaEventDestroy(ready), "cudaEventDestroy");
            cuda_check(cudaStreamDestroy(side), "cudaStreamDestroy");
            cuda_check(cudaStreamDestroy(main), "cudaStreamDestroy");
        }
        cuda_synchronize();
        RecordOutputs outputs;
        outputs.query  = from_device<std::uint16_t>(query, std::size_t(kQueryRows) * kWidth);
        outputs.key    = from_device<std::uint16_t>(key, std::size_t(kKeyRows) * kWidth);
        outputs.value  = from_device<std::uint16_t>(value, std::size_t(kValueRows) * kWidth);
        outputs.z      = from_device<std::uint16_t>(z, std::size_t(kZRows) * kWidth);
        outputs.record = from_device<std::uint16_t>(record, std::size_t(kChannels) * valid);
        outputs.g      = from_device<std::uint32_t>(g, std::size_t(kHeads) * kWidth);
        outputs.beta   = from_device<std::uint32_t>(beta, std::size_t(kHeads) * kWidth);
        return outputs;
    };

    const RecordOutputs complete   = run(false);
    const RecordOutputs overlapped = run(true);
    int failures                   = 0;
    failures += check(overlapped.query == complete.query, label + " query");
    failures += check(overlapped.key == complete.key, label + " key");
    failures += check(overlapped.value == complete.value, label + " value");
    failures += check(overlapped.z == complete.z, label + " z");
    failures += check(overlapped.record == complete.record, label + " record");
    failures += check(overlapped.g == complete.g, label + " g");
    failures += check(overlapped.beta == complete.beta, label + " beta");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (std::int32_t tokens : {1, 2, 3, 14, 15, 16, 28, 29, 42}) {
        failures += run_norm_gating_case(tokens, 2100U + static_cast<std::uint32_t>(tokens));
    }
    if (ops::gdn_norm_gating_split_admits(kHeads, kHidden, 43)) {
        std::cerr << "T=43: split forms admitted outside the fused route\n";
        ++failures;
    }
    DevicePackedWeight parent(quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16,
                                                                      kParentRows, kHidden, 2201U));
    failures += run_record_case(parent, 16, false, 2210U);
    failures += run_record_case(parent, 11, false, 2220U);
    failures += run_record_case(parent, 16, true, 2230U);
    failures += run_record_case(parent, 9, true, 2240U);
    if (ops::gdn_input_proj_conv_record_takes_activation(parent.view(), ops::LinearPolicy::A16Only,
                                                         1, kWidth) ||
        ops::gdn_input_proj_conv_record_takes_activation(parent.view(), ops::LinearPolicy::AllowA8,
                                                         2, kWidth)) {
        std::cerr << "activation form admitted outside the single-block A8 route\n";
        ++failures;
    }
    failures += parent.verify_preserved("FP8 record parent weight");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_record_overlap\n";
    return failures == 0 ? 0 : 1;
}
