#include "models/qwen3_5/program/speculative/prompt_lookup.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace ninfer::models::qwen3_5 {
namespace {

// Per observed round, older evidence decays so the estimate follows the output's current mode
// (free reasoning text vs. copied code).
constexpr double kDecay       = 0.9;
constexpr double kPriorHits   = 1.0;
constexpr double kPriorMisses = 3.0;
constexpr double kMinimum     = 0.01;
constexpr double kMaximum     = 0.97;

std::uint64_t window_hash(const TokenId* tokens, std::uint32_t length) {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL ^ length;
    for (std::uint32_t i = 0; i < length; ++i) {
        hash ^= static_cast<std::uint32_t>(tokens[i]);
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32;
    }
    return hash;
}

} // namespace

void PromptLookup::WindowTable::insert(std::uint64_t key, std::uint32_t end) {
    if ((size + 1) * 2 > keys.size()) {
        std::vector<std::uint64_t> old_keys = std::move(keys);
        std::vector<std::uint32_t> old_ends = std::move(ends);
        const std::size_t capacity          = std::max<std::size_t>(4096, old_keys.size() * 2);
        keys.assign(capacity, 0);
        ends.assign(capacity, 0);
        size = 0;
        for (std::size_t slot = 0; slot < old_keys.size(); ++slot) {
            if (old_ends[slot] != 0) { insert(old_keys[slot], old_ends[slot] - 1); }
        }
    }
    const std::size_t mask = keys.size() - 1;
    for (std::size_t slot = key & mask;; slot = (slot + 1) & mask) {
        if (ends[slot] == 0) {
            keys[slot] = key;
            ends[slot] = end + 1;
            ++size;
            return;
        }
        if (keys[slot] == key) {
            ends[slot] = end + 1;
            return;
        }
    }
}

std::uint32_t PromptLookup::WindowTable::find(std::uint64_t key) const noexcept {
    if (keys.empty()) { return 0; }
    const std::size_t mask = keys.size() - 1;
    for (std::size_t slot = key & mask;; slot = (slot + 1) & mask) {
        if (ends[slot] == 0) { return 0; }
        if (keys[slot] == key) { return ends[slot]; }
    }
}

void PromptLookup::WindowTable::clear() noexcept {
    std::fill(keys.begin(), keys.end(), 0);
    std::fill(ends.begin(), ends.end(), 0);
    size = 0;
}

void PromptLookup::index_until(std::span<const TokenId> context, std::size_t end) {
    for (std::size_t position = indexed_; position < end; ++position) {
        for (std::size_t window = 0; window < kWindows.size(); ++window) {
            const std::uint32_t length = kWindows[window];
            if (position + 1 < length) { continue; }
            tables_[window].insert(window_hash(context.data() + position + 1 - length, length),
                                   static_cast<std::uint32_t>(position));
        }
    }
    if (end > indexed_) {
        indexed_      = end;
        indexed_last_ = context[end - 1];
    }
}

std::uint32_t PromptLookup::propose(std::span<const TokenId> context, std::span<TokenId> out) {
    proposal_.clear();
    proposal_window_         = -1;
    const std::size_t length = context.size();
    if (indexed_ > length || (indexed_ != 0 && context[indexed_ - 1] != indexed_last_)) {
        for (WindowTable& table : tables_) { table.clear(); }
        estimates_ = {};
        indexed_   = 0;
    }
    // Windows ending before the anchor, so a match is an earlier occurrence of the suffix.
    if (length < 2 || out.empty()) { return 0; }
    index_until(context, length - 1);
    for (std::size_t window = 0; window < kWindows.size(); ++window) {
        const std::uint32_t size = kWindows[window];
        if (length < size + 1) { continue; }
        const TokenId* suffix    = context.data() + length - size;
        const std::uint32_t end1 = tables_[window].find(window_hash(suffix, size));
        if (end1 == 0) { continue; }
        const std::size_t end = end1 - 1;
        if (!std::equal(suffix, suffix + size, context.data() + end + 1 - size)) { continue; }
        // The continuation after the occurrence; past the anchor it repeats with the period.
        const std::size_t period = length - 1 - end;
        for (std::size_t j = 0; j < out.size(); ++j) {
            const std::size_t source = end + 1 + j;
            out[j]                   = source < length ? context[source] : out[j - period];
        }
        proposal_.assign(out.begin(), out.end());
        proposal_window_ = static_cast<int>(window);
        return static_cast<std::uint32_t>(out.size());
    }
    return 0;
}

void PromptLookup::observe(std::span<const TokenId> committed) {
    if (proposal_window_ < 0) { return; }
    const std::size_t limit = std::min(proposal_.size(), committed.size());
    std::size_t hits        = 0;
    while (hits < limit && proposal_[hits] == committed[hits]) { ++hits; }
    Estimate& estimate = estimates_[static_cast<std::size_t>(proposal_window_)];
    estimate.hits      = kDecay * estimate.hits + static_cast<double>(hits);
    estimate.misses    = kDecay * estimate.misses + (hits < limit ? 1.0 : 0.0);
    proposal_window_   = -1;
}

float PromptLookup::log_probability() const noexcept {
    if (proposal_window_ < 0) { return 0.0F; }
    const Estimate& estimate = estimates_[static_cast<std::size_t>(proposal_window_)];
    const double probability = (estimate.hits + kPriorHits) /
                               (estimate.hits + estimate.misses + kPriorHits + kPriorMisses);
    return static_cast<float>(std::log(std::clamp(probability, kMinimum, kMaximum)));
}

} // namespace ninfer::models::qwen3_5
