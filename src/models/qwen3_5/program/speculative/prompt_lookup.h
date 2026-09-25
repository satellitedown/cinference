#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

// Prompt-lookup proposals for DFlash2 verify trees. A proposal continues the context (whose last
// token is the round's anchor) with the tokens that followed the most recent earlier occurrence of
// its longest indexed suffix window (8, 4 or 2 tokens). Each window length keeps its own estimate
// of the probability that a proposed token is committed given that the previous one was, learned
// from how far earlier proposals matched the committed tokens.
class PromptLookup {
public:
    static constexpr std::array<std::uint32_t, 3> kWindows{8, 4, 2};

    // Writes up to out.size() tokens and returns how many. The context is indexed incrementally;
    // one that no longer extends the indexed tokens is indexed again from the start.
    std::uint32_t propose(std::span<const TokenId> context, std::span<TokenId> out);

    // Scores the last proposal against the round's committed tokens.
    void observe(std::span<const TokenId> committed);

    // Log-probability of each token of the last proposal (0 when there is none).
    [[nodiscard]] float log_probability() const noexcept;

private:
    // Open-addressed map from a window's 64-bit hash to its latest end position.
    struct WindowTable {
        std::vector<std::uint64_t> keys;
        std::vector<std::uint32_t> ends; // end position + 1; 0 marks an empty slot
        std::size_t size = 0;

        void insert(std::uint64_t key, std::uint32_t end);
        [[nodiscard]] std::uint32_t find(std::uint64_t key) const noexcept;
        void clear() noexcept;
    };

    struct Estimate {
        double hits   = 0.0;
        double misses = 0.0;
    };

    void index_until(std::span<const TokenId> context, std::size_t end);

    std::array<WindowTable, kWindows.size()> tables_;
    std::array<Estimate, kWindows.size()> estimates_;
    std::size_t indexed_  = 0; // window end positions [0, indexed_) are in the tables
    TokenId indexed_last_ = 0;
    std::vector<TokenId> proposal_;
    int proposal_window_ = -1;
};

} // namespace ninfer::models::qwen3_5
