#include "models/qwen3_5/program/speculative/prompt_lookup.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::PromptLookup;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

std::vector<TokenId> propose(PromptLookup& lookup, const std::vector<TokenId>& context,
                             std::size_t count) {
    std::vector<TokenId> out(count, -1);
    out.resize(lookup.propose(context, out));
    return out;
}

} // namespace

int main() {
    {
        // The longest window's most recent earlier occurrence wins; the suffix itself never does.
        PromptLookup lookup;
        const std::vector<TokenId> context{3, 4, 7, 9, 1, 3, 4, 8, 5, 6, 3, 4};
        expect(propose(lookup, context, 3) == std::vector<TokenId>({8, 5, 6}),
               "latest earlier occurrence of the 2-token suffix");
        const std::vector<TokenId> longer{10, 11, 12, 13, 14, 15, 16, 17, 18, 50, 13, 14,
                                          15, 16, 17, 60, 10, 11, 12, 13, 14, 15, 16, 17};
        expect(propose(lookup, longer, 4) == std::vector<TokenId>({18, 50, 13, 14}),
               "8-token match over a more recent 4-token one");
    }
    {
        // Past the anchor the continuation repeats with the occurrence's period.
        PromptLookup lookup;
        const std::vector<TokenId> context{5, 6, 5, 6, 5, 6};
        expect(propose(lookup, context, 5) == std::vector<TokenId>({5, 6, 5, 6, 5}),
               "periodic continuation");
    }
    {
        // No earlier occurrence, then a context that no longer extends the indexed one.
        PromptLookup lookup;
        expect(propose(lookup, {1, 2, 3, 4}, 4).empty(), "no occurrence");
        expect(propose(lookup, {1, 2, 3, 4, 1, 2}, 2) == std::vector<TokenId>({3, 4}),
               "incremental extension");
        expect(propose(lookup, {7, 2, 9, 7, 2}, 2) == std::vector<TokenId>({9, 7}),
               "replaced context re-indexed");
    }
    {
        // Matching proposals raise the estimate; a first-token miss lowers it.
        PromptLookup lookup;
        const std::vector<TokenId> context{1, 2, 3, 4, 5, 1, 2};
        propose(lookup, context, 3);
        const float prior = lookup.log_probability();
        lookup.observe(std::vector<TokenId>{3, 4, 5, 9});
        propose(lookup, context, 3);
        const float after_hits = lookup.log_probability();
        lookup.observe(std::vector<TokenId>{8});
        lookup.observe(std::vector<TokenId>{8});
        propose(lookup, context, 3);
        lookup.observe(std::vector<TokenId>{8});
        propose(lookup, context, 3);
        const float after_miss = lookup.log_probability();
        expect(prior < 0.0F && after_hits > prior, "hits raise the estimate");
        expect(after_miss < after_hits, "misses lower the estimate");
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " prompt_lookup\n";
    return failures == 0 ? 0 : 1;
}
