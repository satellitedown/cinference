#pragma once

// Speculative verify trees for Op tests. A tree over `nodes` columns lists its nodes in DFS
// pre-order (a node's first child is the next column) with every node's largest subtree visited
// last, the layout the tree-aware verify Ops require. parents[0] is -1.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace ninfer::test {

// chain_bias in [0,1]: probability that a new node extends the previous node.
inline std::vector<std::int32_t> random_verify_tree(int nodes, std::uint32_t seed,
                                                    double chain_bias = 0.5) {
    std::mt19937 rng(seed);
    std::vector<int> parent(static_cast<std::size_t>(nodes), -1);
    for (int node = 1; node < nodes; ++node) {
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        std::uniform_int_distribution<int> pick(0, node - 1);
        parent[static_cast<std::size_t>(node)] = coin(rng) < chain_bias ? node - 1 : pick(rng);
    }
    std::vector<std::vector<int>> children(static_cast<std::size_t>(nodes));
    for (int node = 1; node < nodes; ++node) {
        children[static_cast<std::size_t>(parent[static_cast<std::size_t>(node)])].push_back(node);
    }
    std::vector<int> size(static_cast<std::size_t>(nodes), 1);
    for (int node = nodes - 1; node > 0; --node) {
        size[static_cast<std::size_t>(parent[static_cast<std::size_t>(node)])] +=
            size[static_cast<std::size_t>(node)];
    }
    for (auto& list : children) {
        std::stable_sort(list.begin(), list.end(), [&](int lhs, int rhs) {
            return size[static_cast<std::size_t>(lhs)] < size[static_cast<std::size_t>(rhs)];
        });
    }
    std::vector<std::int32_t> column_of(static_cast<std::size_t>(nodes), -1);
    std::vector<std::int32_t> parents;
    parents.reserve(static_cast<std::size_t>(nodes));
    const std::function<void(int, int)> visit = [&](int node, int parent_column) {
        column_of[static_cast<std::size_t>(node)] = static_cast<std::int32_t>(parents.size());
        parents.push_back(parent_column);
        for (const int child : children[static_cast<std::size_t>(node)]) {
            visit(child, column_of[static_cast<std::size_t>(node)]);
        }
    };
    visit(0, -1);
    return parents;
}

// Root-to-node column path, root first.
inline std::vector<std::int32_t> verify_tree_path(const std::vector<std::int32_t>& parents,
                                                  std::int32_t node) {
    std::vector<std::int32_t> path;
    for (std::int32_t column = node; column >= 0;
         column              = parents[static_cast<std::size_t>(column)]) {
        path.push_back(column);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline std::uint32_t verify_tree_ancestor_mask(const std::vector<std::int32_t>& parents,
                                               std::int32_t node) {
    std::uint32_t mask = 0;
    for (const std::int32_t column : verify_tree_path(parents, node)) { mask |= 1U << column; }
    return mask;
}

} // namespace ninfer::test
