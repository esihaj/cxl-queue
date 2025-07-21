/*
 * Serialization/Deserialization Latency Benchmark — Complex Payloads
 * ------------------------------------------------------------------
 * Measures average per‑operation latency (microseconds) for
 * • complex heterogeneous flat objects, and
 * • nested tree structures based on a specific progression of node counts.
 *
 * Tree structures are generated with a balanced shape for each node count.
 *
 * Library‑agnostic via `JsonLibrary`; default implementation uses
 * **nlohmann::json**. Swap codecs by subclassing and re‑compiling.
 *
 * Build (example):
 * g++ -O3 -std=c++20 -march=native -I/path/to/nlohmann \
 * serialization_benchmark_final.cpp -o ser_bench
 * Run:
 * ./ser_bench
 */

#include "nlohmann_json.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <functional> // Required for std::function
#include <algorithm>  // Required for std::min/max
#include <deque>      // Used for BFS tree generation

// ────────────────────────────────────────────────────────────────
//  RNG helpers
// ────────────────────────────────────────────────────────────────
static std::mt19937_64 rng{std::random_device{}()};
static std::string random_ascii(std::size_t len) {
    static constexpr char alphabet[] =
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::uniform_int_distribution<std::size_t> dist(0, sizeof(alphabet) - 2);
    std::string s;
    s.reserve(len);
    while (s.size() < len) s.push_back(alphabet[dist(rng)]);
    return s;
}
static std::vector<uint8_t> random_blob(std::size_t len) {
    std::uniform_int_distribution<uint8_t> dist(0, 255);
    std::vector<uint8_t> v(len);
    for (auto& b : v) b = dist(rng);
    return v;
}

// ────────────────────────────────────────────────────────────────
//  Complex flat payload
// ────────────────────────────────────────────────────────────────
struct ComplexPayload {
    int32_t               id;
    std::string           name;
    double                score;
    bool                  active;
    std::vector<int64_t>  values;
    std::vector<uint8_t>  data;   // dominant size component
};

inline void to_json(nlohmann::json& j, const ComplexPayload& p) {
    j = { {"id", p.id}, {"name", p.name}, {"score", p.score},
          {"active", p.active}, {"values", p.values}, {"data", p.data} };
}
inline void from_json(const nlohmann::json& j, ComplexPayload& p) {
    j.at("id").get_to(p.id);
    j.at("name").get_to(p.name);
    j.at("score").get_to(p.score);
    j.at("active").get_to(p.active);
    j.at("values").get_to(p.values);
    j.at("data").get_to(p.data);
}

// ────────────────────────────────────────────────────────────────
//  Tree payload (nested)
// ────────────────────────────────────────────────────────────────
struct TreeNode {
    int32_t              id;
    std::string          label;
    std::vector<uint8_t> blob;
    std::vector<TreeNode> children;

    // Default constructor
    TreeNode() : id(0) {}

    // Move constructor for efficiency
    TreeNode(TreeNode&& other) noexcept
        : id(other.id),
          label(std::move(other.label)),
          blob(std::move(other.blob)),
          children(std::move(other.children)) {}

    // Move assignment operator
    TreeNode& operator=(TreeNode&& other) noexcept {
        if (this != &other) {
            id = other.id;
            label = std::move(other.label);
            blob = std::move(other.blob);
            children = std::move(other.children);
        }
        return *this;
    }

    // Disable copy operations to prevent accidental performance issues
    TreeNode(const TreeNode&) = delete;
    TreeNode& operator=(const TreeNode&) = delete;
};

inline void to_json(nlohmann::json& j, const TreeNode& n) {
    j = { {"id", n.id}, {"label", n.label}, {"blob", n.blob},
          {"children", n.children} };
}
inline void from_json(const nlohmann::json& j, TreeNode& n) {
    j.at("id").get_to(n.id);
    j.at("label").get_to(n.label);
    j.at("blob").get_to(n.blob);
    j.at("children").get_to(n.children);
}

/**
 * @brief Creates a tree with a specified structure.
 *
 * This function builds a tree with a target number of nodes, respecting
 * constraints on maximum depth and children per node. It uses a breadth-first
 * strategy to add nodes level by level.
 *
 * @param total_nodes The target number of nodes for the entire tree.
 * @param max_depth The maximum depth of the tree.
 * @param max_children The maximum number of children any single node can have.
 * @param node_data_size The size of the data blob in each node.
 * @return The root node of the generated tree.
 */
TreeNode make_tree_by_structure(int total_nodes, int max_depth, int max_children, size_t node_data_size) {
    if (total_nodes <= 0) return {};

    std::uniform_int_distribution<int32_t> id_dist(0, 1'000'000);
    std::uniform_int_distribution<int> child_count_dist(1, std::max(1, max_children));
    const size_t label_size = 16;
    
    TreeNode root;
    root.id = id_dist(rng);
    root.label = random_ascii(label_size);
    root.blob = random_blob(node_data_size);

    int nodes_created = 1;
    if (nodes_created >= total_nodes) return root;

    // Queue of {parent_node, current_depth} for BFS construction
    std::deque<std::pair<TreeNode*, int>> parent_queue;
    parent_queue.push_back({&root, 1});

    while (nodes_created < total_nodes && !parent_queue.empty()) {
        auto [parent_node, current_depth] = parent_queue.front();
        parent_queue.pop_front();

        if (current_depth >= max_depth) continue;

        // Determine how many children to add to this parent
        int num_children_to_add = child_count_dist(rng);
        // Ensure we don't exceed the total node count
        num_children_to_add = std::min(num_children_to_add, total_nodes - nodes_created);

        parent_node->children.reserve(num_children_to_add);
        for (int i = 0; i < num_children_to_add; ++i) {
            TreeNode new_node;
            new_node.id = id_dist(rng);
            new_node.label = random_ascii(label_size);
            new_node.blob = random_blob(node_data_size);
            
            parent_node->children.push_back(std::move(new_node));
            nodes_created++;
        }
        
        // Add the newly created children to the queue to become parents themselves
        for (auto& child : parent_node->children) {
            if (nodes_created >= total_nodes) break;
            parent_queue.push_back({&child, current_depth + 1});
        }
    }

    return root;
}


// ────────────────────────────────────────────────────────────────
//  Size/Node Calculation Helpers
// ────────────────────────────────────────────────────────────────

// Calculates the actual in-memory size of a flat payload.
std::size_t calculate_flat_size(const ComplexPayload& p) {
    std::size_t current_size = sizeof(p);
    current_size += p.name.capacity();
    current_size += p.values.capacity() * sizeof(int64_t);
    current_size += p.data.capacity() * sizeof(uint8_t);
    return current_size;
}

// Recursively calculates the actual in-memory size of a tree.
std::size_t calculate_tree_size(const TreeNode& n) {
    std::size_t current_size = sizeof(n);
    current_size += n.label.capacity();
    current_size += n.blob.capacity();
    current_size += n.children.capacity() * sizeof(TreeNode);
    
    for (const auto& child : n.children) {
        current_size += calculate_tree_size(child);
    }
    return current_size;
}

// Recursively counts the number of nodes in a tree.
std::size_t count_nodes(const TreeNode& n) {
    // An empty node (from a failed creation) shouldn't be counted.
    if (n.label.empty() && n.id == 0) return 0;
    std::size_t count = 1; // Count this node
    for (const auto& child : n.children) {
        count += count_nodes(child);
    }
    return count;
}


// ────────────────────────────────────────────────────────────────
//  Library‑agnostic interface
// ────────────────────────────────────────────────────────────────
class JsonLibrary {
public:
    virtual ~JsonLibrary() = default;
    virtual std::string    serialize(const ComplexPayload&) = 0;
    virtual std::string    serialize(const TreeNode&)       = 0;
    virtual ComplexPayload deserialize_complex(const std::string&) = 0;
    virtual TreeNode       deserialize_tree(const std::string&)    = 0;
};

class NlohmannJsonLib final : public JsonLibrary {
public:
    std::string serialize(const ComplexPayload& p) override {
        return nlohmann::json(p).dump();
    }
    std::string serialize(const TreeNode& n) override {
        return nlohmann::json(n).dump();
    }
    ComplexPayload deserialize_complex(const std::string& s) override {
        return nlohmann::json::parse(s).get<ComplexPayload>();
    }
    TreeNode deserialize_tree(const std::string& s) override {
        return nlohmann::json::parse(s).get<TreeNode>();
    }
};

// ────────────────────────────────────────────────────────────────
//  Timing helper
// ────────────────────────────────────────────────────────────────
using HighResClock = std::chrono::high_resolution_clock;

template <typename F>
static double average_ns(F&& fn, std::size_t iterations) {
    const auto t0 = HighResClock::now();
    for (std::size_t i = 0; i < iterations; ++i) fn();
    const auto t1 = HighResClock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / iterations;
}

// ────────────────────────────────────────────────────────────────
//  Main benchmark
// ────────────────────────────────────────────────────────────────
int main() {
    std::unique_ptr<JsonLibrary> lib = std::make_unique<NlohmannJsonLib>();

    std::cout << std::fixed << std::setprecision(2);
    
    // --- Iteration Info ---
    std::cout << "\n--- Iteration Counts ---\n";
    std::cout << "Flat Objects: 100,000 (for sizes < 16KiB), 10,000 (for sizes >= 16KiB)\n";
    std::cout << "Trees: 50,000 (< 64 nodes), 10,000 (64-511 nodes), 1,000 (>= 512 nodes)\n";

    // --- Flat Object Benchmark ---
    struct SizeCase { const char* label; std::size_t bytes; };
    const std::vector<SizeCase> flat_cases = {
        {"64B",   64}, {"256B",  256}, {"512B",  512}, {"1KiB",  1024},
        {"4KiB",  4 * 1024}, {"16KiB", 16 * 1024}, {"64KiB", 64 * 1024}
    };
    std::cout << "\n--- Flat Object Benchmark (nlohmann::json) ---\n";
    std::cout << "Target Size\tActual Size\tIterations\tSerialize (μs)\tDeserialize (μs)\n";
    std::cout << "-----------\t-----------\t----------\t--------------\t----------------\n";
    for (const auto& [label, bytes] : flat_cases) {
        const std::size_t iters = bytes >= 16 * 1024 ? 10'000 : 100'000;

        ComplexPayload flat;
        flat.id = 42;
        flat.name = random_ascii(16);
        flat.score = 2.71828;
        flat.active = true;
        flat.values = {1,2,3,4,5,6,7};
        const size_t flat_overhead = calculate_flat_size(flat);
        flat.data = random_blob(bytes > flat_overhead ? bytes - flat_overhead : 0);

        const double ser_flat_ns = average_ns([&]{ lib->serialize(flat); }, iters);
        const std::string flat_str = lib->serialize(flat);
        const double des_flat_ns = average_ns([&]{ lib->deserialize_complex(flat_str); }, iters);
        
        std::cout << label << "\t\t" << calculate_flat_size(flat) << "B\t\t" << iters << "\t\t" 
                  << ser_flat_ns / 1000.0 << "\t\t" << des_flat_ns / 1000.0 << "\n";
    }

    // --- Tree Benchmark ---
    struct TreeStructureCase {
        const char* label;
        int total_nodes;
        int max_depth;
        int max_children;
        size_t node_data_size;
    };
    const std::vector<TreeStructureCase> tree_cases = {
        {"1 Node",       1,   2, 2, 128},
        {"4 Nodes",      4,   3, 3, 128},
        {"8 Nodes",      8,   4, 3, 128},
        {"32 Nodes",    32,   5, 4, 128},
        {"64 Nodes",    64,   6, 4, 128},
        {"128 Nodes",  128,   7, 5, 128},
        {"256 Nodes",  256,   8, 5, 128},
        {"512 Nodes",  512,   9, 6, 128},
        {"1000 Nodes", 1000, 10, 7, 128},
    };

    std::cout << "\n--- Tree Benchmark (nlohmann::json) ---\n";
    std::cout << "Structure \tNodes\tTotal Tree Size (KiB)\tIterations\tSerialize (μs)\tDeserialize (μs)\n";
    std::cout << "----------\t-----\t---------------------\t----------\t--------------\t----------------\n";
    for (const auto& tc : tree_cases) {
        const std::size_t iters = tc.total_nodes >= 512 ? 1000 : (tc.total_nodes >= 64 ? 10'000 : 50'000);

        TreeNode tree = make_tree_by_structure(tc.total_nodes, tc.max_depth, tc.max_children, tc.node_data_size);
        const double ser_tree_ns = average_ns([&]{ lib->serialize(tree); }, iters);
        const std::string tree_str = lib->serialize(tree);
        const double des_tree_ns = average_ns([&]{ lib->deserialize_tree(tree_str); }, iters);
        
        std::cout << tc.label << "\t" << count_nodes(tree) << "\t"
                  << static_cast<double>(calculate_tree_size(tree)) / 1024.0 << "\t\t\t"
                  << iters << "\t\t" << ser_tree_ns / 1000.0 << "\t\t" << des_tree_ns / 1000.0 << "\n";
    }

    return 0;
}
