/*
 * Serialization/Deserialization Latency Benchmark — Complex Payloads
 * ------------------------------------------------------------------
 * Measures average per‑operation latency (nanoseconds) for
 * • complex heterogeneous flat objects, and
 * • nested tree structures of a target total size.
 * Payload sizes: 64 B, 256 B, 512 B, 1 KiB, 2 KiB,
 * 4 KiB, 8 KiB, 32 KiB, 1 MiB.
 *
 * Library‑agnostic via `JsonLibrary`; default implementation uses
 * **nlohmann::json**. Swap codecs by subclassing and re‑compiling.
 *
 * Build (example):
 * g++ -O3 -std=c++20 -march=native -I/path/to/nlohmann \
 * serialization_benchmark.cpp -o ser_bench
 * Run:
 * ./ser_bench
 */

#include <nlohmann/json.hpp>
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
    std::vector<uint8_t> blob; // Using uint8_t to match random_blob
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
 * @brief Creates a tree with a total memory footprint approximating the target size.
 *
 * This function is designed for benchmarking. It iteratively adds nodes only if
 * the budget allows, creating a representative structure for the target size.
 *
 * @param total_bytes The desired total size of the tree in bytes.
 * @return The root node of the generated tree.
 */
TreeNode make_tree_of_size(std::size_t total_bytes) {
    // --- Constants ---
    const std::size_t label_size = 8;
    const int max_nodes_for_structure = 100;

    std::vector<TreeNode> node_pool;
    std::size_t remaining_budget = total_bytes;

    // --- Iterative Node Creation ---
    // This loop accurately budgets by measuring the actual overhead of each node
    // before committing it to the tree structure.
    while (node_pool.size() < max_nodes_for_structure) {
        // Create a temporary node to measure its real overhead.
        TreeNode temp_node;
        temp_node.label = random_ascii(label_size);
        
        // The actual overhead includes the object size and the heap-allocated string capacity.
        const std::size_t actual_overhead = sizeof(TreeNode) + temp_node.label.capacity();

        if (remaining_budget >= actual_overhead) {
            remaining_budget -= actual_overhead;
            node_pool.push_back(std::move(temp_node));
        } else {
            // Not enough budget for another node.
            break;
        }
    }
    
    // If the budget was too small for even one node, ensure we have at least a root.
    if (node_pool.empty()) {
        node_pool.emplace_back();
        node_pool[0].label = random_ascii(label_size);
        remaining_budget = 0;
    }

    // --- Memory Distribution ---
    // Distribute the final remaining budget as blob data across all created nodes.
    const std::size_t blob_each = node_pool.empty() ? 0 : remaining_budget / node_pool.size();

    // --- Tree Construction ---
    std::uniform_int_distribution<int32_t> dist32(0, 1'000'000);
    
    // Configure all nodes in the pool with ID and blob data (label is already set).
    for(auto& node : node_pool) {
        node.id = dist32(rng);
        node.blob = random_blob(blob_each);
    }
    
    // Assemble the tree: the first node is the root, the rest are its children.
    TreeNode root = std::move(node_pool[0]);
    if (node_pool.size() > 1) {
        root.children.reserve(node_pool.size() - 1);
        // Move the remaining nodes from the pool to be children of the root.
        for (size_t i = 1; i < node_pool.size(); ++i) {
            root.children.push_back(std::move(node_pool[i]));
        }
    }

    return root;
}


// ────────────────────────────────────────────────────────────────
//  Size Calculation Helpers
// ────────────────────────────────────────────────────────────────

// Calculates the actual in-memory size of a flat payload.
std::size_t calculate_flat_size(const ComplexPayload& p) {
    std::size_t current_size = sizeof(p);
    current_size += p.name.capacity();
    current_size += p.values.capacity() * sizeof(int64_t);
    current_size += p.data.capacity() * sizeof(uint8_t);
    return current_size;
}


// Calculates the actual in-memory size of a tree.
std::size_t calculate_tree_size(const TreeNode& n) {
    std::size_t current_size = sizeof(n);
    current_size += n.label.capacity();
    current_size += n.blob.capacity();
    
    for (const auto& child : n.children) {
        current_size += calculate_tree_size(child);
    }
    current_size += n.children.capacity() * sizeof(TreeNode);
    return current_size;
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
//  Timing helper — use alias distinct from libc clock_t
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

    struct SizeCase { const char* label; std::size_t bytes; };
    const std::vector<SizeCase> cases = {
        {"64B",   64},
        {"256B",  256},
        {"512B",  512},
        {"1KiB",  1024},
        {"2KiB",  2048},
        {"4KiB",  4 * 1024},
        {"8KiB",  8 * 1024},
        {"32KiB", 32 * 1024},
        {"1MiB",  1 * 1024 * 1024}
    };

    std::cout << std::fixed << std::setprecision(1);

    // --- Flat Object Benchmark ---
    std::cout << "\n--- Flat Object Benchmark (nlohmann::json) ---\n";
    std::cout << "Target\tActual\t\tSerialize\tDeserialize\n";
    std::cout << "------\t------\t\t---------\t-----------\n";
    for (const auto& [label, bytes] : cases) {
        const std::size_t iters = bytes >= 32 * 1024 ? 1000 : (bytes <= 2048 ? 100'000 : 10'000);

        ComplexPayload flat;
        flat.id = 42;
        flat.name = random_ascii(16);
        flat.score = 2.71828;
        flat.active = true;
        flat.values = {1,2,3,4,5,6,7};
        // Adjust data size to approximate the target size
        const size_t flat_overhead = calculate_flat_size(flat);
        flat.data = random_blob(bytes > flat_overhead ? bytes - flat_overhead : 0);

        const double ser_flat = average_ns([&]{ lib->serialize(flat); }, iters);
        const std::string flat_str = lib->serialize(flat);
        const double des_flat = average_ns([&]{ lib->deserialize_complex(flat_str); }, iters);
        
        std::cout << label << "\t" << calculate_flat_size(flat) << "B\t\t" << ser_flat << "\t\t" << des_flat << "\n";
    }

    // --- Tree Benchmark ---
    std::cout << "\n--- Tree Benchmark (nlohmann::json) ---\n";
    std::cout << "Target\tActual\t\tNodes\tSerialize\tDeserialize\n";
    std::cout << "------\t------\t\t-----\t---------\t-----------\n";
    for (const auto& [label, bytes] : cases) {
        const std::size_t iters = bytes >= 32 * 1024 ? 1000 : (bytes <= 2048 ? 100'000 : 10'000);

        TreeNode tree = make_tree_of_size(bytes);
        const double ser_tree = average_ns([&]{ lib->serialize(tree); }, iters);
        const std::string tree_str = lib->serialize(tree);
        const double des_tree = average_ns([&]{ lib->deserialize_tree(tree_str); }, iters);
        
        std::cout << label << "\t" << calculate_tree_size(tree) << "B\t\t" 
                  << (1 + tree.children.size()) << "\t"
                  << ser_tree << "\t\t" << des_tree << "\n";
    }

    return 0;
}
