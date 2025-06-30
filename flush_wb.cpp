// flush_page_bench.cpp
#include <x86intrin.h>   // _mm_clwb, _mm_clflushopt, _mm_clflush, _mm_mfence
#include <numa.h>        // numa_alloc_onnode, numa_free, numa_available
#include <ctime>         // clock_gettime
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr std::size_t k_line_bytes = 64;
constexpr std::size_t k_iters      = 100000;

// All three flush methods
enum class flush_method { clwb, clflushopt, clflush };

// Serialize execution before timing
static inline void cpuid_barrier() {
    asm volatile("cpuid" ::: "rax","rbx","rcx","rdx","memory");
}

// Read the timestamp counter (cycles)
static inline uint64_t rdtscp_cycles() {
    uint32_t lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return (uint64_t(hi) << 32) | lo;
}

// Issue a single cache‐line flush
template <flush_method M>
static inline void flush_line(void *p);

template <>
inline void flush_line<flush_method::clwb>(void *p) {
    _mm_clwb(p);
}

template <>
inline void flush_line<flush_method::clflushopt>(void *p) {
    _mm_clflushopt(p);
}

template <>
inline void flush_line<flush_method::clflush>(void *p) {
    _mm_clflush(p);
}

// Benchmark one page buffer of size `page_bytes` with method M
template <flush_method M>
void bench_one(unsigned char *page,
               std::size_t page_bytes,
               const char *method_name)
{
    const std::size_t lines = page_bytes / k_line_bytes;
    uint64_t total_cycles = 0, total_ns = 0;
    struct timespec ts0{}, ts1{};

    for (std::size_t iter = 0; iter < k_iters; ++iter) {
        // 1) Touch & dirty each line
        for (std::size_t i = 0; i < lines; ++i)
            page[i * k_line_bytes]++;

        cpuid_barrier();
        uint64_t c0 = rdtscp_cycles();
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts0);

        // 2) Flush each line
        for (std::size_t i = 0; i < lines; ++i)
            flush_line<M>(page + i * k_line_bytes);

        // 3) Full fence to wait for write-backs
        _mm_mfence();

        clock_gettime(CLOCK_MONOTONIC_RAW, &ts1);
        uint64_t c1 = rdtscp_cycles();

        total_cycles += (c1 - c0);
        total_ns     += uint64_t(ts1.tv_sec  - ts0.tv_sec ) * 1'000'000'000ULL
                      + uint64_t(ts1.tv_nsec - ts0.tv_nsec);
    }

    double avg_c = double(total_cycles) / k_iters;
    double avg_n = double(total_ns)     / k_iters;

    std::cout << std::left
              << std::setw(10) << method_name
              << " | " << std::setw(4) << (page_bytes/1024) << " KiB"
              << " → " << std::fixed << std::setprecision(1)
              << std::setw(8) << avg_c << " cycles, "
              << std::setw(8) << avg_n << " ns\n";
}

int main(int argc, char **argv) {
    if (numa_available() < 0) {
        std::cerr << "Error: NUMA is not available on this system\n";
        return 1;
    }
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <node0> [node1] ...\n";
        return 1;
    }

    // Parse nodes
    std::vector<int> nodes;
    for (int i = 1; i < argc; ++i)
        nodes.push_back(std::stoi(argv[i]));

    // We want to test both 4 KiB and 8 KiB
    const std::vector<std::size_t> page_sizes = { 4096, 8192 };

    std::cout << "NUMA‐aware flush benchmark (" << k_iters << " iters)\n"
              << "method     | size KiB → cycles,    ns\n"
              << "--------------------------------------\n";

    for (int node : nodes) {
        std::cout << "=== NUMA node " << node << " ===\n";

        for (auto sz : page_sizes) {
            // Allocate on that node
            auto page = (unsigned char*)numa_alloc_onnode(sz, node);
            if (!page) {
                std::cerr << "  [node " << node << "] allocation of "
                          << sz << " bytes failed\n";
                continue;
            }
            std::memset(page, 0, sz);

            // Run all three methods
            bench_one<flush_method::clwb>      (page, sz, "CLWB");
            bench_one<flush_method::clflushopt>(page, sz, "CLFLUSHOPT");
            bench_one<flush_method::clflush>   (page, sz, "CLFLUSH");

            numa_free(page, sz);
            std::cout << "--------------------------------------\n";
        }
    }

    return 0;
}

