// doorbell_benchmark.cpp — Compare 64-byte doorbell writes on specific NUMA nodes
//
// Build (GCC ≥ 12 or Clang ≥ 15):
//   g++ -O3 -std=c++20 -march=native -mavx512f -mavx512bw -mclflushopt -mmovdir64b \
//       doorbell_benchmark.cpp -o doorbell_bench -lnuma
//
// Usage examples:
//   sudo ./doorbell_bench            # run on node 0 only, 1 M iterations
//   sudo ./doorbell_bench 0 2        # run on nodes 0 & 2 (e.g. local vs. CXL)
//
// Notes
// • Requires libnuma for node-local allocation.
// • MOVDIR64B needs Sapphire Rapids (or newer) *and* Linux ≥ 5.12 with user-mode
//   enablement.  Run as root so the program can pin the thread and read the TSC.
//
#include <immintrin.h>
#include <x86intrin.h>
#include <cpuid.h>
#include <numa.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static inline void clflush_opt(void *p) { _mm_clflushopt(p); }
static inline void sfence()             { _mm_sfence(); }

static inline bool has_movdir64b() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0x07, 0, eax, ebx, ecx, edx);
    return (ecx & (1u << 28)) != 0;
}

static inline void movdir64b(void *dst, const void *src) {
#if defined(__GNUG__) || defined(__clang__)
    __builtin_ia32_movdir64b(dst, src);
#else
    #error "No MOVDIR64B intrinsic available on this compiler"
#endif
}

// --- 8-bit XOR checksum over the first 63 bytes ------------------------------
static inline uint8_t xor_checksum63(const uint8_t *buf) {
    const uint64_t *p = reinterpret_cast<const uint64_t*>(buf);
    uint64_t x = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) x ^= p[i];
    x ^= x >> 32; x ^= x >> 16; x ^= x >> 8;
    return static_cast<uint8_t>(x);
}

static double rdtsc_ghz() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(0x15, eax, ebx, ecx, edx);
    if (eax && ebx) { return static_cast<double>(ecx) * (static_cast<double>(ebx) / eax) / 1e9; }
    return 3.0;   // conservative fallback
}

static void pin_to_cpu0() {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// -----------------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------------
enum class OpType : uint8_t {
    STREAM_FLUSH_SINGLE,
    NT_ONLY_SINGLE,
    NT_ONLY_WITH_CHECKSUM_SINGLE,   // new
    NT_WITH_FLAG_SINGLE,
    MOVDIR_SINGLE,
    MOVDIR_WITH_CHECKSUM_SINGLE,    // new
    STREAM_FLUSH_DOUBLE,
    NT_ONLY_DOUBLE,
    NT_WITH_FLAG_DOUBLE,
    MOVDIR_DOUBLE
};

struct Result {
    OpType   op;
    int      node;
    uint64_t cycles;
    double   ns;
};

constexpr size_t kIters = 1'000'000;
constexpr size_t kLine  = 64;

void benchmark_node(int node, std::vector<Result>& out) {
    alignas(64) uint8_t src[kLine] = {0};      // keep src[63] = 0 for hashing
    size_t      buf_sz = kLine * 2;
    uint8_t    *dst    = static_cast<uint8_t*>(numa_alloc_onnode(buf_sz, node));
    if (!dst) { perror("numa_alloc_onnode"); std::exit(EXIT_FAILURE); }
    memset(dst, 0, buf_sz);  uint8_t *dst2 = dst + kLine;

    for (OpType op : {
            OpType::STREAM_FLUSH_SINGLE,
            OpType::NT_ONLY_SINGLE,
            OpType::NT_ONLY_WITH_CHECKSUM_SINGLE,
            OpType::NT_WITH_FLAG_SINGLE,
            OpType::MOVDIR_SINGLE,
            OpType::MOVDIR_WITH_CHECKSUM_SINGLE,
            OpType::STREAM_FLUSH_DOUBLE,
            OpType::NT_ONLY_DOUBLE,
            OpType::NT_WITH_FLAG_DOUBLE,
            OpType::MOVDIR_DOUBLE }) {

        if ((op == OpType::MOVDIR_SINGLE      ||
             op == OpType::MOVDIR_WITH_CHECKSUM_SINGLE ||
             op == OpType::MOVDIR_DOUBLE) && !has_movdir64b()) {
            std::cerr << "[WARN] CPU lacks MOVDIR64B — skipping\n";
            continue;
        }

        uint64_t t0 = __rdtsc();
        for (size_t i = 0; i < kIters; ++i) {
            switch (op) {
                // ─── single-line variants ─────────────────────────────
                case OpType::STREAM_FLUSH_SINGLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    clflush_opt(dst); sfence(); break;

                case OpType::NT_ONLY_SINGLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence(); break;

                case OpType::NT_ONLY_WITH_CHECKSUM_SINGLE: {
                    uint8_t chk = xor_checksum63(src);
                    src[63] = chk;
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();  src[63] = 0; break;
                }

                case OpType::NT_WITH_FLAG_SINGLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();
                    _mm_stream_si32(reinterpret_cast<int*>(dst), 1);
                    sfence(); break;

                case OpType::MOVDIR_SINGLE:
                    movdir64b(dst, src);
                    sfence(); break;

                case OpType::MOVDIR_WITH_CHECKSUM_SINGLE: {
                    uint8_t chk = xor_checksum63(src);
                    src[63] = chk;
                    movdir64b(dst, src);
                    sfence();  src[63] = 0; break;
                }

                // ─── double-line variants (unchanged) ─────────────────
                case OpType::STREAM_FLUSH_DOUBLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst2),
                                       *reinterpret_cast<const __m512i*>(src));
                    clflush_opt(dst); clflush_opt(dst2); sfence(); break;

                case OpType::NT_ONLY_DOUBLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst2),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence(); break;

                case OpType::NT_WITH_FLAG_DOUBLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst2),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();
                    _mm_stream_si32(reinterpret_cast<int*>(dst), 1);
                    sfence(); break;

                case OpType::MOVDIR_DOUBLE:
                    movdir64b(dst,  src); sfence();
                    movdir64b(dst2, src); sfence(); break;
            }
        }
        uint64_t avg_cycles = (__rdtsc() - t0) / kIters;
        out.push_back({op, node, avg_cycles, 0.0});
    }
    numa_free(dst, buf_sz);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (numa_available() != 0) {
        std::cerr << "libnuma not available or NUMA disabled by BIOS\n";
        return EXIT_FAILURE;
    }
    pin_to_cpu0();

    std::vector<int> nodes;
    if (argc > 1) for (int i = 1; i < argc; ++i) nodes.push_back(std::atoi(argv[i]));
    else nodes = {0};

    std::vector<Result> results;
    for (int n : nodes) benchmark_node(n, results);

    double ghz = rdtsc_ghz();
    std::cout << "Per-operation latency (" << kIters << " iterations, averages)\n";
    std::cout << "Node  Op                       Cycles   ns\n";
    for (auto &r : results) {
        r.ns = r.cycles / ghz;
        const char *name =
            (r.op == OpType::STREAM_FLUSH_SINGLE)          ? "stream+flush_1"  :
            (r.op == OpType::NT_ONLY_SINGLE)               ? "nt_1"            :
            (r.op == OpType::NT_ONLY_WITH_CHECKSUM_SINGLE) ? "nt+chk_1"        :
            (r.op == OpType::NT_WITH_FLAG_SINGLE)          ? "nt+flag_1"       :
            (r.op == OpType::MOVDIR_SINGLE)                ? "movdir_1"        :
            (r.op == OpType::MOVDIR_WITH_CHECKSUM_SINGLE)  ? "movdir+chk_1"    :
            (r.op == OpType::STREAM_FLUSH_DOUBLE)          ? "stream+flush_2"  :
            (r.op == OpType::NT_ONLY_DOUBLE)               ? "nt_2"            :
            (r.op == OpType::NT_WITH_FLAG_DOUBLE)          ? "nt+flag_2"       :
                                                             "movdir_2";
        std::cout << "  " << r.node << "   " << std::left << std::setw(20)
                  << name << "  " << std::setw(8) << r.cycles << "  "
                  << std::fixed << std::setprecision(2) << r.ns << '\n';
    }
    return 0;
}
