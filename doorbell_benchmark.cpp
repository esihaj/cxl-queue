// doorbell_benchmark.cpp — 64-byte door-bell micro-benchmark
//
// One allocator is created at program start and reused for all tests.
// The allocator type is selected by CLI:
//
//   dax               → default /dev/dax1.0 (81-82 GiB slice)
//   numa <node>       → arena from DRAM on <node>
//
// Build (GCC ≥ 12 or Clang ≥ 15)
//   g++ -O3 -std=c++20 -march=native -mavx512f -mavx512bw \
//       -mclflushopt -mclwb -mmovdir64b doorbell_benchmark.cpp -o doorbell_bench -lnuma
// ---------------------------------------------------------------------------

#include "cxl_allocator.hpp"        // allocator header
#include <immintrin.h>
#include <x86intrin.h>
#include <cpuid.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

/* ─ helpers ────────────────────────────────────────────────────────────── */
static inline void clflush(void* p)      { _mm_clflush(p);    }
static inline void clflush_opt(void* p)  { _mm_clflushopt(p); }
static inline void clwb(void* p)         { _mm_clwb(p);       }
static inline void sfence()              { _mm_sfence();      }

static inline bool has_movdir64b()
{
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(0x07, 0, eax, ebx, ecx, edx);
    return (ecx & (1u << 28)) != 0;
}

static inline void movdir64b(void* dst, const void* src)
{
#if defined(__GNUG__) || defined(__clang__)
    __builtin_ia32_movdir64b(dst, src);
#else
#   error "No MOVDIR64B intrinsic available on this compiler"
#endif
}

using u64_may_alias = uint64_t __attribute__((may_alias));
static inline uint16_t xor_checksum64(const void* p) noexcept
{
    const u64_may_alias* u = reinterpret_cast<const u64_may_alias*>(p);
    uint64_t acc = 0;
    for (int i = 0; i < 8; ++i) acc ^= u[i];
    acc = (acc >> 32) ^ (acc & 0xFFFFFFFFULL);
    acc = (acc >> 16) ^ (acc & 0xFFFFULL);
    return static_cast<uint16_t>(acc);
}

static double rdtsc_ghz()
{
    unsigned int eax, ebx, ecx, edx;
    __cpuid(0x15, eax, ebx, ecx, edx);
    if (eax && ebx)
        return static_cast<double>(ecx) * (static_cast<double>(ebx) / eax) / 1e9;
    return 3.0;     // fallback
}

static void pin_to_cpu0()
{
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(0, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

/* ─ benchmark definitions ─────────────────────────────────────────────── */
enum class OpType : uint8_t {
    REG_CLFLUSH_SINGLE,
    REG_CLFLUSHOPT_SINGLE,
    REG_CLWB_SINGLE,
    SCALAR8_CLWB_SINGLE,
    NT_STREAM_SINGLE,
    NT_STREAM_CHECKSUM_SINGLE,
    NT_STREAM_FLAG_SINGLE,
    NT_LOAD_SINGLE,
    MOVDIR_SINGLE,
    MOVDIR_CHECKSUM_SINGLE,
    REG_CLFLUSHOPT_DOUBLE,
    NT_STREAM_DOUBLE,
    NT_STREAM_FLAG_DOUBLE,
    MOVDIR_DOUBLE
};

struct Result {
    OpType   op;
    uint64_t cycles;
    double   ns;
};

constexpr std::size_t k_iters = 1'000'000;
constexpr std::size_t k_line  = 64;

static uint64_t nt_load_checksum = 0;

/* ─ single benchmark run ──────────────────────────────────────────────── */
void benchmark(cxl::CxlAllocator& alloc, std::vector<Result>& out)
{
    alignas(64) uint8_t src[k_line] = {0};
    alignas(64) uint8_t tmp[k_line] = {0};

    uint8_t* dst  = static_cast<uint8_t*>(alloc.allocate_aligned(k_line));
    uint8_t* dst2 = static_cast<uint8_t*>(alloc.allocate_aligned(k_line));

    for (OpType op :
        { OpType::REG_CLFLUSH_SINGLE,
          OpType::REG_CLFLUSHOPT_SINGLE,
          OpType::REG_CLWB_SINGLE,
          OpType::SCALAR8_CLWB_SINGLE,
          OpType::NT_STREAM_SINGLE,
          OpType::NT_STREAM_CHECKSUM_SINGLE,
          OpType::NT_STREAM_FLAG_SINGLE,
          OpType::NT_LOAD_SINGLE,
          OpType::MOVDIR_SINGLE,
          OpType::MOVDIR_CHECKSUM_SINGLE,
          OpType::REG_CLFLUSHOPT_DOUBLE,
          OpType::NT_STREAM_DOUBLE,
          OpType::NT_STREAM_FLAG_DOUBLE,
          OpType::MOVDIR_DOUBLE })
    {
        if ((op == OpType::MOVDIR_SINGLE ||
             op == OpType::MOVDIR_CHECKSUM_SINGLE ||
             op == OpType::MOVDIR_DOUBLE) && !has_movdir64b())
        {
            std::cerr << "[WARN] CPU lacks MOVDIR64B — skipping\n";
            continue;
        }

        uint64_t start = __rdtsc();
        for (std::size_t i = 0; i < k_iters; ++i) {
            switch (op) {
                case OpType::REG_CLFLUSH_SINGLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    clflush(dst); sfence(); break;

                case OpType::REG_CLFLUSHOPT_SINGLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    clflush_opt(dst); sfence(); break;

                case OpType::REG_CLWB_SINGLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    clwb(dst); sfence(); break;

                case OpType::SCALAR8_CLWB_SINGLE: {
                    const uint64_t* s64 = reinterpret_cast<const uint64_t*>(src);
                    uint64_t*       d64 = reinterpret_cast<uint64_t*>(dst);
#pragma unroll
                    for (int j = 0; j < 8; ++j) d64[j] = s64[j];
                    clwb(dst); sfence(); break;
                }

                case OpType::NT_STREAM_SINGLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence(); break;

                case OpType::NT_STREAM_CHECKSUM_SINGLE: {
                    uint16_t chk = xor_checksum64(src);
                    src[62] = static_cast<uint8_t>(chk);       // low byte
                    src[63] = static_cast<uint8_t>(chk >> 8);  // high byte
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();
                    src[62] = src[63] = 0;
                    break;
                }

                case OpType::NT_STREAM_FLAG_SINGLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();
                    _mm_stream_si32(reinterpret_cast<int*>(dst), 1);
                    sfence(); break;

                case OpType::NT_LOAD_SINGLE: {
                    clflush_opt(dst); sfence();
                    __m512i v = _mm512_stream_load_si512(dst);
                    _mm512_storeu_si512(reinterpret_cast<void*>(tmp), v);
                    nt_load_checksum += xor_checksum64(tmp);
                    break;
                }

                case OpType::MOVDIR_SINGLE:
                    movdir64b(dst, src); sfence(); break;

                case OpType::MOVDIR_CHECKSUM_SINGLE: {
                    uint16_t chk = xor_checksum64(src);
                    src[62] = static_cast<uint8_t>(chk);
                    src[63] = static_cast<uint8_t>(chk >> 8);
                    movdir64b(dst, src);
                    src[62] = src[63] = 0;
                    sfence(); break;
                }

                case OpType::REG_CLFLUSHOPT_DOUBLE:
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst),
                                       *reinterpret_cast<const __m512i*>(src));
                    _mm512_store_si512(reinterpret_cast<__m512i*>(dst2),
                                       *reinterpret_cast<const __m512i*>(src));
                    clflush_opt(dst); clflush_opt(dst2); sfence(); break;

                case OpType::NT_STREAM_DOUBLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst2),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence(); break;

                case OpType::NT_STREAM_FLAG_DOUBLE:
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst),
                                        *reinterpret_cast<const __m512i*>(src));
                    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst2),
                                        *reinterpret_cast<const __m512i*>(src));
                    sfence();
                    _mm_stream_si32(reinterpret_cast<int*>(dst), 1);
                    sfence(); break;

                case OpType::MOVDIR_DOUBLE:
                    movdir64b(dst, src);  sfence();
                    movdir64b(dst2, src); sfence(); break;
            }
        }
        uint64_t avg_cycles = (__rdtsc() - start) / k_iters;
        out.push_back({op, avg_cycles, 0.0});
    }
}

/* ─ main ──────────────────────────────────────────────────────────────── */
int main(int argc, char** argv)
{
    auto usage = [prog = argv[0]]() {
        std::cerr << "Usage:\n"
                  << "  " << prog << " dax\n"
                  << "  " << prog << " numa <alloc_node>\n";
    };

    if (numa_available() != 0) {
        std::cerr << "libnuma not available or NUMA disabled by BIOS\n";
        return EXIT_FAILURE;
    }
    pin_to_cpu0();

    if (argc < 2) {
        std::cerr << "Error: missing allocator type\n";
        usage();
        return EXIT_FAILURE;
    }

    std::unique_ptr<cxl::CxlAllocator> allocator;
    const std::string mode = argv[1];

    if (mode == "dax") {
        allocator = std::make_unique<cxl::DaxAllocator>();
    } else if (mode == "numa") {
        if (argc < 3) { std::cerr << "Error: missing NUMA node\n"; usage(); return EXIT_FAILURE; }
        int alloc_node = std::atoi(argv[2]);
        std::size_t arena_size = k_line * 2;              // two cache lines
        allocator = std::make_unique<cxl::NumaAllocator>(alloc_node, arena_size,
                                                         cxl::DebugLevel::low);
    } else {
        std::cerr << "Error: unknown allocator type '" << mode << "'\n";
        usage();
        return EXIT_FAILURE;
    }

    if (!allocator->test_memory()) {
        std::cerr << "Allocator self-test failed\n";
        return EXIT_FAILURE;
    }

    /* run benchmark */
    std::vector<Result> results;
    benchmark(*allocator, results);

    /* print results */
    double ghz = rdtsc_ghz();
    std::cout << "Per-operation latency (" << k_iters
              << " iterations, averages)\n\n"
              << "Operation                                   Cycles      ns\n\n";

    bool prev_double = false;
    for (auto& r : results) {
        r.ns = r.cycles / ghz;

        const char* name =
            (r.op == OpType::REG_CLFLUSH_SINGLE)        ? "64B_regular_store+clflush"               :
            (r.op == OpType::REG_CLFLUSHOPT_SINGLE)     ? "64B_regular_store+clflushopt"            :
            (r.op == OpType::REG_CLWB_SINGLE)           ? "64B_regular_store+clwb"                  :
            (r.op == OpType::SCALAR8_CLWB_SINGLE)       ? "8x8B_scalar_store+clwb"                  :
            (r.op == OpType::NT_STREAM_SINGLE)          ? "64B_non_temporal_stream"                 :
            (r.op == OpType::NT_STREAM_CHECKSUM_SINGLE) ? "64B_non_temporal_stream+checksum"        :
            (r.op == OpType::NT_STREAM_FLAG_SINGLE)     ? "64B_non_temporal_stream+flag"            :
            (r.op == OpType::NT_LOAD_SINGLE)            ? "64B_non_temporal_stream_load"            :
            (r.op == OpType::MOVDIR_SINGLE)             ? "movdir64B"                               :
            (r.op == OpType::MOVDIR_CHECKSUM_SINGLE)    ? "movdir64B+checksum"                      :
            (r.op == OpType::REG_CLFLUSHOPT_DOUBLE)     ? "2x64B_regular_store+clflushopt"          :
            (r.op == OpType::NT_STREAM_DOUBLE)          ? "2x64B_non_temporal_stream"               :
            (r.op == OpType::NT_STREAM_FLAG_DOUBLE)     ? "2x64B_non_temporal_stream+flag"          :
            (r.op == OpType::MOVDIR_DOUBLE)             ? "2xmovdir64B"                             :
                                                           "unknown";

        bool is_double =
            (r.op == OpType::REG_CLFLUSHOPT_DOUBLE) ||
            (r.op == OpType::NT_STREAM_DOUBLE)      ||
            (r.op == OpType::NT_STREAM_FLAG_DOUBLE) ||
            (r.op == OpType::MOVDIR_DOUBLE);

        if (is_double && !prev_double) std::cout << '\n';
        prev_double = is_double;

        std::cout << std::left << std::setw(42) << name << "  "
                  << std::setw(10) << r.cycles << "  "
                  << std::fixed << std::setprecision(2) << r.ns << '\n';
    }

    std::cout << "\n  nt_load_checksum (ignore. Just to prevent compiler optimizations): " << nt_load_checksum << '\n';
    
    return 0;
}