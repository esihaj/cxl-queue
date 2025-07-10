/*
 * backoff_bench.cpp
 *
 * Measure the real-time cost of software back-off schedules.
 *   – Works on x86-64 / clang++ or g++ (Linux, macOS).
 *   – No external dependencies.
 *
 * Compile:
 *   g++ -O3 -std=c++20 -march=native backoff_bench.cpp -o backoff_bench
 *
 * Run:
 *   ./backoff_bench          # prints one table per configuration
 *
 * The program prints:  pause-slot,   programmed cycles,   median real cycles,
 *                      and the same converted to nanoseconds (ns)
 *                      assuming the invariant-TSC frequency detected at start-up.
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cpuid.h>
#include <x86intrin.h>   // rdtsc, _mm_pause

// -----------------------------------------------------------------------------
// low-level helpers  -----------------------------------------------------------
// -----------------------------------------------------------------------------

static inline uint64_t rdtsc_start() {
    unsigned junk;
    _mm_mfence();
    __cpuid(0, junk, junk, junk, junk);      // serialise
    return __rdtsc();
}

static inline uint64_t rdtsc_end() {
    _mm_mfence();
    unsigned junk;
    uint64_t t = __rdtsc();
    __cpuid(0, junk, junk, junk, junk);      // serialise
    return t;
}

/// Busy-wait for `n` pause instructions
static inline void pause_for_cycles(uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i) _mm_pause();
}

/// Convert cycles → ns from the measured core TSC frequency
static double cycles_to_ns(uint64_t cyc_per_ns, uint64_t cycles) {
    return static_cast<double>(cycles) / cyc_per_ns;
}

// -----------------------------------------------------------------------------
// micro-benchmark  -------------------------------------------------------------
// -----------------------------------------------------------------------------

/// Measure median cycles taken by pause_for_cycles(iterations)
uint64_t median_pause_cost(uint32_t iterations, int reps = 33)
{
    std::vector<uint64_t> samples;
    samples.reserve(reps);

    for (int i = 0; i < reps; ++i) {
        uint64_t s = rdtsc_start();
        pause_for_cycles(iterations);
        uint64_t e = rdtsc_end();
        samples.push_back(e - s);
    }
    std::nth_element(samples.begin(),
                     samples.begin() + reps / 2,
                     samples.end());
    return samples[reps / 2];
}

/// Walk an exponential back-off schedule and print a timing table
void run_config(uint32_t min_wait,
                uint32_t max_wait,
                double   grow)
{
    printf("\n----  min = %-5u  max = %-6u  grow = %.2f  ----\n",
           min_wait, max_wait, grow);

    uint64_t tsc_per_sec = std::chrono::high_resolution_clock::period::den
                         / std::chrono::high_resolution_clock::period::num; // Hz
    double   cyc_per_ns  = static_cast<double>(tsc_per_sec) / 1e9;

    printf("slot  programmed  median-cycles  median-ns\n");
    printf("----  ----------  -------------  ---------\n");

    uint32_t wait = min_wait;
    int      slot = 0;
    while (wait <= max_wait) {
        uint64_t med = median_pause_cost(wait);
        printf("%3d   %10u   %13llu   %8.1f\n",
               slot++, wait,
               static_cast<unsigned long long>(med),
               cycles_to_ns(cyc_per_ns, med));
        wait = static_cast<uint32_t>(wait * grow + 0.5);
        if (wait == 0) break;               // overflow guard
    }
}

// -----------------------------------------------------------------------------
// main ------------------------------------------------------------------------
// -----------------------------------------------------------------------------

int main()
{
    // ❶ Three configs: min ∈ {73, 80, 128}, grow = 2
    for (uint32_t m : {73u, 80u, 128u})
        run_config(m, 16'384u, 2.0);

    // ❷ min = 32, grow ∈ {1.5, 1.7}
    for (double g : {1.5, 1.7})
        run_config(32u, 16'384u, g);

    return 0;
}
