// ─────────────────────────────────────────────────────────────────────────────
//  MPSC → SPSC Test Suite   (rpc_id casts now use uint16_t)
//  Compile on an AVX-512 capable machine, e.g.:
//      g++ -std=c++20 -O3 -march=native -pthread mpsc_queue_tests.cpp -o mpsc_queue_tests
// ─────────────────────────────────────────────────────────────────────────────
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>

#include "cxl_mpsc_queue.hpp"   // ← include or replace with the full queue code

using namespace std::chrono_literals;

constexpr uint32_t ORDER = 4;                 // 256-slot ring for easy wrap tests
constexpr uint32_t CAP   = 1u << ORDER;       // queue capacity

// ---------------------------------------------------------------------------
//  Simple RAII harness for allocating ring + tail cache-line
// ---------------------------------------------------------------------------
#include <numa.h>

struct TestEnv {
    Entry*         ring;
    uint64_t*      tail_cxl;
    CxlMpscQueue*  q;

    explicit TestEnv(int numa_node = 0) {
        const size_t ring_bytes = sizeof(Entry) * CAP;

        ring     = static_cast<Entry*>(numa_alloc_onnode(ring_bytes, numa_node));
        tail_cxl = static_cast<uint64_t*>(numa_alloc_onnode(64,        numa_node));

        // Ensure both regions are 64-byte aligned
        assert(reinterpret_cast<uintptr_t>(ring)     % 64 == 0 && "ring not 64-byte aligned");
        assert(reinterpret_cast<uintptr_t>(tail_cxl) % 64 == 0 && "tail_cxl not 64-byte aligned");

        std::memset(ring,     0, ring_bytes);
        std::memset(tail_cxl, 0, 64);

        q = new CxlMpscQueue(ring, ORDER, tail_cxl);
    }

    ~TestEnv() {
        delete q;
        numa_free(ring,     sizeof(Entry) * CAP);
        numa_free(tail_cxl, 64);
    }
};

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
constexpr const char* GREEN = "\033[32m";
constexpr const char* RED   = "\033[31m";
constexpr const char* RESET = "\033[0m";

inline void pass(const char* n)  { std::cout << GREEN << '[' << n << "] PASSED"        << RESET << '\n'; }
inline void fail(const char* n, const char* m)
{ std::cout << RED   << '[' << n << "] FAILED: " << m << RESET << '\n'; }

// ---------------------------------------------------------------------------
//  1. Single enqueue / dequeue
// ---------------------------------------------------------------------------
void test_enqueue_dequeue_single() {
    constexpr const char* N = "test_enqueue_dequeue_single";
    TestEnv env;

    Entry in{};  in.meta.f.rpc_id = 42;
    if (!env.q->enqueue(in, true))                    return fail(N, "enqueue failed");

    Entry out{};
    if (!env.q->dequeue(out, true))                   return fail(N, "dequeue failed");
    if (out.meta.f.rpc_id != 42)                      return fail(N, "value mismatch");
    pass(N);
}

// ---------------------------------------------------------------------------
//  2. Multiple enqueue / dequeue (FIFO)
// ---------------------------------------------------------------------------
void test_enqueue_dequeue_multiple() {
    constexpr const char* N = "test_enqueue_dequeue_multiple";
    TestEnv env;

    for (uint32_t i = 0; i < CAP - 1; ++i) {
        Entry e{}; e.meta.f.rpc_id = static_cast<uint16_t>(i);
        if (!env.q->enqueue(e, true))                 return fail(N, "unexpected full");
    }
    for (uint32_t i = 0; i < CAP - 1; ++i) {
        Entry e{};
        if (!env.q->dequeue(e, true))                 return fail(N, "unexpected empty");
        if (e.meta.f.rpc_id != i)                     return fail(N, "order mismatch");
    }
    pass(N);
}

// ---------------------------------------------------------------------------
//  3. Wrap-around correctness
// ---------------------------------------------------------------------------
void test_wraparound_behavior() {
    constexpr const char* N = "test_wraparound_behavior";
    TestEnv env;

    // 3.1 fill queue
    for (uint32_t i = 0; i < CAP; ++i) {
        Entry e{}; e.meta.f.rpc_id = static_cast<uint16_t>(i);
        if (!env.q->enqueue(e, true))                 return fail(N, "fill failed");
    }
    // 3.2 dequeue half
    for (uint32_t i = 0; i < CAP / 2; ++i) {
        Entry e{};
        if (!env.q->dequeue(e, true))                 return fail(N, "deq half failed");
        if (e.meta.f.rpc_id != i)                     return fail(N, "order mismatch (phase 1)");
    }
    // 3.3 enqueue another half
    for (uint32_t i = CAP; i < CAP + CAP / 2; ++i) {
        Entry e{}; e.meta.f.rpc_id = static_cast<uint16_t>(i);
        if (!env.q->enqueue(e, true))                 return fail(N, "wrap enqueue failed");
    }
    std::cout << '[' << N << "] wraparound test: start i=" << CAP / 2
              << ", end i=" << CAP + CAP / 2 - 1 << ", CAP=" << CAP << '\n';
    // 3.4 dequeue remaining CAP elements
    for (uint32_t i = CAP / 2; i < CAP + CAP / 2; ++i) {
        Entry e{};
        if (!env.q->dequeue(e, true))                 return fail(N, "final deq failed");
        if (e.meta.f.rpc_id != i) {
            std::cout << '[' << N << "] order mismatch (phase 2): expected=" << i
                      << "  got=" << static_cast<int>(e.meta.f.rpc_id)
                      << "  (dequeue-idx=" << (i - CAP / 2) << ")\n";
            return fail(N, "order mismatch (phase 2)");
        }
    }
    pass(N);
}

// ---------------------------------------------------------------------------
//  4. Dequeue on empty queue
// ---------------------------------------------------------------------------
void test_dequeue_empty() {
    constexpr const char* N = "test_dequeue_empty";
    TestEnv env;
    Entry e{};
    if (env.q->dequeue(e))                            return fail(N, "dequeue succeeded on empty");
    pass(N);
}

// ---------------------------------------------------------------------------
//  5. Enqueue on full queue
// ---------------------------------------------------------------------------
void test_enqueue_full() {
    constexpr const char* N = "test_enqueue_full";
    TestEnv env;

    for (uint32_t i = 0; i < CAP; ++i) {
        Entry e{};
        if (!env.q->enqueue(e))                       return fail(N, "prematurely full");
    }
    Entry extra{};
    if (env.q->enqueue(extra))                        return fail(N, "enqueue succeeded when full");
    pass(N);
}

// ---------------------------------------------------------------------------
//  6. Re-use queue after emptying              (no hard-coded constants)
// ---------------------------------------------------------------------------
void test_reuse_after_emptying() {
    constexpr const char* N = "test_reuse_after_emptying";
    TestEnv env;

    const uint32_t BATCH1 = CAP / 2;      // first fill → half the ring
    const uint32_t BATCH2 = CAP / 4;      // second fill → quarter of the ring
    const uint16_t OFFSET = static_cast<uint16_t>(CAP);

    /* ── round 1: fill half, then drain it ───────────────────────────────── */
    for (uint32_t i = 0; i < BATCH1; ++i) {
        Entry e{}; e.meta.f.rpc_id = static_cast<uint16_t>(i);
        if (!env.q->enqueue(e))                        return fail(N, "enqueue r1");
    }

    Entry tmp{};
    for (uint32_t i = 0; i < BATCH1; ++i) {
        if (!env.q->dequeue(tmp))                      return fail(N, "dequeue r1");
        if (tmp.meta.f.rpc_id != i)                    return fail(N, "order r1");
    }

    /* ── round 2: reuse queue with a smaller batch ───────────────────────── */
    for (uint32_t i = 0; i < BATCH2; ++i) {
        Entry e2{}; e2.meta.f.rpc_id = static_cast<uint16_t>(i + OFFSET);
        if (!env.q->enqueue(e2))                       return fail(N, "enqueue r2");
    }

    for (uint32_t i = 0; i < BATCH2; ++i) {
        if (!env.q->dequeue(tmp))                      return fail(N, "dequeue r2");
        if (tmp.meta.f.rpc_id != i + OFFSET)           return fail(N, "order r2");
    }

    pass(N);
}


// ---------------------------------------------------------------------------
//  7. Threaded producer / consumer smoke test
// ---------------------------------------------------------------------------
void test_threaded_spsc() {
    constexpr const char* N = "test_threaded_spsc";
    constexpr uint32_t   ITERS = 50'000;

    TestEnv env;
    std::atomic<uint32_t> produced{0}, consumed{0};
    std::atomic<bool> done{false};

    std::thread prod([&]{
        Entry e{};
        for (uint32_t i = 0; i < ITERS; ++i) {
            e.meta.f.rpc_id = static_cast<uint16_t>(i);
            while (!env.q->enqueue(e)) {}
            ++produced;
        }
        done.store(true, std::memory_order_release);
    });

    std::thread cons([&]{
        Entry e{};
        while (!done.load(std::memory_order_acquire) || consumed < produced) {
            if (env.q->dequeue(e)) {
                if (e.meta.f.rpc_id != consumed)
                    fail(N, "order mismatch");
                ++consumed;
            }
        }
    });

    prod.join();
    cons.join();
    if (consumed != ITERS)                            return fail(N, "lost messages");
    pass(N);
}

// ---------------------------------------------------------------------------
//  8. Interleaved timing (sleep jitters)
// ---------------------------------------------------------------------------
void test_interleaved_timing() {
    constexpr const char* N = "test_interleaved_timing";
    constexpr uint32_t ITERS = 10'000;
    TestEnv env;

    std::thread prod([&]{
        Entry e{};
        for (uint32_t i = 0; i < ITERS; ++i) {
            e.meta.f.rpc_id = static_cast<uint16_t>(i);
            while (!env.q->enqueue(e)) {}
            if (i % 256 == 0) std::this_thread::sleep_for(100ns);
        }
    });

    uint32_t seen = 0;
    std::thread cons([&]{
        Entry e{};
        while (seen < ITERS) {
            if (env.q->dequeue(e)) {
                if (e.meta.f.rpc_id != seen)
                    fail(N, "order mismatch");
                ++seen;
                if (seen % 128 == 0) std::this_thread::sleep_for(150ns);
            }
        }
    });

    prod.join();
    cons.join();
    if (seen != ITERS)                                 return fail(N, "lost messages");
    pass(N);
}

// ---------------------------------------------------------------------------
//  9. No overwrite / skip detection   (interleaved producer/consumer)
// ---------------------------------------------------------------------------
void test_no_overwrite_or_skip() {
    constexpr const char* N = "test_no_overwrite_or_skip";
    constexpr uint32_t ITERS = CAP * 4;        // several full-ring wraps
    TestEnv env;

    std::vector<bool> flags(ITERS, false);

    uint32_t written = 0;      // how many items we have attempted to enqueue
    uint32_t read    = 0;      // how many items we have dequeued / verified

    Entry e{};
    while (read < ITERS) {

        /* ── try to enqueue if we still have work left ──────────────────── */
        if (written < ITERS) {
            e.meta.f.rpc_id = static_cast<uint16_t>(written & 0xFFFF);
            if (env.q->enqueue(e)) {          // success → move producer cursor
                ++written;
                continue;                     // attempt next write first
            }
        }

        /* ── queue was full OR producer finished → make progress by reading */
        Entry out{};
        if (env.q->dequeue(out)) {
            uint32_t wrap = read / 0x10000u;                 // 65536-entry eras
            uint32_t idx  = out.meta.f.rpc_id + 0x10000u*wrap;

            if (idx >= ITERS)          return fail(N, "index out of range");
            if (flags[idx])            return fail(N, "duplicate slot read");

            flags[idx] = true;
            ++read;
        }
    }

    /* verify we actually saw every logical slot exactly once */
    for (bool f : flags) if (!f)       return fail(N, "missed slot");
    pass(N);
}

// ---------------------------------------------------------------------------
// 10. Checksum logic: standalone + in-queue
// ---------------------------------------------------------------------------
void test_checksum_logic() {
    constexpr const char* N = "test_checksum_logic";
    TestEnv env;

    // ── 1. Stand-alone verification on a crafted Entry ────────────────────
    Entry e{};
    for (int i = 0; i < 7; ++i) e.args[i] = 0x1111111111111111ULL * (i + 1);
    e.meta.f.rpc_method = 7;
    e.meta.f.rpc_id     = 77;
    e.meta.f.seal_index = -123;

    e.meta.f.checksum = xor_checksum64(&e);      // make XOR of 512-bit line zero
    if (!verify_checksum(&e))
        return fail(N, "verify_checksum failed on pristine entry");

    // Flip a bit → checksum must break
    reinterpret_cast<uint8_t*>(&e)[5] ^= 0x01;
    if (verify_checksum(&e))
        return fail(N, "checksum still valid after corruption");

    // ── 2. Queue-integrated check ─────────────────────────────────────────
    // 2.a enqueue a "good" entry → dequeue must succeed
    Entry good{};
    for (int i = 0; i < 7; ++i) good.args[i] = 0xAA55AA55AA55AA55ULL + i;
    good.meta.f.rpc_method = 3;
    good.meta.f.rpc_id     = 0xEE;
    good.meta.f.seal_index = 42;

    env.q->enqueue(good);
    Entry out{};
    if (!env.q->dequeue(out))
        return fail(N, "queue rejected good entry");
    if (!verify_checksum(&out))
        return fail(N, "checksum wrong on dequeue");

    // 2.b enqueue another entry, then corrupt it in-place → dequeue must fail
    Entry bad = good;
    bad.meta.f.rpc_id = 0xEF;
    env.q->enqueue(bad);

    // We know this is slot 1 (we only enqueued twice, capacity ≫ 2)
    env.ring[1].args[0] ^= 0x1;                  // corrupt payload

    if (env.q->dequeue(out))
        return fail(N, "queue accepted corrupted entry");

    pass(N);
}


// ---------------------------------------------------------------------------
//  Main: invoke all tests
// ---------------------------------------------------------------------------
int main() {
    test_enqueue_dequeue_single();   std::cout << '\n';
    test_enqueue_dequeue_multiple(); std::cout << '\n';
    test_wraparound_behavior();      std::cout << '\n';
    test_dequeue_empty();            std::cout << '\n';
    test_enqueue_full();             std::cout << '\n';
    test_reuse_after_emptying();     std::cout << '\n';
    test_threaded_spsc();            std::cout << '\n';
    test_interleaved_timing();       std::cout << '\n';
    test_no_overwrite_or_skip();     std::cout << '\n';
    test_checksum_logic();
    return 0;
}
