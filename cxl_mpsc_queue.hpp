// TODO:
// - Lock when reading CXL tail to avoid races between multiple producers


// ─────────────────────────────────────────────────────────────────────────────
//  CxlMpscQueue — NT-store / NT-load AVX-512 queue for CXL-resident buffers
//  *   One 64-byte non-temporal **store** (+ sfence) per enqueue()
//  *   One 64-byte non-temporal **load**           per dequeue()
//  *   Optional adaptive back-off on the consumer side (spin→yield→sleep)
//  *   Extensive run-time metrics (enqueue/dequeue calls, CXL probes,
//      back-off activity, etc.)
//  Build (Sapphire-Rapids or newer):
//      g++ -std=c++20 -O3 -march=native -pthread cxl_mpsc_queue.cpp -lnuma -o cxl_mpsc_queue
// ─────────────────────────────────────────────────────────────────────────────

#ifndef __AVX512F__
#error "This queue implementation requires AVX-512F for 64-byte stream ops"
#endif

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <syncstream>
#include <numa.h>
#include <thread>
#include <chrono>
#include <algorithm>        // std::max
#include <string_view>

// ─────────────────────────────────────────────────────────────────────────────
//  Low-level helpers (AVX-512 only)
//  Non-temporal stores are used to bypass the cache.
//  Non Temporal loads are only effective on UC/WC memory. So we use cache flush for loads instead
// ─────────────────────────────────────────────────────────────────────────────

// src and dst **must** be 64-B aligned.
static inline void store_nt_64B(void* dst, const void* src) noexcept
{
    const __m512i v = _mm512_load_si512(src);                 // src in L1
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), v); // NT-store
    _mm_sfence();
}

// src and dst **must** be 64-B aligned.
static inline void load_fresh_64B(void* dst, void* src) noexcept
{
    _mm_clflushopt(src);
    _mm_sfence(); // complete the eviction

    const __m512i v = _mm512_load_si512(src);
    _mm512_store_si512(dst, v);
}

static inline void store_nt_u64(uint64_t* dst, uint64_t val) noexcept
{
    _mm_stream_si64(reinterpret_cast<long long*>(dst),
                    static_cast<long long>(val));
    _mm_sfence();
}

static inline uint64_t load_fresh_u64(uint64_t* src) noexcept
{
    _mm_clflushopt(src);
    _mm_sfence();

    return *src;  // regular read
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

// [[gnu::always_inline]]
// static inline std::uint16_t xor_checksum64(const void* ptr) noexcept
// {
//     // ———————————————————————— 1. Load 64 B -------------------------------------------------
//     // vmovdqa64 zmm0, [rdi]
//     // Aligned load = 1 µ-op, 1 cycle issue, no penalties.
//     __m512i v512 = _mm512_load_si512(ptr);

//     // ———————————————————————— 2. 512 → 256 bits --------------------------------------------
//     //  Split the ZMM into two YMM halves and XOR them.
//     //
//     //  _mm512_castsi512_si256:   no-op cast (low 256 bits)
//     //  _mm512_extracti64x4_epi64: shuffle to grab high 256 (imm=1)
//     //  _mm256_xor_si256:         vp xorq ymm0, ymm0, ymm1
//     __m256i v256 = _mm256_xor_si256(
//         _mm512_castsi512_si256(v512),          // low 256
//         _mm512_extracti64x4_epi64(v512, 1));   // high 256

//     // ———————————————————————— 3. 256 → 128 bits --------------------------------------------
//     //  Same idea: split YMM into two XMM lanes and XOR.
//     __m128i v128 = _mm_xor_si128(
//         _mm256_castsi256_si128(v256),          // low 128
//         _mm256_extracti128_si256(v256, 1));    // high 128

//     // ———————————————————————— 4. 128 → 64 bits ---------------------------------------------
//     //  One 1-cycle shuffle to move the upper 64 bits down,
//     //  then XOR – cheaper latency than going through GP regs.
//     __m128i v64  = _mm_xor_si128(v128, _mm_srli_si128(v128, 8));

//     // ———————————————————————— 5. scalar fold 64 → 16 bits -------------------------------
//     //  movq   rax, xmm         (no µ-op on Intel)
//     //  Two shift+XOR folds – classic CRC style parity collapse.
//     std::uint64_t acc = _mm_cvtsi128_si64(v64);
//     acc ^= acc >> 32;
//     acc ^= acc >> 16;
//     return static_cast<std::uint16_t>(acc);
// }

static inline bool verify_checksum(const void* p) noexcept
{
    return xor_checksum64(p) == 0;          // whole-line XOR must be 0
}

// ─────────────────────────────────────────────────────────────────────────────
//  Queue entry – exactly 64 bytes
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(64) Entry {
    uint64_t args[7];           // 56 B payload
    union Meta {
        struct __attribute__((packed)) {
            uint8_t  epoch;
            uint8_t  rpc_method;
            uint16_t rpc_id;
            int16_t  seal_index;
            uint16_t checksum;
        } f;
    } meta;
};
static_assert(sizeof(Entry) == 64, "Entry must be 64 B");

// ─────────────────────────────────────────────────────────────────────────────
//  Metrics – all counters are 64-bit atomics (relaxed updates)
// ─────────────────────────────────────────────────────────────────────────────

struct Metrics {
    /* call counters ------------------------------------------------- */
    std::atomic<size_t> enqueue_calls   {0};
    size_t dequeue_calls   {0};

    /* queue-state probes ------------------------------------------- */
    std::atomic<size_t> read_cxl_tail   {0};
    std::atomic<size_t> queue_full      {0};
    size_t no_new_items    {0};
    size_t checksum_failed {0};
    size_t flush_tail      {0};

    /* back-off activity -------------------------------------------- */
    size_t backoff_spin    {0};   // # _mm_pause()’s
    size_t backoff_yield   {0};   // # std::this_thread::yield()’s
    size_t backoff_sleep   {0};   // # short sleeps

    size_t backoff_total() const noexcept
    {
        return backoff_spin + backoff_yield + backoff_sleep;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Adaptive back-off helper (per consumer thread)
// ─────────────────────────────────────────────────────────────────────────────

struct Backoff {
    uint32_t spins = 0;

    inline void pause(Metrics& m) noexcept
    {
        if (spins < 8) {
            _mm_pause();
            m.backoff_spin++;
        } else if (spins < 16) {
            std::this_thread::yield();
            m.backoff_yield++;
        } else {
            std::this_thread::sleep_for(std::chrono::nanoseconds{100});
            m.backoff_sleep++;
        }
        ++spins;
    }

    inline void reset() noexcept { spins = 0; }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Queue class
// ─────────────────────────────────────────────────────────────────────────────

class CxlMpscQueue {
public:
    CxlMpscQueue(Entry* ring, uint32_t order_log2, uint64_t* cxl_tail)
        : ring_(ring),
          order_(order_log2),
          mask_((1u << order_log2) - 1),
          head_(0),
          shadow_tail_(0),
          tail_(0),
          cxl_tail_(cxl_tail)
    {
        /* verify 64-byte alignment */
        assert((reinterpret_cast<std::uintptr_t>(ring_)     & 63u) == 0 &&
               "ring_ is not 64-byte aligned");
        assert((reinterpret_cast<std::uintptr_t>(cxl_tail_) & 63u) == 0 &&
               "cxl_tail_ is not 64-byte aligned");
        static_assert(alignof(Entry) == 64, "Entry must be alignas(64)");

        std::memset(ring_, 0, sizeof(Entry) * (1u << order_));
        store_nt_u64(cxl_tail_, tail_);
    }

    [[nodiscard]] std::size_t capacity() const noexcept
    {
        return static_cast<std::size_t>(1u) << order_;  // 2^order_
    }

    // ────────────────────────────────────────────────────────────────
    //  enqueue
    // ────────────────────────────────────────────────────────────────
    bool enqueue(Entry& in2, bool debug = false)
    {
        uint32_t slot = head_.load(std::memory_order_relaxed);
        const uint32_t cap = 1u << order_;

        /* fast check: ring looks full? */
        if (static_cast<int32_t>(slot - shadow_tail_) >=
            static_cast<int32_t>(cap))
        {
            if (debug)
                std::osyncstream(std::cout)
                    << "[enqueue] ring-full slot=" << slot
                    << " shadow_tail=" << shadow_tail_
                    << " cap=" << cap << '\n';

            /* refresh tail from CXL */
            shadow_tail_ = static_cast<uint32_t>(load_fresh_u64(cxl_tail_));
            ++metrics.enqueue_calls; // we do this after the call to load_fresh_u64
            ++metrics.read_cxl_tail;

            /* still full after refresh → give up */
            if (static_cast<int32_t>(slot - shadow_tail_) >=
                static_cast<int32_t>(cap))
            {
                ++metrics.queue_full;
                if (debug)
                    std::osyncstream(std::cout)
                        << "[enqueue] queue_full (after CXL tail read)\n";
                return false;
            }
        } else {
            ++metrics.enqueue_calls;
        }

        /* prepare entry (checksum over 64 B) */
        // fill in instead of tmp
        in2.meta.f.epoch    = static_cast<uint8_t>(slot >> order_) + 1;
        in2.meta.f.checksum = 0;
        in2.meta.f.checksum = xor_checksum64(&in2);

        store_nt_64B(&ring_[slot & mask_], &in2);
        _mm_sfence();                               // order NT-store

        head_.store(slot + 1, std::memory_order_release);
        return true;
    }

    // ────────────────────────────────────────────────────────────────
    //  dequeue — with adaptive back-off
    // ────────────────────────────────────────────────────────────────
    bool dequeue(Entry& out, bool debug = false)
    {
        static thread_local Backoff backoff;

        load_fresh_64B(&out, &ring_[tail_ & mask_]);
        ++metrics.dequeue_calls;

        const uint8_t expected_epoch =
            static_cast<uint8_t>(tail_ >> order_) + 1;

        /* epoch mismatch → nothing new yet */
        if (out.meta.f.epoch != expected_epoch) {
            ++metrics.no_new_items;
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] epoch mismatch tail=" << tail_
                    << " exp=" << +expected_epoch
                    << " got=" << +out.meta.f.epoch << '\n';

            backoff.pause(metrics);
            return false;
        }

        /* checksum mismatch */
        if (!verify_checksum(&out)) {
            ++metrics.checksum_failed;
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] checksum failed at tail=" << tail_ << '\n';

            backoff.pause(metrics);
            return false;
        }

        /* success */
        ++tail_;
        backoff.reset();

        /* flush tail back every (cap/4) dequeues, minimum 1 */
        const uint32_t flush_interval = std::max(1u, (1u << order_) / 4);
        if ((tail_ & (flush_interval - 1)) == 0)
            flush_tail(debug);

        return true;
    }

    // ────────────────────────────────────────────────────────────────
    //  read-only access to metrics
    // ────────────────────────────────────────────────────────────────
    const Metrics& get_metrics() const noexcept { return metrics; }

    /* ---------------------------------------------------------------
     *  Dump run-time counters.
     *  @param label  A short queue name (e.g. "REQ", "RSP").
     *  @param os     Output stream (defaults to std::cout).
     * ------------------------------------------------------------- */
    void print_metrics(std::string_view label = "",
                       std::ostream&     os = std::cout) const
    {
        os << "── Metrics [" << label << "] ─────────────────────\n"
           << "Enqueue calls        : " << metrics.enqueue_calls.load()   << '\n'
           << "Dequeue calls        : " << metrics.dequeue_calls   << '\n'
           << "CXL-tail reads       : " << metrics.read_cxl_tail.load()   << '\n'
           << "Still-full           : " << metrics.queue_full.load()      << '\n'
           << "No-new-item polls    : " << metrics.no_new_items    << '\n'
           << "Checksum failures    : " << metrics.checksum_failed << '\n'
           << "Tail flushes         : " << metrics.flush_tail      << '\n'
           << "Back-off (total)     : " << metrics.backoff_total()        << '\n'
           << "Back-off (spin)      : " << metrics.backoff_spin    << '\n'
           << "Back-off (yield)     : " << metrics.backoff_yield   << '\n'
           << "Back-off (sleep)     : " << metrics.backoff_sleep   << '\n';
    }

private:
    // ────────────────────────────────────────────────────────────────
    //  flush_tail – write tail to CXL & count
    // ────────────────────────────────────────────────────────────────
    inline void flush_tail(bool debug = false)
    {
        store_nt_u64(cxl_tail_, tail_);
        ++metrics.flush_tail;

        if (debug)
            std::osyncstream(std::cout)
                << "[flush_tail] WRITE cxl_tail=" << tail_ << '\n';
    }

    /* fixed data ---------------------------------------------------- */
    Entry* const             ring_;
    const uint32_t           order_;
    const uint32_t           mask_;
    std::atomic<uint32_t>    head_;
    uint32_t                 shadow_tail_;
    uint32_t                 tail_;
    uint64_t* const          cxl_tail_;

    /* metrics block ------------------------------------------------- */
    Metrics                  metrics;
};
