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
// ─────────────────────────────────────────────────────────────────────────────

static inline void nt_store_64B(void* dst, const void* src) noexcept
{
    const __m512i v = _mm512_loadu_si512(src);                 // src in L1
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), v);   // NT-store
    _mm_sfence();
}

static inline void nt_load_64B(void* dst, const void* src) noexcept
{
    const __m512i v = _mm512_stream_load_si512(const_cast<void*>(src));
    _mm512_storeu_si512(dst, v);
}

static inline uint64_t nt_load_u64(const uint64_t* src) noexcept
{
    __m128i v = _mm_stream_load_si128(
        const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src)));
    return static_cast<uint64_t>(_mm_cvtsi128_si64(v));
}

static inline void nt_store_u64(uint64_t* dst, uint64_t val) noexcept
{
    _mm_stream_si64(reinterpret_cast<long long*>(dst),
                    static_cast<long long>(val));
    _mm_sfence();
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
    std::atomic<size_t> dequeue_calls   {0};

    /* queue-state probes ------------------------------------------- */
    std::atomic<size_t> read_cxl_tail   {0};
    std::atomic<size_t> queue_full      {0};
    std::atomic<size_t> no_new_items    {0};
    std::atomic<size_t> checksum_failed {0};
    std::atomic<size_t> flush_tail      {0};

    /* back-off activity -------------------------------------------- */
    std::atomic<size_t> backoff_spin    {0};   // # _mm_pause()’s
    std::atomic<size_t> backoff_yield   {0};   // # std::this_thread::yield()’s
    std::atomic<size_t> backoff_sleep   {0};   // # short sleeps

    size_t backoff_total() const noexcept
    {
        return backoff_spin.load(std::memory_order_relaxed) +
               backoff_yield.load(std::memory_order_relaxed) +
               backoff_sleep.load(std::memory_order_relaxed);
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
            m.backoff_spin.fetch_add(1, std::memory_order_relaxed);
        } else if (spins < 16) {
            std::this_thread::yield();
            m.backoff_yield.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::this_thread::sleep_for(std::chrono::nanoseconds{100});
            m.backoff_sleep.fetch_add(1, std::memory_order_relaxed);
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
        nt_store_u64(cxl_tail_, tail_);
    }

    // ────────────────────────────────────────────────────────────────
    //  enqueue
    // ────────────────────────────────────────────────────────────────
    bool enqueue(const Entry& in, bool debug = false)
    {
        ++metrics.enqueue_calls;

        uint32_t slot = head_.load(std::memory_order_relaxed);
        const uint32_t cap = 1u << order_;

        /* fast check: ring looks full? */
        if (static_cast<int32_t>(slot - shadow_tail_) >=
            static_cast<int32_t>(cap))
        {
            ++metrics.read_cxl_tail;
            if (debug)
                std::osyncstream(std::cout)
                    << "[enqueue] ring-full slot=" << slot
                    << " shadow_tail=" << shadow_tail_
                    << " cap=" << cap << '\n';

            /* refresh tail from CXL */
            shadow_tail_ = static_cast<uint32_t>(nt_load_u64(cxl_tail_));

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
        }

        /* prepare entry (checksum over 64 B) */
        Entry tmp           = in;
        tmp.meta.f.epoch    = static_cast<uint8_t>(slot >> order_) + 1;
        tmp.meta.f.checksum = 0;
        tmp.meta.f.checksum = xor_checksum64(&tmp);

        nt_store_64B(&ring_[slot & mask_], &tmp);
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
        ++metrics.dequeue_calls;

        Entry tmp;
         _mm_clflushopt(&ring_[tail_ & mask_]); // not needed on UC/WC memory
        nt_load_64B(&tmp, &ring_[tail_ & mask_]);

        const uint8_t expected_epoch =
            static_cast<uint8_t>(tail_ >> order_) + 1;

        /* epoch mismatch → nothing new yet */
        if (tmp.meta.f.epoch != expected_epoch) {
            ++metrics.no_new_items;
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] epoch mismatch tail=" << tail_
                    << " exp=" << +expected_epoch
                    << " got=" << +tmp.meta.f.epoch << '\n';

            backoff.pause(metrics);
            return false;
        }

        /* checksum mismatch */
        if (!verify_checksum(&tmp)) {
            ++metrics.checksum_failed;
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] checksum failed at tail=" << tail_ << '\n';

            backoff.pause(metrics);
            return false;
        }

        /* success */
        out = tmp;
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
           << "Dequeue calls        : " << metrics.dequeue_calls.load()   << '\n'
           << "CXL-tail reads       : " << metrics.read_cxl_tail.load()   << '\n'
           << "Still-full           : " << metrics.queue_full.load()      << '\n'
           << "No-new-item polls    : " << metrics.no_new_items.load()    << '\n'
           << "Checksum failures    : " << metrics.checksum_failed.load() << '\n'
           << "Tail flushes         : " << metrics.flush_tail.load()      << '\n'
           << "Back-off (total)     : " << metrics.backoff_total()        << '\n'
           << "Back-off (spin)      : " << metrics.backoff_spin.load()    << '\n'
           << "Back-off (yield)     : " << metrics.backoff_yield.load()   << '\n'
           << "Back-off (sleep)     : " << metrics.backoff_sleep.load()   << '\n';
    }

private:
    // ────────────────────────────────────────────────────────────────
    //  flush_tail – write tail to CXL & count
    // ────────────────────────────────────────────────────────────────
    inline void flush_tail(bool debug = false)
    {
        nt_store_u64(cxl_tail_, tail_);
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
