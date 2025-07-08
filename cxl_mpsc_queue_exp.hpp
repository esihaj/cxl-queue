// ─────────────────────────────────────────────────────────────────────────────
//  CxlMpscQueue — NT-store / NT-load AVX-512 queue for CXL-resident buffers
//  * One 64-byte non-temporal **store** (+ sfence) per enqueue()
//  * One 64-byte non-temporal **load** per dequeue()
//  * Optional adaptive back-off on the consumer side (spin→yield→sleep)
//  * Extensive run-time metrics (enqueue/dequeue calls, CXL probes,
//       back-off activity, etc.)
//  Build (Sapphire-Rapids or newer):
//       g++ -std=c++20 -O3 -march=native -pthread cxl_mpsc_queue.cpp -lnuma -o cxl_mpsc_queue
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
#include <algorithm>      // std::min, std::max
#include <string_view>

// ─────────────────────────────────────────────────────────────────────────────
//  Low-level helpers (AVX-512 only)
// ─────────────────────────────────────────────────────────────────────────────

// src and dst **must** be 64-B aligned.
static inline void store_nt_64B(void* dst, const void* src) noexcept
{
    const __m512i v = _mm512_load_si512(src);              // src in L1
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

// Pauses the CPU for a number of cycles. This is a local operation
// that does not generate memory traffic.
static inline void cpu_relax_for_cycles(uint32_t cycles) noexcept
{
    for (uint32_t i = 0; i < cycles; ++i) {
        _mm_pause();
    }
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
    return xor_checksum64(p) == 0;        // whole-line XOR must be 0
}

// ─────────────────────────────────────────────────────────────────────────────
//  Queue entry – exactly 64 bytes
// ─────────────────────────────────────────────────────────────────────────────

struct alignas(64) Entry {
    uint64_t args[7];      // 56 B payload
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
    size_t enqueue_calls   {0};
    size_t dequeue_calls   {0};

    /* queue-state probes ------------------------------------------- */
    size_t read_cxl_tail   {0};
    size_t queue_full      {0};
    size_t no_new_items    {0};
    size_t checksum_failed {0};
    size_t flush_tail      {0};

    /* Consumer (dequeue) back-off activity ------------------------- */
    size_t consumer_backoff_events        {0};
    size_t consumer_backoff_cycles_waited {0};

    /* Producer (enqueue) back-off activity ------------------------- */
    size_t producer_backoff_events        {0};
    size_t producer_backoff_cycles_waited {0};
};

// ─────────────────────────────────────────────────────────────────────────────
//  Exponential back-off helper (per consumer/producer thread)
// ─────────────────────────────────────────────────────────────────────────────

struct ExponentialBackoff {
    // Max wait is shared, but min wait is configurable per instance
    static constexpr uint32_t MAX_WAIT_CYCLES = 16384;
    const uint32_t MIN_WAIT_CYCLES;

    uint32_t current_wait;

    // Constructor to allow different minimum wait times
    explicit ExponentialBackoff(uint32_t min_wait)
        : MIN_WAIT_CYCLES(min_wait), current_wait(min_wait) {}

    // Pause locally, then increase wait time for the next attempt.
    inline void pause(size_t& events_counter,
                      size_t& cycles_counter) noexcept
    {
        cpu_relax_for_cycles(current_wait);
        events_counter++;
        cycles_counter+= current_wait;
        current_wait = std::min(current_wait * 2, MAX_WAIT_CYCLES);
    }

    // Reset the wait time after a successful operation.
    inline void reset() noexcept {
        current_wait = MIN_WAIT_CYCLES;
    }
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
        assert((reinterpret_cast<std::uintptr_t>(ring_)    & 63u) == 0 &&
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
    //  enqueue — with back-off on queue full
    // ────────────────────────────────────────────────────────────────
    bool enqueue(Entry& in, bool debug = false)
    {
        static thread_local ExponentialBackoff backoff_full{128};
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
            shadow_tail_ = static_cast<uint32_t>(load_fresh_u64(cxl_tail_));

            /* still full after refresh → backoff and give up */
            if (static_cast<int32_t>(slot - shadow_tail_) >=
                static_cast<int32_t>(cap))
            {
                ++metrics.queue_full;
                backoff_full.pause(metrics.producer_backoff_events,
                                   metrics.producer_backoff_cycles_waited);
                if (debug)
                    std::osyncstream(std::cout)
                        << "[enqueue] queue_full (after CXL tail read)\n";
                return false;
            }
        }
        
        // If we got here, the queue is not full, so reset producer backoff
        backoff_full.reset();

        /* prepare entry (checksum over 64 B) */
        in.meta.f.epoch    = static_cast<uint8_t>(slot >> order_) + 1;
        in.meta.f.checksum = 0;
        in.meta.f.checksum = xor_checksum64(&in);

        store_nt_64B(&ring_[slot & mask_], &in);
        _mm_sfence();                       // order NT-store

        head_.store(slot + 1, std::memory_order_release);
        return true;
    }

    // ────────────────────────────────────────────────────────────────
    //  dequeue — with separate back-offs for empty and checksum failure
    // ────────────────────────────────────────────────────────────────
    bool dequeue(Entry& out, bool debug = false)
    {
        static thread_local ExponentialBackoff backoff_empty{50};
        static thread_local ExponentialBackoff backoff_checksum{100};
        ++metrics.dequeue_calls;

        load_fresh_64B(&out, &ring_[tail_ & mask_]);

        const uint8_t expected_epoch =
            static_cast<uint8_t>(tail_ >> order_) + 1;

        /* epoch mismatch → nothing new yet */
        if (out.meta.f.epoch != expected_epoch) {
            ++metrics.no_new_items;
            backoff_empty.pause(metrics.consumer_backoff_events,
                                metrics.consumer_backoff_cycles_waited);
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] epoch mismatch tail=" << tail_
                    << " exp=" << +expected_epoch
                    << " got=" << +out.meta.f.epoch << '\n';
            return false;
        }

        /* checksum mismatch */
        if (!verify_checksum(&out)) {
            ++metrics.checksum_failed;
            backoff_checksum.pause(metrics.consumer_backoff_events,
                                   metrics.consumer_backoff_cycles_waited);
            if (debug)
                std::osyncstream(std::cout)
                    << "[dequeue] checksum failed at tail=" << tail_ << '\n';
            return false;
        }

        /* success */
        ++tail_;
        backoff_empty.reset();
        backoff_checksum.reset();

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
     * Dump run-time counters.
     * @param label  A short queue name (e.g. "REQ", "RSP").
     * @param os     Output stream (defaults to std::cout).
     * ------------------------------------------------------------- */
    void print_metrics(std::string_view label = "",
                           std::ostream&      os = std::cout) const
    {
        os << "── Metrics [" << label << "] ─────────────────────\n"
           << "Enqueue calls           : " << metrics.enqueue_calls    << '\n'
           << "Dequeue calls           : " << metrics.dequeue_calls    << '\n'
           << "CXL-tail reads (P)      : " << metrics.read_cxl_tail    << '\n'
           << "Queue-full events (P)   : " << metrics.queue_full       << '\n'
           << "No-new-item polls (C)   : " << metrics.no_new_items     << '\n'
           << "Checksum failures (C)   : " << metrics.checksum_failed  << '\n'
           << "Tail flushes (C)        : " << metrics.flush_tail       << '\n'
           << "── Back-off ──────────────────────────\n"
           << "Producer Events         : " << metrics.producer_backoff_events << '\n'
           << "Producer Cycles Waited  : " << metrics.producer_backoff_cycles_waited << '\n'
           << "Consumer Events         : " << metrics.consumer_backoff_events << '\n'
           << "Consumer Cycles Waited  : " << metrics.consumer_backoff_cycles_waited << '\n';
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
    Entry* const              ring_;
    const uint32_t            order_;
    const uint32_t            mask_;
    std::atomic<uint32_t>     head_;
    uint32_t                  shadow_tail_;
    uint32_t                  tail_;
    uint64_t* const           cxl_tail_;

    /* metrics block ------------------------------------------------- */
    Metrics                   metrics;
};