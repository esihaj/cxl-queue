// ---------------------------------------------------------------------------
//  CXL-Resident MPSC Queue  (NT-store / NT-load, AVX-512 only)
// ---------------------------------------------------------------------------
//  * One 64-byte **non-temporal store**  (+ sfence) per enqueue.
//  * One 64-byte **non-temporal load**          per dequeue.
//  * Requires AVX-512F.  Compilation aborts if the target CPU lacks it.
//  * Queue ring and the 64-byte tail cache-line live in user-provided memory
//    (e.g. NUMA / CXL); the class owns no dynamic storage.
// ---------------------------------------------------------------------------
//
//  Build (Sapphire-Rapids or newer):
//      g++ -std=c++20 -O3 -march=native -pthread cxl_mpsc_queue.cpp -lnuma -o cxl_mpsc_queue
//  Run:
//      ./cxl_mpsc_queue            ← alloc on node 0   (default)
//      ./cxl_mpsc_queue 2          ← alloc on node 2
// ---------------------------------------------------------------------------

#ifndef __AVX512F__
#error "This queue implementation requires AVX-512F for 64-byte stream ops"
#endif

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <numa.h>
#include <thread>
#include <chrono>

//-------------------------------------------------------------------
//  Low-level helpers (AVX-512 only)
//-------------------------------------------------------------------

static inline void nt_store_64B(void* dst, const void* src) noexcept {
    const __m512i v = _mm512_loadu_si512(src);                 // src in L1
    _mm512_stream_si512(reinterpret_cast<__m512i*>(dst), v);   // NT-store
    _mm_sfence();
}

static inline void nt_load_64B(void* dst, const void* src) noexcept {
    const __m512i v = _mm512_stream_load_si512(const_cast<void*>(src));
    _mm512_storeu_si512(dst, v);
}

static inline uint64_t nt_load_u64(const uint64_t* src) noexcept {
    __m128i v = _mm_stream_load_si128(
        const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src)));
    return static_cast<uint64_t>(_mm_cvtsi128_si64(v));
}

static inline uint16_t xor_checksum64(const void* p) noexcept {
    const uint64_t* u = reinterpret_cast<const uint64_t*>(p);
    uint64_t acc = 0;
    for (int i = 0; i < 8; ++i) acc ^= u[i];
    acc = (acc >> 32) ^ (acc & 0xFFFFFFFFULL);
    acc = (acc >> 16) ^ (acc & 0xFFFFULL);
    return static_cast<uint16_t>(acc);
}

static inline bool verify_checksum(const void* p) noexcept {
    /* whole-line XOR must be zero */
    return xor_checksum64(p) == 0;
}

static void pin_to_cpu0() {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

//-------------------------------------------------------------------
//  Queue entry – exactly 64 bytes
//-------------------------------------------------------------------

struct alignas(64) Entry {
    uint64_t args[7];           // 56 B payload
    union Meta {
        struct __attribute__((packed)) {
            uint8_t  epoch;
            uint8_t  rpc_method;
            uint8_t  rpc_id;
            uint8_t  _pad;
            int16_t  seal_index;
            uint16_t checksum;
        } f;
        uint64_t u64{0};
    } meta;
};
static_assert(sizeof(Entry) == 64, "Entry not 64 B");

//-------------------------------------------------------------------
//  Queue class
//-------------------------------------------------------------------

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
        assert(order_ >= 8);
        std::memset(ring_, 0, sizeof(Entry) * (1u << order_));
        alignas(64) uint64_t zero[8] = {0};
        nt_store_64B(cxl_tail_, zero);
    }

    /* returns false if ring presently full */
    bool enqueue(const Entry& in)
{
    /* -------- reserve slot (single-producer = plain load) -------- */
    uint32_t slot = head_.load(std::memory_order_relaxed);
    uint32_t cap  = 1u << order_;

    if (static_cast<int32_t>(slot - shadow_tail_) >= static_cast<int32_t>(cap)) {
        ++full;
        shadow_tail_ = static_cast<uint32_t>(nt_load_u64(cxl_tail_));
        if (static_cast<int32_t>(slot - shadow_tail_) >= static_cast<int32_t>(cap))
            return false;                       // still full
    }

    /* -------- write entry -------- */
    Entry tmp = in;
    tmp.meta.f.epoch    = static_cast<uint8_t>(slot >> order_);
    tmp.meta.f.checksum = xor_checksum64(&tmp);          // field was 0

    nt_store_64B(&ring_[slot & mask_], &tmp);
    _mm_sfence();                                        // order NT-store

    /* -------- publish slot -------- */
    head_.store(slot + 1, std::memory_order_release);
    return true;
}



    /* returns false if ring slot not yet ready */
    bool dequeue(Entry& out) {
        Entry   tmp;
        Entry*  e_remote = &ring_[tail_ & mask_];
        nt_load_64B(&tmp, e_remote);

        uint8_t exp_epoch = static_cast<uint8_t>(tail_ >> order_);
        if (tmp.meta.f.epoch != exp_epoch) {failed++; return false;       }// not written yet
        if (!verify_checksum(&tmp))        return false;       // torn read

        out = tmp;
        ++tail_;
        if ((tail_ & 0x7F) == 0) flush_tail();
        return true;
    }

    std::atomic<size_t> failed{0};
    std::atomic<size_t> full{0};
private:
    void flush_tail() noexcept {
        alignas(64) uint64_t buf[8] = { static_cast<uint64_t>(tail_) };
        nt_store_64B(cxl_tail_, buf);
    }

    Entry* const ring_;
    const uint32_t order_;
    const uint32_t mask_;

    std::atomic<uint32_t> head_;
    uint32_t shadow_tail_;
    uint32_t tail_;

    uint64_t* const cxl_tail_;


};

//-------------------------------------------------------------------
//  Sanity / micro-benchmark
//-------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (numa_available() < 0) {
        std::cerr << "libnuma not available\n";
        return 1;
    }

    int node = 0;                    // default NUMA node
    if (argc >= 2) {
        node = std::atoi(argv[1]);
        if (node < 0 || node > numa_max_node()) {
            std::cerr << "Invalid NUMA node id " << node << '\n';
            return 1;
        }
    }
    std::cout << "Allocating ring on NUMA node " << node << '\n';
    pin_to_cpu0();  // keep code/data on same node

    constexpr uint32_t ORDER = 14;             // 16 384 entries
    constexpr size_t   ITER  = 1'000'000;

    const size_t RING_BYTES = (1u << ORDER) * sizeof(Entry);

    Entry*    ring   = static_cast<Entry*>(numa_alloc_onnode(RING_BYTES, node));
    uint64_t* tailCL = static_cast<uint64_t*>(numa_alloc_onnode(64, node));

    CxlMpscQueue q(ring, ORDER, tailCL);

    std::atomic<bool>   done{false};
    std::atomic<size_t> produced{0}, consumed{0};
    std::chrono::nanoseconds t_prod{0}, t_cons{0};

    // Producer --------------------------------------------------------------
    std::thread producer([&](){
        Entry e{}; e.meta.f.rpc_method = 1; e.meta.f.seal_index = -1;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ITER; ++i) {
            for (;;) {
                e.meta.f.rpc_id = static_cast<uint8_t>(i);
                if (q.enqueue(e)) break;
            }
            ++produced;
        }
        t_prod = std::chrono::steady_clock::now() - t0;
        done.store(true, std::memory_order_release);
    });
    // std::this_thread::sleep_for(std::chrono::milliseconds(1)); // let producer start

    // Consumer --------------------------------------------------------------
    std::thread consumer([&](){
        Entry e{};
        auto t0 = std::chrono::steady_clock::now();
        while (!done.load(std::memory_order_acquire) || consumed < produced) {
            if (q.dequeue(e)) ++consumed;
        }
        t_cons = std::chrono::steady_clock::now() - t0;
    });

    producer.join();
    consumer.join();

    auto ns_per = [](std::chrono::nanoseconds ns) {
        return static_cast<double>(ns.count()) / ITER;
    };

    std::cout << "\nProduced / Consumed : " << produced << " / " << consumed << '\n';
    assert(consumed == ITER);
    std::cout << "Producer time       : " << ns_per(t_prod) << " ns/op\n";
    std::cout << "Consumer time       : " << ns_per(t_cons) << " ns/op\n";
    std::cout << "Queue failed reads  : " << q.failed.load() << '\n';
    std::cout << "Queue full reads    : " << q.full.load() << '\n';
    return 0;
}
