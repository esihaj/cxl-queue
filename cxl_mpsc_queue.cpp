// cxl_mpsc_queue.cpp
// ─────────────────────────────────────────────────────────────────────────────
//  Sanity / micro-benchmark using CXL-backed allocators
//
//  CLI (choose **one** form) ––––––––––––––––––––––––––––––––––––––––––––––
//    pin <cpu_id> numa <node_id> [iter_count]
//    pin <cpu_id> dax            [iter_count]
//
//    • cpu_id      : logical CPU to pin the main thread to
//    • node_id     : NUMA node for DRAM allocation            (numa form)
//    • iter_count  : #iterations (default = 10’000’000 = 10 M)
//
//  Examples
//    sudo ./doorbell_bench pin 15 numa 0               # 10 M iters on node 0
//    sudo ./doorbell_bench pin 3  dax  20_000_000      # 20 M iters on /dev/dax
//
//  Build (GCC ≥ 12 or Clang ≥ 15)
//    g++ -O3 -std=c++20 -march=native -mavx512f -mavx512bw \
//        -mclflushopt -mclwb -mmovdir64b  \
//        cxl_mpsc_queue.cpp  -o cxl_mpsc_queue \
//        -lnuma -lpthread
// ─────────────────────────────────────────────────────────────────────────────

#include "cxl_mpsc_queue_exp.hpp"
#include "cxl_allocator.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numa.h>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>

//-------------------------------------------------------------------
//  Helpers
//-------------------------------------------------------------------
static void pin_to_cpu(int cpu_id) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu_id, &set);
    if (::sched_setaffinity(0, sizeof(set), &set) != 0)
        std::perror("sched_setaffinity");
}

[[noreturn]] static void print_usage(const char* prog) {
    std::cerr <<
        "usage  : " << prog << " pin <cpu_id> numa <node_id> [iter_count]\n"
        "       | " << prog << " pin <cpu_id> dax [iter_count]\n"
        "notes  : iter_count defaults to 10M when omitted\n";
    std::exit(EXIT_FAILURE);
}
//-------------------------------------------------------------------
//  Main
//-------------------------------------------------------------------
int main(int argc, char* argv[]) {
    constexpr std::size_t DEFAULT_ITERS = 10'000'000ULL;

    //-----------------------------------------------------------------------
    //  Parse CLI
    //-----------------------------------------------------------------------
    if (argc < 4) print_usage(argv[0]);

    if (std::string{argv[1]} != "pin") print_usage(argv[0]);
    const int         cpu_id = std::stoi(argv[2]);
    const std::string mode   = argv[3];

    bool        use_dax   = false;
    int         numa_node = -1;
    std::size_t ITER      = DEFAULT_ITERS;

    if (mode == "numa") {
        if (argc < 5) print_usage(argv[0]);
        numa_node = std::stoi(argv[4]);
        if (argc >= 6) ITER = std::stoull(argv[5]);
    } else if (mode == "dax") {
        use_dax = true;
        if (argc >= 5) ITER = std::stoull(argv[4]);
    } else {
        print_usage(argv[0]);
    }

    if (!use_dax && (numa_node < 0 || numa_node > numa_max_node())) {
        std::cerr << "Invalid NUMA node id " << numa_node << '\n';
        return EXIT_FAILURE;
    }

    //-----------------------------------------------------------------------
    //  Pin main thread and create allocator
    //-----------------------------------------------------------------------
    pin_to_cpu(cpu_id);

    std::unique_ptr<cxl::CxlAllocator> alloc;
    try {
        if (use_dax) {
            alloc = std::make_unique<cxl::DaxAllocator>();
            std::cout << "Using DAX allocator on /dev/dax* slice\n";
        } else {
            alloc = std::make_unique<cxl::NumaAllocator>(numa_node);
            std::cout << "Using NUMA allocator on node " << numa_node << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Allocator init failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Pinned to CPU " << cpu_id << '\n'
              << "Iterations      : " << ITER << "\n\n";

    //-----------------------------------------------------------------------
    //  Queue setup – allocate from CXL allocator
    //-----------------------------------------------------------------------
    constexpr uint32_t ORDER       = 14;                       // 16 384 entries
    const std::size_t  RING_BYTES  = (1u << ORDER) * sizeof(Entry);

    Entry*    ring      = static_cast<Entry*>   (alloc->allocate_aligned(RING_BYTES, 64));
    uint64_t* tail_cxl  = static_cast<uint64_t*>(alloc->allocate_aligned(64,          64));

    CxlMpscQueue q(ring, ORDER, tail_cxl);

    //-----------------------------------------------------------------------
    //  Producer / Consumer micro-benchmark with warm-up
    //-----------------------------------------------------------------------
    const std::size_t WARMUP = q.capacity() / 4;   // pre-produce ¼ of queue
    assert(WARMUP < ITER && "warm-up must be < total iterations");

    {   // warm-up phase
        Entry e{};
        e.meta.f.rpc_method = 1;
        e.meta.f.seal_index = -1;
        for (std::size_t i = 0; i < WARMUP; ++i) {
            e.meta.f.rpc_id = static_cast<uint8_t>(i);
            while (!q.enqueue(e)) { /* should not happen */ }
        }
    }

    // ── Record baseline metrics AFTER warm-up ──────────────────────────────
    const std::size_t enqueue_warmup_calls =
        q.get_metrics().enqueue_calls;
    const std::size_t dequeue_warmup_calls =
        q.get_metrics().dequeue_calls;

    // ── Timed phase ────────────────────────────────────────────────────────
    std::chrono::nanoseconds t_prod{0}, t_cons{0};

    std::thread producer([&] {
        Entry e{};
        e.meta.f.rpc_method = 1;
        e.meta.f.seal_index = -1;

        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = WARMUP; i < ITER; ++i) {
            e.meta.f.rpc_id = static_cast<uint8_t>(i);
            while (!q.enqueue(e)) { /* spin */ }
        }
        t_prod = std::chrono::steady_clock::now() - t0;
    });

    std::thread consumer([&] {
        Entry e{};
        std::size_t consumed = 0;
        const auto t0 = std::chrono::steady_clock::now();

        while (consumed < ITER) {
            if (q.dequeue(e)) ++consumed;
        }
        t_cons = std::chrono::steady_clock::now() - t0;
    });

    producer.join();
    consumer.join();

    //-----------------------------------------------------------------------
    //  Results
    //-----------------------------------------------------------------------
    const auto ns_per = [](std::size_t calls, std::chrono::nanoseconds ns) {
        return static_cast<double>(ns.count()) / calls;
    };

    const std::size_t produced_items = ITER - WARMUP;
    const std::size_t enqueue_total_calls =
        q.get_metrics().enqueue_calls - enqueue_warmup_calls;
    const std::size_t dequeue_total_calls =
        q.get_metrics().dequeue_calls - dequeue_warmup_calls;

    std::cout << "\nProduced / Consumed : " << ITER << " items\n";

    // Throughput per *successful* item
    std::cout << "Producer time       : " << std::fixed << std::setprecision(2)
              << ns_per(produced_items, t_prod) << " ns/op\n";
    std::cout << "Consumer time       : "
              << ns_per(ITER, t_cons) << " ns/op\n";

    // Average cost per enqueue / dequeue *call* (includes retries / polls)
    std::cout << "Enqueue time        : "
              << ns_per(enqueue_total_calls, t_prod) << " ns/enq\n";
    std::cout << "Dequeue time        : "
              << ns_per(dequeue_total_calls, t_cons) << " ns/deq\n\n";
    std::cout << "Memory time        : "
              << ns_per(dequeue_total_calls + enqueue_total_calls, t_cons) << " ns/deq\n\n";
    q.print_metrics();
    return 0;
}

