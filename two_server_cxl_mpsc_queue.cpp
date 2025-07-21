// cxl_mpsc_queue.cpp
// ─────────────────────────────────────────────────────────────────────────────
//  Sanity / micro-benchmark using CXL-backed allocators for two processes.
//
//  CLI (choose **one** form) ––––––––––––––––––––––––––––––––––––––––––––––
//    <producer|consumer> pin <cpu_id> dax [iter_count]
//
//    • producer|consumer : Role of this process
//    • cpu_id            : logical CPU to pin the main thread to
//    • iter_count        : #iterations (default = 10’000’000 = 10 M)
//
//  Examples
//    # On machine 1 (Producer)
//    sudo ./cxl_mpsc_queue producer pin 15 dax 20000000
//
//    # On machine 2 (Consumer)
//    sudo ./cxl_mpsc_queue consumer pin 3 dax 20000000
//
//  Build (GCC ≥ 12 or Clang ≥ 15)
//    g++ -O3 -std=c++20 -march=native -mavx512f -mavx512bw \
//        -mclflushopt -mclwb -mmovdir64b  \
//        cxl_mpsc_queue.cpp  -o cxl_mpsc_queue \
//        -lnuma -lpthread
// ─────────────────────────────────────────────────────────────────────────────

#include "cxl_mpsc_queue_exp.hpp"
#include "cxl_allocator.hpp" // Assuming this file exists and is correct

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
#include <vector>

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
        "usage  : " << prog << " <producer|consumer> pin <cpu_id> dax [iter_count]\n"
        "notes  : iter_count defaults to 10M when omitted\n"
        "       : 'dax' mode is required for multi-process test\n";
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
    if (argc < 5) print_usage(argv[0]);

    const std::string role = argv[1];
    if (role != "producer" && role != "consumer") print_usage(argv[0]);

    if (std::string{argv[2]} != "pin") print_usage(argv[0]);
    const int         cpu_id = std::stoi(argv[3]);
    const std::string mode   = argv[4];

    if (mode != "dax") {
        std::cerr << "Error: Two-process mode requires 'dax' allocator.\n";
        print_usage(argv[0]);
    }

    std::size_t ITER = (argc >= 6) ? std::stoull(argv[5]) : DEFAULT_ITERS;

    //-----------------------------------------------------------------------
    //  Pin main thread and create allocator
    //-----------------------------------------------------------------------
    pin_to_cpu(cpu_id);

    std::unique_ptr<cxl::CxlAllocator> alloc;
    try {
        alloc = std::make_unique<cxl::DaxAllocator>();
        std::cout << "[" << role << "] Using DAX allocator on /dev/dax* slice\n";
    } catch (const std::exception& ex) {
        std::cerr << "Allocator init failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "[" << role << "] Pinned to CPU " << cpu_id << '\n'
              << "[" << role << "] Iterations      : " << ITER << "\n\n";

    //-----------------------------------------------------------------------
    //  Queue setup – allocate from CXL allocator
    //-----------------------------------------------------------------------
    constexpr uint32_t ORDER       = 14; // 16,384 entries
    const std::size_t  RING_BYTES  = (1u << ORDER) * sizeof(Entry);

    // Allocate all shared memory regions
    Entry* ring             = static_cast<Entry*>   (alloc->allocate_aligned(RING_BYTES, 64));
    uint64_t* tail_cxl         = static_cast<uint64_t*>(alloc->allocate_aligned(sizeof(uint64_t), 64));
    uint64_t* producer_ready   = static_cast<uint64_t*>(alloc->allocate_aligned(sizeof(uint64_t), 64));
    uint64_t* consumer_ready   = static_cast<uint64_t*>(alloc->allocate_aligned(sizeof(uint64_t), 64));
    uint64_t* start_signal     = static_cast<uint64_t*>(alloc->allocate_aligned(sizeof(uint64_t), 64));

    //-----------------------------------------------------------------------
    //  Run process-specific logic
    //-----------------------------------------------------------------------
    if (role == "producer") {
        // Producer initializes all shared memory, including sync flags
        store_nt_u64(producer_ready, 0);
        store_nt_u64(consumer_ready, 0);
        store_nt_u64(start_signal, 0);

        CxlMpscQueue q_producer(ring, ORDER, tail_cxl, /*do_initialize=*/true);

        // --- Warm-up phase ---
        std::cout << "[producer] Warming up...\n";
        const std::size_t WARMUP = q_producer.capacity() / 4;
        assert(WARMUP < ITER && "warm-up must be < total iterations");
        Entry e{};
        e.meta.f.rpc_method = 1;
        e.meta.f.seal_index = -1;
        for (std::size_t i = 0; i < WARMUP; ++i) {
            e.meta.f.rpc_id = static_cast<uint16_t>(i);
            while (!q_producer.enqueue(e, false)) { /* spin */ }
        }

        // --- Handshake ---
        std::cout << "[producer] Warm-up complete. Signaling readiness.\n";
        store_nt_u64(producer_ready, 1);

        std::cout << "[producer] Waiting for consumer...\n";
        while (load_fresh_u64(consumer_ready) == 0) { cpu_relax_for_cycles(100); }

        std::cout << "[producer] Consumer ready. Starting benchmark.\n";
        store_nt_u64(start_signal, 1);
        
        // --- Timed phase ---
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = WARMUP; i < ITER; ++i) {
            e.meta.f.rpc_id = static_cast<uint16_t>(i);
            
            // Loop until enqueue succeeds, with debug logging enabled.
            // The internal backoff in enqueue will prevent busy-spinning.
            while (!q_producer.enqueue(e, false)) {}

            // Log every successful enqueue operation.
            // std::osyncstream(std::cout) << "[producer] Successfully enqueued item " << i << ".\n";
        }
        const auto t_prod = std::chrono::steady_clock::now() - t0;

        // --- Results ---
        const auto ns_per = [](std::size_t calls, std::chrono::nanoseconds ns) {
            if (calls == 0) return 0.0;
            return static_cast<double>(ns.count()) / calls;
        };
        const std::size_t produced_items = ITER - WARMUP;
        std::cout << "\n[producer] Producer time: " << std::fixed << std::setprecision(2)
                  << ns_per(produced_items, t_prod) << " ns/op\n";
        q_producer.print_metrics("Producer");

    } else { // Consumer role
        // --- Handshake ---
        std::cout << "[consumer] Waiting for producer to be ready...\n";
        while (load_fresh_u64(producer_ready) == 0) { cpu_relax_for_cycles(100); }

        CxlMpscQueue q_consumer(ring, ORDER, tail_cxl, /*do_initialize=*/false);

        std::cout << "[consumer] Producer is ready. Signaling own readiness.\n";
        store_nt_u64(consumer_ready, 1);

        std::cout << "[consumer] Waiting for start signal...\n";
        while (load_fresh_u64(start_signal) == 0) { cpu_relax_for_cycles(100); }

        std::cout << "[consumer] Start signal received. Beginning consumption.\n";

        // --- Timed phase ---
        Entry e{};
        std::size_t consumed = 0;
        const auto t0 = std::chrono::steady_clock::now();
        while (consumed < ITER) {
            // Attempt to dequeue with debug logging enabled.
            // The internal backoff in dequeue will prevent busy-spinning.
            if (q_consumer.dequeue(e, true)) {
                consumed++;
                if (e.meta.f.rpc_id != static_cast<uint16_t>(consumed)) {
                    std::osyncstream(std::cerr) << "[consumer] VERIFICATION FAILED! "
                                                << "Expected rpc_id: " << consumed
                                                << ", but got: " << e.meta.f.rpc_id << ".\n";
                    exit(EXIT_FAILURE); // Exit immediately on data corruption.
                }
            }
        }
        const auto t_cons = std::chrono::steady_clock::now() - t0;

        // --- Results ---
        const auto ns_per = [](std::size_t calls, std::chrono::nanoseconds ns) {
            if (calls == 0) return 0.0;
            return static_cast<double>(ns.count()) / calls;
        };
        std::cout << "\n[consumer] Consumer time: " << std::fixed << std::setprecision(2)
                  << ns_per(ITER, t_cons) << " ns/op\n";
        q_consumer.print_metrics("Consumer");
    }

    // Deallocation is left to the OS on process exit for this benchmark.
    // In a real app, a more robust cleanup mechanism would be needed.
    return 0;
}
