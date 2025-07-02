// cxl_ping_pong.cpp
// ─────────────────────────────────────────────────────────────────────────────
//  Ping-pong benchmark for CxlMpscQueue (single-producer ⇆ single-consumer)
//  • Two-field payload (args[0], args[1])
//  • rpc_id populated and echoed back
//  • Verbose logs: client/server send-/receive events each iteration
// ─────────────────────────────────────────────────────────────────────────────
//  Build:
//      g++ -std=c++20 -O3 -march=native -pthread -lnuma \
//          cxl_ping_pong.cpp -o cxl_ping_pong
//
//  Usage:
//      ./cxl_ping_pong pin <cpu_id> numa <node_id> [iter_count]
//      ./cxl_ping_pong pin <cpu_id> dax            [iter_count]
//
//      cpu_id      – logical CPU the *client* thread is pinned to
//      node_id     – NUMA node from which DRAM is allocated
//      iter_count  – ping-pong iterations (default 1'000'000)
// ─────────────────────────────────────────────────────────────────────────────

#include "cxl_mpsc_queue.hpp"
#include "cxl_allocator.hpp"          // ← NEW
#include <iomanip>
#include <pthread.h>
#include <cstring>
#include <cstdint>
#include <syncstream>
#include <memory>
#include <string_view>

static void pin_to_cpu(unsigned cpu)
{
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        perror("pthread_setaffinity_np");
}

// ─── helper: CLI usage ──────────────────────────────────────────────────────
static void print_usage(const char* prog)
{
    std::cerr << "Usage:\n"
              << "  " << prog << " pin <cpu_id> numa <node_id> [iter_count]\n"
              << "  " << prog << " pin <cpu_id> dax            [iter_count]\n"
              << "    iter_count defaults to 1'000'000 (1M)\n";
}

constexpr uint32_t ORDER = 14;           // 16 Ki-entry ring (capacity = 16384)
using Steady = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    // ───── parse arguments ────────────────────────────────────────────────
    if (argc < 4) { print_usage(argv[0]); return 1; }

    if (std::string_view(argv[1]) != "pin") { print_usage(argv[0]); return 1; }
    const unsigned client_cpu = std::stoul(argv[2]);

    std::unique_ptr<cxl::CxlAllocator> alloc;
    size_t iters = 1'000'000;

    const std::string_view mem_kind = argv[3];

    if (mem_kind == "numa") {
        if (argc < 5) { print_usage(argv[0]); return 1; }
        const int numa_node = std::stoi(argv[4]);
        if (argc >= 6) iters = std::stoull(argv[5]);
        alloc = std::make_unique<cxl::NumaAllocator>(numa_node);
        std::cout << "Allocator: NUMA node " << numa_node << '\n';
    }
    else if (mem_kind == "dax") {
        if (argc >= 5) iters = std::stoull(argv[4]);
        alloc = std::make_unique<cxl::DaxAllocator>();           // default /dev/dax1.0
        std::cout << "Allocator: DAX (/dev/dax1.0 slice)\n";
    }
    else {
        print_usage(argv[0]); return 1;
    }

    std::cout << "Client pinned to CPU " << client_cpu << '\n';
    std::cout << "Iterations           : " << iters << '\n';

    const size_t cap = 1u << ORDER;

    // ───── allocate queue memory via CXL allocator ───────────────────────
    Entry*     req_ring = static_cast<Entry*>   (alloc->allocate_aligned(sizeof(Entry) * cap));
    uint64_t*  req_tail = static_cast<uint64_t*>(alloc->allocate_aligned(64));
    Entry*     rsp_ring = static_cast<Entry*>   (alloc->allocate_aligned(sizeof(Entry) * cap));
    uint64_t*  rsp_tail = static_cast<uint64_t*>(alloc->allocate_aligned(64));

    std::memset(req_ring, 0, sizeof(Entry) * cap);
    std::memset(rsp_ring, 0, sizeof(Entry) * cap);
    std::memset(req_tail, 0, 64);
    std::memset(rsp_tail, 0, 64);

    // ───── construct queues ──────────────────────────────────────────────
    CxlMpscQueue q_req(req_ring, ORDER, req_tail);   // client → server
    CxlMpscQueue q_rsp(rsp_ring, ORDER, rsp_tail);   // server → client

    std::atomic<bool> server_ready{false};

    // ───── server thread ────────────────────────────────────────────────
    std::thread server([&, client_cpu]{
        pin_to_cpu((client_cpu + 1) % std::thread::hardware_concurrency());
        server_ready.store(true, std::memory_order_release);

        Entry req{}, rsp{};
        for (size_t i = 0; i < iters; ++i) {
            while (!q_req.dequeue(req)) { /* spin */ }

            // (optional) validate
            if (req.meta.f.rpc_id != static_cast<uint16_t>(i & 0xFFFF) ||
                req.args[0] != i) {
                std::cerr << "[server] validation error on i=" << i << std::endl;
                std::abort();
            }

            rsp = req;                                // echo back
            while (!q_rsp.enqueue(rsp)) { /* spin */ }
        }
    });

    // Wait until server thread ready
    while (!server_ready.load(std::memory_order_acquire))
        std::this_thread::yield();

    pin_to_cpu(client_cpu);        // client thread

    Entry req{}, rsp{};
    const auto t0 = Steady::now();

    for (size_t i = 0; i < iters; ++i) {
        req.args[0]            = i;                       // first argument
        req.meta.f.rpc_id      = static_cast<uint16_t>(i & 0xFFFF);
        req.meta.f.rpc_method  = 0;

        while (!q_req.enqueue(req)) { /* spin */ }
        while (!q_rsp.dequeue(rsp)) { /* spin */ }

        // client-side validation
        if (rsp.meta.f.rpc_id != req.meta.f.rpc_id ||
            rsp.args[0] != req.args[0]) {
            std::cerr << "[client] validation error on i=" << i << std::endl;
            return 1;
        }
    }

    const auto t1 = Steady::now();
    server.join();

    // ───── results ──────────────────────────────────────────────────────
    const double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double rtt_ns   = total_ns / static_cast<double>(iters);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nTotal elapsed (ms)   : " << total_ns / 1e6  << '\n'
              << "Round-trip latency/ns: " << rtt_ns           << '\n'
              << "One-way latency/ns   : " << rtt_ns / 2.0     << '\n';

    std::cout << "\n[queue stats]\n";
    q_req.print_metrics("request");
    std::cout << '\n';
    q_rsp.print_metrics("response");

    // No explicit free — allocator releases memory in its destructor
    return 0;
}
