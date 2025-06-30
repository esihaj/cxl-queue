// cxl_ping_pong.cpp
// ─────────────────────────────────────────────────────────────────────────────
//  Ping‑pong benchmark for CxlMpscQueue (single‑producer ⇆ single‑consumer)
//  • Two‑field payload (args[0], args[1])
//  • rpc_id populated and echoed back
//  • Verbose logs: client/server send‑/receive events each iteration
// ─────────────────────────────────────────────────────────────────────────────
//  Build:
//      g++ -std=c++20 -O3 -march=native -pthread -lnuma cxl_ping_pong.cpp -o cxl_ping_pong
//
//  Usage:
//      ./cxl_ping_pong [numa_node] [iters]
//          numa_node  – NUMA node to allocate queue buffers (default 0)
//          iters      – number of ping‑pong iterations (default 1'000'000)
//
//  NOTE: The per‑iteration logs are useful for functional debug. For large
//  iters values, stdout will dominate run time. Trim or gate logs if needed.
// ─────────────────────────────────────────────────────────────────────────────

#include "cxl_mpsc_queue.hpp"
#include <iomanip>
#include <pthread.h>
#include <cstring>
#include <cstdint>
#include <syncstream>

static void pin_to_cpu(unsigned cpu)
{
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0)
        perror("pthread_setaffinity_np");
}

constexpr uint32_t ORDER = 14;           // 16 Ki‑entry ring (capacity = 16384)
using Steady = std::chrono::steady_clock;

int main(int argc, char* argv[])
{
    int    numa_node = 0;
    size_t iters     = 1'000'000;

    if (argc >= 2)  numa_node = std::stoi(argv[1]);
    if (argc >= 3)  iters     = std::stoull(argv[2]);

    std::cout << "Allocating queues on NUMA node " << numa_node << '\n';
    std::cout << "Iterations: " << iters << '\n';

    const size_t cap = 1u << ORDER;

    // ───── allocate queue memory on NUMA node ───────────────────────────
    Entry*     req_ring = static_cast<Entry*>(numa_alloc_onnode(sizeof(Entry)*cap, numa_node));
    uint64_t*  req_tail = static_cast<uint64_t*>(numa_alloc_onnode(64,                numa_node));
    Entry*     rsp_ring = static_cast<Entry*>(numa_alloc_onnode(sizeof(Entry)*cap, numa_node));
    uint64_t*  rsp_tail = static_cast<uint64_t*>(numa_alloc_onnode(64,                numa_node));

    std::memset(req_ring, 0, sizeof(Entry)*cap);
    std::memset(rsp_ring, 0, sizeof(Entry)*cap);
    std::memset(req_tail, 0, 64);
    std::memset(rsp_tail, 0, 64);

    CxlMpscQueue q_req(req_ring, ORDER, req_tail);   // client → server
    CxlMpscQueue q_rsp(rsp_ring, ORDER, rsp_tail);   // server → client

    std::atomic<bool> server_ready{false};

    // ───── server thread ────────────────────────────────────────────────
    std::thread server([&]{
        pin_to_cpu(1);
        server_ready.store(true, std::memory_order_release);

        Entry req{}, rsp{};
        for (size_t i = 0; i < iters; ++i) {
            while (!q_req.dequeue(req)) { /* spin */ }
            // std::osyncstream(std::cout)
            //     << "[server] recv  i=" << i
            //     << " rpc_id=" << req.meta.f.rpc_id << '\n';

            // (optional) validate
            if (req.meta.f.rpc_id != static_cast<uint16_t>(i & 0xFFFF) ||
                req.args[0] != i) {
                std::cerr << "[server] validation error on i=" << i << std::endl;
                std::abort();
            }

            rsp = req;                                // echo back
            while (!q_rsp.enqueue(rsp)) { /* spin */ }
            // std::osyncstream(std::cout)
            //     << "[server] sent  i=" << i
            //     << " rpc_id=" << rsp.meta.f.rpc_id << '\n';
        }
    });

    // Wait until server thread ready
    while (!server_ready.load(std::memory_order_acquire))
        std::this_thread::yield();

    pin_to_cpu(0);             // client on CPU‑0

    Entry req{}, rsp{};
    const auto t0 = Steady::now();

    for (size_t i = 0; i < iters; ++i) {
        req.args[0]        = i;        // first argument
        req.meta.f.rpc_id  = static_cast<uint16_t>(i & 0xFFFF);
        req.meta.f.rpc_method = 0;

        // std::osyncstream(std::cout)
        //     << "[client] send  i=" << i
        //     << " rpc_id=" << req.meta.f.rpc_id << '\n';

        while (!q_req.enqueue(req)) { /* spin */ }
        while (!q_rsp.dequeue(rsp)) { /* spin */ }

        // std::osyncstream(std::cout)
        //     << "[client] recv  i=" << i
        //     << " rpc_id=" << rsp.meta.f.rpc_id << '\n';

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
    std::cout << "\nIterations           : " << iters          << '\n'
              << "Total elapsed (ms)   : " << total_ns / 1e6  << '\n'
              << "Round‑trip latency/ns: " << rtt_ns         << '\n'
              << "One‑way latency/ns   : " << rtt_ns / 2.0   << '\n';

    std::cout << "\n[queue stats]\n";
    q_req.print_metrics("request");
    std::cout << '\n';
    q_rsp.print_metrics("response");
    // ───── cleanup ──────────────────────────────────────────────────────
    numa_free(req_ring, sizeof(Entry)*cap);
    numa_free(rsp_ring, sizeof(Entry)*cap);
    numa_free(req_tail, 64);
    numa_free(rsp_tail, 64);
    return 0;
}
