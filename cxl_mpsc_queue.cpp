
#include "cxl_mpsc_queue.hpp"

//-------------------------------------------------------------------
//  Sanity / micro-benchmark
//-------------------------------------------------------------------
static void pin_to_cpu0() {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(0, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

int main(int argc, char* argv[]) {
    if (numa_available() < 0) {
        std::cerr << "libnuma not available\n";
        return 1;
    }

    int node = 0;                    // default NUMA node
    size_t ITER     = 1'000'000;

    if (argc >= 2)  node = std::stoi(argv[1]);
    if (argc >= 3)  ITER     = std::stoull(argv[2]);

    if (node < 0 || node > numa_max_node()) {
        std::cerr << "Invalid NUMA node id " << node << '\n';
        return 1;
    }
    
    std::cout << "Allocating ring on NUMA node " << node << '\n';
    std::cout << "Iterations: " << ITER << '\n';

    pin_to_cpu0();  // keep code/data on same node

    constexpr uint32_t ORDER = 14;             // 16 384 entries

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

    auto ns_per = [ITER](std::chrono::nanoseconds ns) {
        return static_cast<double>(ns.count()) / ITER;
    };

    std::cout << "\nProduced / Consumed : " << produced << " / " << consumed << '\n';
    assert(consumed == ITER);
    std::cout << "Producer time       : " << ns_per(t_prod) << " ns/op\n";
    std::cout << "Consumer time       : " << ns_per(t_cons) << " ns/op\n";
    q.print_metrics();
    
    return 0;
}
