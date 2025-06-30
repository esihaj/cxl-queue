# ─────────────────────────────────────────────────────────────────────────────
#  Makefile
#    • doorbell_bench        – micro-benchmark for doorbell queue
#    • cxl_mpsc_queue        – benchmark / demo for CXL-resident MPSC queue
#    • test_mpsc_queue       – unit-tests (SPSC focus) in test_mpsc_queue.cpp
#    • cxl_ping_pong         – ping-pong latency benchmark
# ─────────────────────────────────────────────────────────────────────────────

CXX := g++

# ---------------------------------------------------------------------------
#  Common flags
# ---------------------------------------------------------------------------
STD        := -std=c++20
OPT        := -O3 -march=native
THREADING  := -pthread
LDFLAGS    := -lnuma $(THREADING)

# ---------------------------------------------------------------------------
#  ISA-specific flags required by queue implementation
# ---------------------------------------------------------------------------
ISAFLAGS   := -mavx512f -mavx512bw -mclflushopt -mmovdir64b

CXXFLAGS_COMMON := $(STD) $(OPT) $(THREADING)
CXXFLAGS_QUEUE  := $(CXXFLAGS_COMMON) $(ISAFLAGS)

# ---------------------------------------------------------------------------
#  Targets
# ---------------------------------------------------------------------------
TARGETS := doorbell_bench cxl_mpsc_queue test_mpsc_queue cxl_ping_pong
HEADERS := cxl_mpsc_queue.hpp   # ← the queue implementation we want to track

.PHONY: all clean
all: $(TARGETS)

doorbell_bench: doorbell_benchmark.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_QUEUE) $< -o $@ $(LDFLAGS)

cxl_mpsc_queue: cxl_mpsc_queue.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_QUEUE) $< -o $@ $(LDFLAGS)

test_mpsc_queue: test_mpsc_queue.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_QUEUE) $< -o $@ $(LDFLAGS)

cxl_ping_pong: cxl_ping_pong.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_QUEUE) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
