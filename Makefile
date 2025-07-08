# ─────────────────────────────────────────────────────────────────────────────
#  Makefile (trimmed but with doorbell extras)
# ─────────────────────────────────────────────────────────────────────────────

CXX := g++

# ---------------------------------------------------------------------------
#  Common flags
# ---------------------------------------------------------------------------
STD        := -std=c++20
OPT        := -O2 -g -march=native -fno-omit-frame-pointer
# -fno-inline is useful for profiling
THREADING  := -pthread
LDFLAGS    := -lnuma $(THREADING)

ISAFLAGS   := -mavx512f -mavx512bw -mclflushopt -mmovdir64b

CXXFLAGS_COMMON := $(STD) $(OPT) $(THREADING) $(ISAFLAGS)

# ---------------------------------------------------------------------------
#  Sources / headers
# ---------------------------------------------------------------------------
HEADERS := cxl_allocator.hpp cxl_mpsc_queue.hpp   # queue implementation

# ---------------------------------------------------------------------------
#  Binaries (unchanged simple one-liners)
# ---------------------------------------------------------------------------
.PHONY: all clean
all: doorbell_bench cxl_mpsc_queue test_mpsc_queue cxl_ping_pong

doorbell_bench: doorbell_benchmark.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_COMMON) $< -o $@ $(LDFLAGS)

# Intel-syntax assembly only (no .o left behind)
doorbell_asm: doorbell_benchmark.s

doorbell_benchmark.s: doorbell_benchmark.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_COMMON) -S -fverbose-asm -masm=intel $< -o $@

cxl_mpsc_queue: cxl_mpsc_queue.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_COMMON) $< -o $@ $(LDFLAGS)

test_mpsc_queue: test_mpsc_queue.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_COMMON) $< -o $@ $(LDFLAGS)

cxl_ping_pong: cxl_ping_pong.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS_COMMON) $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------------
#  House-keeping
# ---------------------------------------------------------------------------
clean:
	rm -f doorbell_bench cxl_mpsc_queue test_mpsc_queue cxl_ping_pong
	rm -f doorbell_benchmark.s
