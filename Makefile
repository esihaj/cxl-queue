# ---------------------------------------------------------------------------
#  Makefile: doorbell_bench  ⟷  cxl_mpsc_queue
# ---------------------------------------------------------------------------

CXX := g++

# ---------------------------------------------------------------------------
#  Shared options
# ---------------------------------------------------------------------------
STD      := -std=c++20
OPT      := -O3 -march=native
LDFLAGS  := -lnuma

# ---------------------------------------------------------------------------
#  Target-specific compile flags
# ---------------------------------------------------------------------------
DOORBELL_CXXFLAGS := $(STD) $(OPT) -mavx512f -mavx512bw -mclflushopt -mmovdir64b -pthread
CXL_CXXFLAGS      := $(STD) $(OPT) -pthread

# ---------------------------------------------------------------------------
#  Targets and sources
# ---------------------------------------------------------------------------
TARGETS := doorbell_bench cxl_mpsc_queue

.PHONY: all clean
all: $(TARGETS)

doorbell_bench: doorbell_benchmark.cpp
	$(CXX) $(DOORBELL_CXXFLAGS) $< -o $@ $(LDFLAGS)

cxl_mpsc_queue: cxl_mpsc_queue.cpp
	$(CXX) $(CXL_CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS)
