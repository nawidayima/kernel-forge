BUILD_DIR := build
RESULTS_DIR := benchmark_results

.PHONY: build run profile profile-fast bench plot clean

build:
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j

run: build
	@./$(BUILD_DIR)/$(EX) $(K)

# Full capture: every metric set, replays the kernel many times, slow.
# Use for the final deep-dive pass; opens in Nsight Compute GUI.
profile: build
	@mkdir -p $(RESULTS_DIR)
	@ncu --set full --export $(RESULTS_DIR)/$(EX)_kernel_$(K) --force-overwrite ./$(BUILD_DIR)/$(EX) $(K)

# Fast iteration: targeted sections printed to stdout, no .ncu-rep round trip.
# Diagnostic order: SpeedOfLight (memory-vs-compute bound?) → MemoryWorkloadAnalysis
# (coalescing, sectors/requests) → WarpStateStats (stall breakdown) → Occupancy.
profile-fast: build
	@ncu --section SpeedOfLight --section MemoryWorkloadAnalysis \
	     --section WarpStateStats --section Occupancy \
	     ./$(BUILD_DIR)/$(EX) $(K)

# Sweep kernels x sizes for one exercise. Overrides:
#   make bench EX=matmul KERNELS="0 1 2" SIZES="512 1024 2048 4096"
bench: build
	@EX=$(EX) OUT=$(RESULTS_DIR) BUILD=$(BUILD_DIR) \
	    KERNELS="$(KERNELS)" SIZES="$(SIZES)" \
	    ./scripts/benchmark.sh

plot:
	@python3 scripts/plot.py $(RESULTS_DIR)/$(EX).csv

clean:
	@rm -rf $(BUILD_DIR)
