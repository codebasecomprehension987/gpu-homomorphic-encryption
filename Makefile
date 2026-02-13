# Makefile for FHE-CUDA Library
# Alternative to CMake for quick builds

# CUDA Configuration
NVCC := nvcc
CUDA_ARCH := -arch=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

# Compiler Flags
NVCC_FLAGS := -O3 -std=c++17 --expt-relaxed-constexpr -Xptxas -v
NVCC_FLAGS += -use_fast_math --generate-line-info
NVCC_FLAGS += -Xcompiler -Wall,-Wextra

# Directories
SRC_DIR := src
KERNEL_DIR := kernels
INCLUDE_DIR := include
BUILD_DIR := build
TEST_DIR := tests

# Include paths
INCLUDES := -I$(INCLUDE_DIR) -I$(KERNEL_DIR)

# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(KERNEL_DIR)/*.cu)
OBJECTS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(notdir $(SOURCES)))

# Test files
TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cu)
TEST_EXECUTABLES := $(patsubst $(TEST_DIR)/%.cu,$(BUILD_DIR)/%,$(TEST_SOURCES))

# Library
LIBRARY := $(BUILD_DIR)/libfhe_cuda.a

# Targets
.PHONY: all clean test lib

all: $(LIBRARY) $(TEST_EXECUTABLES)

# Build library
lib: $(LIBRARY)

$(LIBRARY): $(OBJECTS) | $(BUILD_DIR)
	ar rcs $@ $^
	@echo "Library built: $@"

# Build object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(KERNEL_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Build test executables
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cu $(LIBRARY) | $(BUILD_DIR)
	$(NVCC) $(CUDA_ARCH) $(NVCC_FLAGS) $(INCLUDES) $< -L$(BUILD_DIR) -lfhe_cuda -lcufft -o $@
	@echo "Test executable built: $@"

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run tests
test: $(TEST_EXECUTABLES)
	@echo "Running tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done
	@echo "All tests passed!"

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "FHE-CUDA Build System"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build library and tests (default)"
	@echo "  lib     - Build library only"
	@echo "  test    - Build and run tests"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this message"
	@echo ""
	@echo "Configuration:"
	@echo "  CUDA_ARCH - Target GPU architectures (currently: $(CUDA_ARCH))"
	@echo "  NVCC      - CUDA compiler (currently: $(NVCC))"

# Performance benchmark target
benchmark: $(BUILD_DIR)/test_fhe
	@echo "Running performance benchmark..."
	$(BUILD_DIR)/test_fhe --benchmark

# Install (system-wide)
install: $(LIBRARY)
	@echo "Installing FHE-CUDA library..."
	install -d /usr/local/lib
	install -m 644 $(LIBRARY) /usr/local/lib/
	install -d /usr/local/include/fhe
	install -m 644 $(INCLUDE_DIR)/*.cuh /usr/local/include/fhe/
	@echo "Installation complete"

# Uninstall
uninstall:
	@echo "Uninstalling FHE-CUDA library..."
	rm -f /usr/local/lib/libfhe_cuda.a
	rm -rf /usr/local/include/fhe
	@echo "Uninstallation complete"

# Debug build
debug: NVCC_FLAGS += -g -G -lineinfo
debug: clean all
	@echo "Debug build complete"

# Profile build
profile: NVCC_FLAGS += -lineinfo
profile: clean all
	@echo "Profile build complete (use with nvprof or nsys)"

# Documentation generation (requires Doxygen)
docs:
	@if command -v doxygen >/dev/null 2>&1; then \
		echo "Generating documentation..."; \
		doxygen Doxyfile; \
	else \
		echo "Error: Doxygen not found. Install with: sudo apt-get install doxygen"; \
	fi
