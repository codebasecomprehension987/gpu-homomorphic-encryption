#!/bin/bash

################################################################################
# FHE-CUDA Benchmark Script
#
# Runs comprehensive performance benchmarks and generates reports
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="build"
TEST_EXECUTABLE="${BUILD_DIR}/test_fhe"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"

# Banner
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}  FHE-CUDA Benchmark Suite${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Check if build exists
if [ ! -f "${TEST_EXECUTABLE}" ]; then
    echo -e "${RED}Error: Test executable not found at ${TEST_EXECUTABLE}${NC}"
    echo -e "${YELLOW}Please build the project first:${NC}"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make"
    exit 1
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Get GPU info
echo -e "${GREEN}GPU Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader | head -1
echo ""

# Save GPU info to file
echo "FHE-CUDA Benchmark Results" > "${RESULTS_FILE}"
echo "Timestamp: ${TIMESTAMP}" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"
echo "GPU Information:" >> "${RESULTS_FILE}"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv >> "${RESULTS_FILE}"
echo "" >> "${RESULTS_FILE}"

# Run benchmarks
echo -e "${GREEN}Running benchmarks...${NC}"
echo "This may take a few minutes..."
echo ""

# Run test executable with benchmark flag (if supported)
echo "Test Results:" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"

if ${TEST_EXECUTABLE} --benchmark 2>&1 | tee -a "${RESULTS_FILE}"; then
    echo ""
    echo -e "${GREEN}✓ Benchmarks completed successfully${NC}"
else
    # If --benchmark flag not supported, run normal tests
    echo -e "${YELLOW}Note: Running standard tests (no benchmark flag)${NC}"
    ${TEST_EXECUTABLE} 2>&1 | tee -a "${RESULTS_FILE}"
fi

echo "" >> "${RESULTS_FILE}"

# Performance analysis
echo -e "${GREEN}Performance Analysis:${NC}"
echo ""
echo "Performance Analysis:" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"

# Extract timing information (if available in output)
if grep -q "time:" "${RESULTS_FILE}"; then
    echo "Key Operations:" >> "${RESULTS_FILE}"
    grep "time:" "${RESULTS_FILE}" | tail -10 >> "${RESULTS_FILE}"
fi

# Summary
echo ""
echo -e "${GREEN}Results saved to: ${RESULTS_FILE}${NC}"
echo ""

# Optional: Compare with previous benchmarks
PREV_RESULTS=$(ls -t ${RESULTS_DIR}/benchmark_*.txt 2>/dev/null | head -2 | tail -1)
if [ -n "${PREV_RESULTS}" ] && [ "${PREV_RESULTS}" != "${RESULTS_FILE}" ]; then
    echo -e "${YELLOW}Previous benchmark: ${PREV_RESULTS}${NC}"
    echo "Run 'diff ${PREV_RESULTS} ${RESULTS_FILE}' to compare"
    echo ""
fi

# Generate summary table
echo "Benchmark Summary" >> "${RESULTS_FILE}"
echo "======================================" >> "${RESULTS_FILE}"
echo "See full results above" >> "${RESULTS_FILE}"

echo -e "${GREEN}Benchmark complete!${NC}"
echo ""

# Optional: Generate plots (requires gnuplot)
if command -v gnuplot &> /dev/null; then
    echo -e "${BLUE}Generating plots...${NC}"
    # Add gnuplot commands here if needed
    echo "  (Plot generation not yet implemented)"
else
    echo -e "${YELLOW}Note: Install gnuplot for automatic plot generation${NC}"
fi

exit 0
