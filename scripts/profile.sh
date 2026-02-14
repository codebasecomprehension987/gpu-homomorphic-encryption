#!/bin/bash

################################################################################
# FHE-CUDA Profiling Script
#
# Profiles GPU performance using NVIDIA Nsight Systems (nsys)
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BUILD_DIR="build"
TEST_EXECUTABLE="${BUILD_DIR}/test_fhe"
PROFILE_DIR="profile_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROFILE_OUTPUT="${PROFILE_DIR}/fhe_profile_${TIMESTAMP}"

# Banner
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}  FHE-CUDA Profiler${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Check for nsys
if ! command -v nsys &> /dev/null; then
    echo -e "${RED}Error: nsys (NVIDIA Nsight Systems) not found${NC}"
    echo -e "${YELLOW}Please install NVIDIA Nsight Systems:${NC}"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

# Check if executable exists
if [ ! -f "${TEST_EXECUTABLE}" ]; then
    echo -e "${RED}Error: Test executable not found at ${TEST_EXECUTABLE}${NC}"
    echo -e "${YELLOW}Build with profiling support:${NC}"
    echo "  make profile"
    exit 1
fi

# Create profile directory
mkdir -p "${PROFILE_DIR}"

echo -e "${GREEN}Starting profiling...${NC}"
echo "Output: ${PROFILE_OUTPUT}.nsys-rep"
echo ""

# Run nsys with comprehensive options
nsys profile \
    --output="${PROFILE_OUTPUT}" \
    --force-overwrite=true \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    "${TEST_EXECUTABLE}"

echo ""
echo -e "${GREEN}✓ Profiling complete${NC}"
echo ""

# Generate statistics report
echo -e "${BLUE}Generating statistics...${NC}"
nsys stats "${PROFILE_OUTPUT}.nsys-rep" > "${PROFILE_OUTPUT}_stats.txt"

echo ""
echo -e "${GREEN}Results:${NC}"
echo "  Profile data: ${PROFILE_OUTPUT}.nsys-rep"
echo "  Statistics:   ${PROFILE_OUTPUT}_stats.txt"
echo ""

# Display key statistics
echo -e "${BLUE}Key Statistics:${NC}"
echo ""

if [ -f "${PROFILE_OUTPUT}_stats.txt" ]; then
    # Show CUDA kernel summary
    echo -e "${YELLOW}Top 10 GPU Kernels by Time:${NC}"
    grep -A 15 "CUDA Kernel Statistics" "${PROFILE_OUTPUT}_stats.txt" | head -20 || true
    echo ""
    
    # Show memory operations
    echo -e "${YELLOW}Memory Operations:${NC}"
    grep -A 10 "CUDA Memory Operation Statistics" "${PROFILE_OUTPUT}_stats.txt" | head -15 || true
    echo ""
fi

# Instructions for GUI
echo -e "${GREEN}To view in GUI:${NC}"
echo "  nsys-ui ${PROFILE_OUTPUT}.nsys-rep"
echo ""

# Alternative: Use nvprof if nsys not available
if ! command -v nsys &> /dev/null && command -v nvprof &> /dev/null; then
    echo -e "${YELLOW}Note: Using legacy nvprof (consider upgrading to nsys)${NC}"
    nvprof --print-gpu-trace --log-file "${PROFILE_OUTPUT}_nvprof.txt" "${TEST_EXECUTABLE}"
fi

exit 0
