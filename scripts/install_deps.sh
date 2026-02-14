#!/bin/bash

################################################################################
# FHE-CUDA Dependency Installation Script
#
# Installs required dependencies for building and running FHE-CUDA
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Banner
echo -e "${BLUE}=================================${NC}"
echo -e "${BLUE}  FHE-CUDA Dependency Installer${NC}"
echo -e "${BLUE}=================================${NC}"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    echo -e "${RED}Error: Cannot detect OS${NC}"
    exit 1
fi

echo -e "${GREEN}Detected OS: ${OS} ${VERSION}${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install based on OS
case "${OS}" in
    ubuntu|debian)
        echo -e "${BLUE}Installing dependencies for Ubuntu/Debian...${NC}"
        echo ""
        
        # Update package list
        echo "Updating package list..."
        sudo apt update
        
        # Install build essentials
        echo "Installing build essentials..."
        sudo apt install -y build-essential
        
        # Install CMake
        if ! command_exists cmake; then
            echo "Installing CMake..."
            sudo apt install -y cmake
        else
            echo "CMake already installed: $(cmake --version | head -1)"
        fi
        
        # Install Git
        if ! command_exists git; then
            echo "Installing Git..."
            sudo apt install -y git
        else
            echo "Git already installed: $(git --version)"
        fi
        
        # Check for CUDA
        if ! command_exists nvcc; then
            echo -e "${YELLOW}CUDA Toolkit not found${NC}"
            echo "Would you like to install CUDA 12.3? (y/n)"
            read -r response
            if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
                echo "Installing CUDA Toolkit..."
                
                # Add CUDA repository
                wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
                sudo dpkg -i cuda-keyring_1.1-1_all.deb
                sudo apt update
                sudo apt install -y cuda-toolkit-12-3
                
                # Add to PATH
                echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
                echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
                
                echo -e "${GREEN}CUDA installed. Please run 'source ~/.bashrc' or restart terminal${NC}"
            else
                echo -e "${YELLOW}Skipping CUDA installation${NC}"
                echo "Download from: https://developer.nvidia.com/cuda-downloads"
            fi
        else
            echo "CUDA already installed: $(nvcc --version | grep release)"
        fi
        
        # Install optional tools
        echo ""
        echo "Installing optional development tools..."
        sudo apt install -y \
            python3-dev \
            python3-pip \
            gdb \
            valgrind || true
        
        ;;
        
    centos|rhel|fedora)
        echo -e "${BLUE}Installing dependencies for CentOS/RHEL/Fedora...${NC}"
        echo ""
        
        # Install development tools
        echo "Installing development tools..."
        sudo yum groupinstall -y "Development Tools"
        
        # Install CMake
        if ! command_exists cmake; then
            echo "Installing CMake..."
            sudo yum install -y cmake3
            sudo ln -sf /usr/bin/cmake3 /usr/local/bin/cmake
        fi
        
        # Install Git
        if ! command_exists git; then
            sudo yum install -y git
        fi
        
        # Check for CUDA
        if ! command_exists nvcc; then
            echo -e "${YELLOW}CUDA Toolkit not found${NC}"
            echo "Please install from: https://developer.nvidia.com/cuda-downloads"
        fi
        
        ;;
        
    arch|manjaro)
        echo -e "${BLUE}Installing dependencies for Arch Linux...${NC}"
        echo ""
        
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm base-devel cmake git
        
        if ! command_exists nvcc; then
            echo -e "${YELLOW}Install CUDA from AUR:${NC}"
            echo "  yay -S cuda"
        fi
        
        ;;
        
    *)
        echo -e "${YELLOW}Unsupported OS: ${OS}${NC}"
        echo "Please install manually:"
        echo "  - GCC/Clang compiler"
        echo "  - CMake >= 3.18"
        echo "  - CUDA Toolkit >= 11.0"
        echo "  - Git"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=================================${NC}"
echo -e "${GREEN}  Dependency Check${NC}"
echo -e "${GREEN}=================================${NC}"
echo ""

# Verify installations
ERRORS=0

# Check GCC/Clang
if command_exists gcc; then
    echo -e "${GREEN}✓${NC} GCC: $(gcc --version | head -1)"
elif command_exists clang; then
    echo -e "${GREEN}✓${NC} Clang: $(clang --version | head -1)"
else
    echo -e "${RED}✗${NC} No C++ compiler found"
    ERRORS=1
fi

# Check CMake
if command_exists cmake; then
    CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
    if awk "BEGIN {exit !($CMAKE_VERSION >= 3.18)}"; then
        echo -e "${GREEN}✓${NC} CMake: $(cmake --version | head -1)"
    else
        echo -e "${YELLOW}⚠${NC} CMake version too old: $CMAKE_VERSION (need >= 3.18)"
        ERRORS=1
    fi
else
    echo -e "${RED}✗${NC} CMake not found"
    ERRORS=1
fi

# Check Git
if command_exists git; then
    echo -e "${GREEN}✓${NC} Git: $(git --version)"
else
    echo -e "${RED}✗${NC} Git not found"
    ERRORS=1
fi

# Check CUDA
if command_exists nvcc; then
    echo -e "${GREEN}✓${NC} CUDA: $(nvcc --version | grep release)"
    
    # Check compute capability
    if command_exists nvidia-smi; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        if awk "BEGIN {exit !($COMPUTE_CAP >= 7.5)}"; then
            echo -e "${GREEN}✓${NC} GPU Compute Capability: $COMPUTE_CAP"
        else
            echo -e "${YELLOW}⚠${NC} GPU Compute Capability: $COMPUTE_CAP (need >= 7.5)"
        fi
    fi
else
    echo -e "${RED}✗${NC} CUDA not found"
    ERRORS=1
fi

echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All dependencies satisfied!${NC}"
    echo ""
    echo -e "${BLUE}Ready to build:${NC}"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make -j\$(nproc)"
else
    echo -e "${YELLOW}Some dependencies are missing or outdated${NC}"
    echo "Please install them manually before building"
fi

echo ""
echo -e "${BLUE}Additional Optional Tools:${NC}"

# Check optional tools
if command_exists nsys; then
    echo -e "${GREEN}✓${NC} NVIDIA Nsight Systems (for profiling)"
else
    echo -e "${YELLOW}○${NC} NVIDIA Nsight Systems (optional for profiling)"
    echo "    Download: https://developer.nvidia.com/nsight-systems"
fi

if command_exists gdb; then
    echo -e "${GREEN}✓${NC} GDB (for debugging)"
else
    echo -e "${YELLOW}○${NC} GDB (optional for debugging)"
fi

if command_exists python3; then
    echo -e "${GREEN}✓${NC} Python 3 (for scripts)"
else
    echo -e "${YELLOW}○${NC} Python 3 (optional for scripts)"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"

exit 0
