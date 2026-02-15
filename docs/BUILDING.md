# Building FHE-CUDA

Comprehensive build instructions for all platforms and configurations.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Platform-Specific Instructions](#platform-specific-instructions)
4. [Build Configurations](#build-configurations)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Configuration](#advanced-configuration)

---

## Prerequisites

### Required Software

#### CUDA Toolkit
```bash
# Minimum version: 11.0
# Recommended: 12.0 or later

# Check CUDA version
nvcc --version
```

**Download:** https://developer.nvidia.com/cuda-downloads

#### CMake
```bash
# Minimum version: 3.18
# Recommended: 3.20 or later

# Check CMake version
cmake --version
```

**Download:** https://cmake.org/download/

#### C++ Compiler
- **Linux**: GCC 7.0+ or Clang 5.0+
- **Windows**: Visual Studio 2019 or later
- **macOS**: Not officially supported (no NVIDIA GPUs)

### Hardware Requirements

#### GPU
- **Minimum**: NVIDIA GPU with Compute Capability 7.5 (Turing)
- **Recommended**: Compute Capability 8.0+ (Ampere or newer)
- **Supported Architectures**:
  - SM 7.5: Turing (RTX 20-series, GTX 16-series)
  - SM 8.0: Ampere (A100, RTX 30-series)
  - SM 8.6: Ampere (RTX 30-series mobile)
  - SM 8.9: Ada Lovelace (RTX 40-series)
  - SM 9.0: Hopper (H100)

#### Memory
- **Minimum**: 4GB VRAM
- **Recommended**: 8GB+ VRAM for N=8192 or higher

#### System RAM
- **Minimum**: 8GB
- **Recommended**: 16GB+

### Check Your System

```bash
# Check GPU
nvidia-smi

# Check CUDA capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Expected output: 7.5 or higher
```

---

## Quick Start

### Option 1: CMake (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/fhe-cuda.git
cd fhe-cuda

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build (use all available cores)
make -j$(nproc)

# Run tests
./test_fhe
```

### Option 2: GNU Make

```bash
# Clone repository
git clone https://github.com/yourusername/fhe-cuda.git
cd fhe-cuda

# Build
make -j$(nproc)

# Run tests
./build/test_fhe
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

#### Install Dependencies

```bash
# Update package list
sudo apt update

# Install build tools
sudo apt install -y build-essential cmake git

# Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

### Linux (CentOS/RHEL)

#### Install Dependencies

```bash
# Install development tools
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3 git

# Install CUDA Toolkit
sudo yum-config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo yum install -y cuda-toolkit-12-3

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Build

```bash
mkdir build && cd build
cmake3 ..
make -j$(nproc)
```

---

### Windows

#### Prerequisites

1. **Visual Studio 2019 or later** with C++ support
2. **CUDA Toolkit 11.0+**
3. **CMake 3.18+**

#### Build with Visual Studio

```powershell
# Open Visual Studio Developer Command Prompt

# Clone repository
git clone https://github.com/yourusername/fhe-cuda.git
cd fhe-cuda

# Create build directory
mkdir build
cd build

# Configure (for Visual Studio 2019)
cmake .. -G "Visual Studio 16 2019" -A x64

# Build
cmake --build . --config Release

# Run tests
.\Release\test_fhe.exe
```

#### Build with MSVC Command Line

```powershell
# Open Visual Studio Developer Command Prompt

mkdir build
cd build
cmake .. -G "NMake Makefiles"
nmake

# Run tests
.\test_fhe.exe
```

---

### Windows (WSL2)

If you have WSL2 with CUDA support:

```bash
# Inside WSL2
sudo apt update
sudo apt install -y build-essential cmake git

# CUDA should already be available via WSL2
nvcc --version

# Build as on Linux
mkdir build && cd build
cmake ..
make -j$(nproc)
```

---

## Build Configurations

### Release Build (Default)

Optimized for performance:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Flags:**
- `-O3`: Maximum optimization
- `-use_fast_math`: Fast math operations
- No debug symbols

**Use for:** Production, benchmarks

---

### Debug Build

With debug symbols and assertions:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

**Flags:**
- `-g -G`: Debug symbols for host and device
- `-O0`: No optimization
- Assertions enabled

**Use for:** Development, debugging

With Makefile:
```bash
make debug
```

---

### Profile Build

Optimized with profiling info:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

**Flags:**
- `-O3`: Full optimization
- `-lineinfo`: Line info for profiling

**Use for:** Performance profiling with nsys/nvprof

With Makefile:
```bash
make profile
```

---

### Custom Compute Capability

Specify target GPU architecture:

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

**Common Values:**
- `75`: Turing (RTX 20-series)
- `80`: Ampere (A100)
- `86`: Ampere (RTX 30-series)
- `89`: Ada Lovelace (RTX 40-series)
- `90`: Hopper (H100)

---

### Verbose Build

See full compilation commands:

```bash
# CMake
make VERBOSE=1

# Or
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make
```

---

## Troubleshooting

### CUDA Not Found

**Error:**
```
CMake Error: Could not find CUDA
```

**Solution:**
```bash
# Set CUDA path manually
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

---

### Wrong Compute Capability

**Error:**
```
nvcc fatal: Unsupported gpu architecture 'compute_90'
```

**Solution:**
Update CUDA Toolkit or remove unsupported architectures:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86"
```

---

### Out of Memory During Build

**Error:**
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution:**
Reduce parallel jobs:
```bash
make -j2  # Instead of -j$(nproc)
```

---

### PTX Assembly Errors

**Error:**
```
error: expected primary-expression before 'asm'
```

**Solution:**
Ensure `--expt-relaxed-constexpr` flag is set (should be automatic).

---

### Missing cuFFT

**Error:**
```
cannot find -lcufft
```

**Solution:**
```bash
# Ensure CUDA libraries are in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or install cuFFT separately
sudo apt install libcufft-dev
```

---

### Windows: MSVC Version Mismatch

**Error:**
```
CUDA version X does not support MSVC version Y
```

**Solution:**
Install compatible Visual Studio version or downgrade CUDA. See [NVIDIA compatibility matrix](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).

---

## Advanced Configuration

### Custom Install Prefix

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/fhe-cuda
make install
```

### Static vs Shared Library

Currently builds static library (`.a`). To build shared:

Edit `CMakeLists.txt`:
```cmake
add_library(fhe_cuda SHARED ${FHE_SOURCES})  # Instead of STATIC
```

---

### Disable Specific Optimizations

```bash
# Disable fast math
cmake .. -DCMAKE_CUDA_FLAGS="-O3"

# Disable specific architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80"  # Only SM 8.0
```

---

### Cross-Compilation

Not officially supported due to CUDA requirements, but for ARM64 with CUDA:

```bash
cmake .. -DCMAKE_SYSTEM_PROCESSOR=aarch64
```

---

### Build Only Library (No Tests)

```bash
make lib  # With Makefile

# Or with CMake
cmake .. -DBUILD_TESTING=OFF
```

---

### Integration with Existing Project

#### CMake

```cmake
# In your CMakeLists.txt
find_package(CUDA REQUIRED)

add_subdirectory(fhe-cuda)

target_link_libraries(your_target fhe_cuda)
```

#### Manual Linking

```bash
# Compile your code
nvcc -c your_code.cu -I/path/to/fhe-cuda/include

# Link
nvcc your_code.o -L/path/to/fhe-cuda/build -lfhe_cuda -lcufft -o your_program
```

---

## Verification

### Test Build

```bash
./test_fhe

# Expected output:
# === GPU-Accelerated FHE Library Tests ===
# Using GPU: [Your GPU Name]
# ...
# === All Tests Passed ===
```

### Run Benchmarks

```bash
./test_fhe --benchmark

# Or with script
./scripts/benchmark.sh
```

---

## Performance Tuning

### Compiler Flags

For maximum performance:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_FLAGS="-O3 -use_fast_math -maxrregcount=64"
```

### GPU Clocking

Ensure GPU is at max performance:

```bash
# Set persistence mode
sudo nvidia-smi -pm 1

# Set max clock speeds
sudo nvidia-smi -lgc 2100  # Adjust for your GPU
```

---

## Build Output Structure

After successful build:

```
build/
├── libfhe_cuda.a           # Static library
├── test_fhe                # Test executable
├── CMakeFiles/             # CMake metadata
└── *.o                     # Object files
```

Install locations (after `make install`):

```
/usr/local/lib/libfhe_cuda.a
/usr/local/include/fhe/
    ├── bigint.cuh
    ├── fhe.cuh
    ├── ntt.cuh
    ├── polynomial.cuh
    └── rns.cuh
```

---

## Clean Build

```bash
# CMake
cd build
make clean

# Or remove build directory
cd ..
rm -rf build

# Makefile
make clean
```

---

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.3.0-devel-ubuntu22.04
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Install Dependencies
      run: |
        apt update
        apt install -y cmake build-essential
    
    - name: Build
      run: |
        mkdir build && cd build
        cmake ..
        make -j$(nproc)
    
    - name: Test
      run: |
        cd build
        ./test_fhe
```

---

## Next Steps

After successful build:

1. **Run Tests**: `./test_fhe`
2. **Try Examples**: See `examples/` directory
3. **Read API**: Check `docs/API_REFERENCE.md`
4. **Profile Performance**: Use `scripts/profile.sh`

---

## Getting Help

- **Build Issues**: Check [Troubleshooting](#troubleshooting)
- **CUDA Errors**: See [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **CMake Problems**: See [CMake Documentation](https://cmake.org/documentation/)
- **Report Bugs**: Open an issue on GitHub

---

**Happy Building!** 🚀
