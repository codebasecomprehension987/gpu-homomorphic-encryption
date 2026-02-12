#include "ntt.cuh"
#include <cmath>

namespace fhe {

// Shared memory optimized Cooley-Tukey NTT kernel
__global__ void ntt_forward_optimized_kernel(
    uint256_t* data,
    const uint256_t* twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
) {
    extern __shared__ uint256_t shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    
    // Load data into shared memory with coalesced access
    uint32_t global_idx = bid * block_size + tid;
    if (global_idx < n) {
        shared_data[tid] = data[global_idx];
    }
    __syncthreads();
    
    // Perform in-place NTT on shared memory block
    uint32_t log_n = __popc(n - 1) + 1; // log2(n)
    
    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t m = 1 << stage;
        uint32_t m2 = m << 1;
        
        // Each thread handles one butterfly
        uint32_t k = tid / m;
        uint32_t j = tid % m;
        
        if (k * m2 + j + m < block_size) {
            uint32_t idx1 = k * m2 + j;
            uint32_t idx2 = idx1 + m;
            
            // Compute twiddle factor index
            uint32_t twiddle_idx = (j << (log_n - stage - 1));
            
            // Butterfly operation
            uint256_t u = shared_data[idx1];
            uint256_t v = mul_mod_montgomery(shared_data[idx2], 
                                            twiddle_factors[twiddle_idx],
                                            modulus, mont_inv);
            
            shared_data[idx1] = add_mod(u, v, modulus);
            shared_data[idx2] = sub_mod(u, v, modulus);
        }
        
        __syncthreads();
    }
    
    // Write back to global memory
    if (global_idx < n) {
        data[global_idx] = shared_data[tid];
    }
}

// Optimized inverse NTT with normalization
__global__ void ntt_inverse_optimized_kernel(
    uint256_t* data,
    const uint256_t* inv_twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    const uint256_t n_inv,
    uint32_t n
) {
    extern __shared__ uint256_t shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t block_size = blockDim.x;
    uint32_t global_idx = bid * block_size + tid;
    
    // Load into shared memory
    if (global_idx < n) {
        shared_data[tid] = data[global_idx];
    }
    __syncthreads();
    
    uint32_t log_n = __popc(n - 1) + 1;
    
    // Gentleman-Sande butterflies
    for (int stage = log_n - 1; stage >= 0; stage--) {
        uint32_t m = 1 << stage;
        uint32_t m2 = m << 1;
        
        uint32_t k = tid / m;
        uint32_t j = tid % m;
        
        if (k * m2 + j + m < block_size) {
            uint32_t idx1 = k * m2 + j;
            uint32_t idx2 = idx1 + m;
            
            uint32_t twiddle_idx = (j << (log_n - stage - 1));
            
            // GS Butterfly
            uint256_t u = shared_data[idx1];
            uint256_t v = shared_data[idx2];
            
            shared_data[idx1] = add_mod(u, v, modulus);
            
            uint256_t diff = sub_mod(u, v, modulus);
            shared_data[idx2] = mul_mod_montgomery(diff, 
                                                   inv_twiddle_factors[twiddle_idx],
                                                   modulus, mont_inv);
        }
        
        __syncthreads();
    }
    
    // Normalize by n^(-1)
    if (global_idx < n) {
        data[global_idx] = mul_mod_montgomery(shared_data[tid], n_inv, modulus, mont_inv);
    }
}

// Pointwise multiplication in NTT domain (element-wise)
__global__ void ntt_pointwise_mul_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = mul_mod_montgomery(a[idx], b[idx], modulus, mont_inv);
    }
}

// Bit-reverse permutation for decimation-in-frequency
__global__ void bit_reverse_kernel(uint256_t* data, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        uint32_t log_n = __popc(n - 1) + 1;
        
        // Compute bit-reversed index
        uint32_t rev_idx = 0;
        for (uint32_t i = 0; i < log_n; i++) {
            if (idx & (1 << i)) {
                rev_idx |= (1 << (log_n - 1 - i));
            }
        }
        
        // Swap only if idx < rev_idx to avoid double-swapping
        if (idx < rev_idx) {
            uint256_t temp = data[idx];
            data[idx] = data[rev_idx];
            data[rev_idx] = temp;
        }
    }
}

// Multi-stream NTT for batch processing
__global__ void ntt_forward_batch_kernel(
    uint256_t* data,
    const uint256_t* twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n,
    uint32_t batch_offset
) {
    extern __shared__ uint256_t shared_data[];
    
    uint32_t tid = threadIdx.x;
    uint32_t batch_idx = blockIdx.x / (n / blockDim.x);
    uint32_t local_block = blockIdx.x % (n / blockDim.x);
    
    uint32_t global_idx = batch_offset + batch_idx * n + local_block * blockDim.x + tid;
    
    if (tid < n) {
        shared_data[tid] = data[global_idx];
    }
    __syncthreads();
    
    // Same NTT computation as before
    uint32_t log_n = __popc(n - 1) + 1;
    
    for (uint32_t stage = 0; stage < log_n; stage++) {
        uint32_t m = 1 << stage;
        uint32_t m2 = m << 1;
        
        uint32_t k = tid / m;
        uint32_t j = tid % m;
        
        if (k * m2 + j + m < blockDim.x) {
            uint32_t idx1 = k * m2 + j;
            uint32_t idx2 = idx1 + m;
            uint32_t twiddle_idx = (j << (log_n - stage - 1));
            
            ct_butterfly(shared_data[idx1], shared_data[idx2],
                        twiddle_factors[twiddle_idx], modulus, mont_inv);
        }
        
        __syncthreads();
    }
    
    if (tid < n) {
        data[global_idx] = shared_data[tid];
    }
}

// Stockham auto-sort NTT (no bit-reversal needed)
__global__ void ntt_stockham_kernel(
    uint256_t* output,
    const uint256_t* input,
    const uint256_t* twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n,
    uint32_t stage
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        uint32_t m = 1 << stage;
        uint32_t m2 = m << 1;
        
        uint32_t k = idx / m;
        uint32_t j = idx % m;
        
        uint32_t idx1 = k * m2 + j;
        uint32_t idx2 = idx1 + m;
        
        uint32_t twiddle_idx = j * (n / m2);
        
        uint256_t u = input[idx1];
        uint256_t v = mul_mod_montgomery(input[idx2], twiddle_factors[twiddle_idx],
                                        modulus, mont_inv);
        
        output[idx1] = add_mod(u, v, modulus);
        output[idx2] = sub_mod(u, v, modulus);
    }
}

} // namespace fhe
