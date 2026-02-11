#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "bigint.cuh"

namespace fhe {

// NTT parameters for Ring-LWE
struct NTTParams {
    uint32_t n;                    // Polynomial degree (power of 2)
    uint256_t modulus;             // Prime modulus (q)
    uint256_t root_of_unity;       // n-th root of unity mod q
    uint256_t* twiddle_factors;    // Precomputed powers of root
    uint256_t* inv_twiddle_factors; // Inverse twiddle factors
    MontgomeryParams mont;         // Montgomery parameters
    
    // For multiple RNS bases
    uint32_t num_primes;
    uint256_t* rns_moduli;
    MontgomeryParams* rns_params;
};

// Forward NTT (Cooley-Tukey butterfly)
__global__ void ntt_forward_kernel(
    uint256_t* data,
    const uint256_t* twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n,
    uint32_t stage
);

// Inverse NTT (Gentleman-Sande butterfly)
__global__ void ntt_inverse_kernel(
    uint256_t* data,
    const uint256_t* inv_twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n,
    uint32_t stage
);

// Optimized NTT using shared memory and bank conflict avoidance
__global__ void ntt_forward_optimized_kernel(
    uint256_t* data,
    const uint256_t* twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
);

__global__ void ntt_inverse_optimized_kernel(
    uint256_t* data,
    const uint256_t* inv_twiddle_factors,
    const uint256_t modulus,
    const uint256_t mont_inv,
    const uint256_t n_inv, // Inverse of n mod q
    uint32_t n
);

// Pointwise multiplication in NTT domain (Hadamard product)
__global__ void ntt_pointwise_mul_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
);

// Class for managing NTT operations
class NTTEngine {
public:
    NTTEngine(uint32_t polynomial_degree, const uint256_t& modulus);
    ~NTTEngine();
    
    // Forward transform
    void forward(uint256_t* d_data);
    
    // Inverse transform
    void inverse(uint256_t* d_data);
    
    // Polynomial multiplication via NTT
    void multiply(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b);
    
    // Batch operations for multiple polynomials
    void forward_batch(uint256_t* d_data, uint32_t batch_size);
    void inverse_batch(uint256_t* d_data, uint32_t batch_size);
    
private:
    uint32_t n_;
    uint256_t modulus_;
    MontgomeryParams mont_params_;
    
    uint256_t* d_twiddle_factors_;
    uint256_t* d_inv_twiddle_factors_;
    uint256_t n_inv_; // 1/n mod q
    
    cudaStream_t stream_;
    
    void precompute_twiddle_factors();
    void compute_root_of_unity();
};

// RNS-based NTT for handling larger moduli
class RNS_NTTEngine {
public:
    RNS_NTTEngine(uint32_t polynomial_degree, 
                  const uint256_t* rns_moduli, 
                  uint32_t num_primes);
    ~RNS_NTTEngine();
    
    // Convert to RNS representation
    void to_rns(uint256_t* d_rns_data, const uint256_t* d_data);
    
    // Convert from RNS to standard representation
    void from_rns(uint256_t* d_data, const uint256_t* d_rns_data);
    
    // NTT on all RNS components
    void forward_rns(uint256_t* d_rns_data);
    void inverse_rns(uint256_t* d_rns_data);
    
    // Multiply in RNS-NTT domain
    void multiply_rns(uint256_t* d_result, 
                     const uint256_t* d_a, 
                     const uint256_t* d_b);
    
private:
    uint32_t n_;
    uint32_t num_primes_;
    uint256_t* rns_moduli_;
    
    NTTEngine** ntt_engines_; // One per RNS prime
    
    uint256_t* d_rns_temp_;
    cudaStream_t* streams_;
};

// Utility functions
__host__ uint256_t find_primitive_root(uint32_t n, const uint256_t& modulus);
__host__ uint256_t mod_inverse(const uint256_t& a, const uint256_t& modulus);

// Bit-reverse permutation for NTT
__global__ void bit_reverse_kernel(uint256_t* data, uint32_t n);

// Cooley-Tukey butterfly operation
__device__ __forceinline__
void ct_butterfly(uint256_t& a, uint256_t& b, 
                  const uint256_t& twiddle,
                  const uint256_t& modulus,
                  const uint256_t& mont_inv) {
    uint256_t temp = mul_mod_montgomery(b, twiddle, modulus, mont_inv);
    b = sub_mod(a, temp, modulus);
    a = add_mod(a, temp, modulus);
}

// Gentleman-Sande butterfly operation
__device__ __forceinline__
void gs_butterfly(uint256_t& a, uint256_t& b,
                  const uint256_t& twiddle,
                  const uint256_t& modulus,
                  const uint256_t& mont_inv) {
    uint256_t temp = add_mod(a, b, modulus);
    b = sub_mod(a, b, modulus);
    b = mul_mod_montgomery(b, twiddle, modulus, mont_inv);
    a = temp;
}

} // namespace fhe
