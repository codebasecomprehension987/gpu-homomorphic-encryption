#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "bigint.cuh"

namespace fhe {

// RNS base configuration
struct RNSBase {
    uint32_t num_primes;           // Number of RNS primes
    uint256_t* primes;             // RNS prime moduli (coprime)
    uint256_t product;             // Product of all primes
    MontgomeryParams* mont_params; // Montgomery params for each prime
    
    // CRT reconstruction parameters
    uint256_t* crt_multipliers;    // M_i = M / q_i
    uint256_t* crt_inverses;       // M_i^(-1) mod q_i
};

// RNS representation of a polynomial coefficient
struct RNSValue {
    uint256_t* residues;  // One residue per prime
    uint32_t num_primes;
};

class RNSContext {
public:
    RNSContext(const std::vector<uint256_t>& primes);
    ~RNSContext();
    
    // Convert to RNS representation
    void to_rns(uint256_t* d_rns_residues, const uint256_t* d_values, uint32_t count);
    
    // Convert from RNS using Chinese Remainder Theorem
    void from_rns(uint256_t* d_values, const uint256_t* d_rns_residues, uint32_t count);
    
    // RNS arithmetic operations
    void add_rns(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b, uint32_t count);
    void sub_rns(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b, uint32_t count);
    void mul_rns(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b, uint32_t count);
    
    // Modulus switching in RNS
    void mod_switch_rns(uint256_t* d_result, const uint256_t* d_input, 
                        uint32_t old_level, uint32_t new_level, uint32_t count);
    
    // Base extension for bootstrapping
    void base_extend(uint256_t* d_extended, const uint256_t* d_input,
                    const RNSBase& target_base, uint32_t count);
    
    uint32_t num_primes() const { return base_.num_primes; }
    const RNSBase& base() const { return base_; }
    
private:
    RNSBase base_;
    
    // Device memory for intermediate computations
    uint256_t* d_temp_rns_;
    uint256_t* d_crt_workspace_;
    
    cudaStream_t stream_;
    
    void precompute_crt_parameters();
    void allocate_device_memory();
};

// Kernels for RNS operations
__global__ void to_rns_kernel(
    uint256_t* rns_residues,  // Output: [num_primes * count]
    const uint256_t* values,   // Input: [count]
    const uint256_t* primes,   // RNS moduli
    uint32_t num_primes,
    uint32_t count
);

__global__ void from_rns_crt_kernel(
    uint256_t* values,         // Output: [count]
    const uint256_t* residues, // Input: [num_primes * count]
    const uint256_t* crt_multipliers,
    const uint256_t* crt_inverses,
    const uint256_t* primes,
    const uint256_t product,
    uint32_t num_primes,
    uint32_t count
);

__global__ void rns_add_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t* primes,
    uint32_t num_primes,
    uint32_t count
);

__global__ void rns_sub_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t* primes,
    uint32_t num_primes,
    uint32_t count
);

__global__ void rns_mul_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t* primes,
    const MontgomeryParams* mont_params,
    uint32_t num_primes,
    uint32_t count
);

// Fast base conversion (Bajard et al.)
__global__ void fast_base_conversion_kernel(
    uint256_t* output_residues,
    const uint256_t* input_residues,
    const uint256_t* source_primes,
    const uint256_t* target_primes,
    const uint256_t* conversion_matrix, // Precomputed constants
    uint32_t source_size,
    uint32_t target_size,
    uint32_t count
);

// Approximate modulus switching with RNS
__global__ void rns_mod_switch_kernel(
    uint256_t* result,
    const uint256_t* input,
    const uint256_t* old_primes,
    const uint256_t* new_primes,
    uint32_t old_level,
    uint32_t new_level,
    uint32_t count
);

// Utility: Generate RNS-friendly primes
__host__ std::vector<uint256_t> generate_rns_primes(
    uint32_t bit_length,
    uint32_t num_primes,
    uint32_t ntt_size // Ensure primes work for NTT
);

// Utility: Check if number is prime (Miller-Rabin)
__host__ bool is_prime(const uint256_t& n, uint32_t iterations = 20);

// Utility: Find NTT-friendly prime (q ≡ 1 mod 2n)
__host__ uint256_t find_ntt_prime(uint32_t bit_length, uint32_t ntt_size);

} // namespace fhe
