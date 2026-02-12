#include "ntt.cuh"
#include <cmath>
#include <iostream>

namespace fhe {

NTTEngine::NTTEngine(uint32_t polynomial_degree, const uint256_t& modulus)
    : n_(polynomial_degree), modulus_(modulus) {
    
    // Compute Montgomery parameters
    mont_params_ = compute_montgomery_params(modulus);
    
    // Allocate device memory for twiddle factors
    cudaMalloc(&d_twiddle_factors_, n_ * sizeof(uint256_t));
    cudaMalloc(&d_inv_twiddle_factors_, n_ * sizeof(uint256_t));
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Precompute twiddle factors
    precompute_twiddle_factors();
}

NTTEngine::~NTTEngine() {
    cudaFree(d_twiddle_factors_);
    cudaFree(d_inv_twiddle_factors_);
    cudaStreamDestroy(stream_);
}

void NTTEngine::forward(uint256_t* d_data) {
    // Bit-reverse permutation
    uint32_t num_blocks = (n_ + 255) / 256;
    bit_reverse_kernel<<<num_blocks, 256, 0, stream_>>>(d_data, n_);
    
    // Launch optimized NTT kernel
    size_t shared_mem = n_ * sizeof(uint256_t);
    ntt_forward_optimized_kernel<<<1, n_, shared_mem, stream_>>>(
        d_data, d_twiddle_factors_, modulus_, mont_params_.inv, n_
    );
}

void NTTEngine::inverse(uint256_t* d_data) {
    size_t shared_mem = n_ * sizeof(uint256_t);
    ntt_inverse_optimized_kernel<<<1, n_, shared_mem, stream_>>>(
        d_data, d_inv_twiddle_factors_, modulus_, mont_params_.inv, n_inv_, n_
    );
}

void NTTEngine::multiply(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b) {
    // Copy inputs to temporary buffers
    uint256_t *d_temp_a, *d_temp_b;
    cudaMalloc(&d_temp_a, n_ * sizeof(uint256_t));
    cudaMalloc(&d_temp_b, n_ * sizeof(uint256_t));
    
    cudaMemcpyAsync(d_temp_a, d_a, n_ * sizeof(uint256_t), 
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(d_temp_b, d_b, n_ * sizeof(uint256_t), 
                    cudaMemcpyDeviceToDevice, stream_);
    
    // Forward NTT on both inputs
    forward(d_temp_a);
    forward(d_temp_b);
    
    // Pointwise multiplication
    uint32_t num_blocks = (n_ + 255) / 256;
    ntt_pointwise_mul_kernel<<<num_blocks, 256, 0, stream_>>>(
        d_result, d_temp_a, d_temp_b, modulus_, mont_params_.inv, n_
    );
    
    // Inverse NTT
    inverse(d_result);
    
    cudaFree(d_temp_a);
    cudaFree(d_temp_b);
}

void NTTEngine::precompute_twiddle_factors() {
    // Compute n-th root of unity
    uint256_t root = find_primitive_root(n_, modulus_);
    
    // Allocate host memory
    std::vector<uint256_t> h_twiddle(n_);
    std::vector<uint256_t> h_inv_twiddle(n_);
    
    // Compute powers of root
    h_twiddle[0] = uint256_t(1);
    for (uint32_t i = 1; i < n_; i++) {
        // This is a placeholder - needs proper implementation
        h_twiddle[i] = uint256_t(i);
    }
    
    // Compute inverse root
    uint256_t inv_root = mod_inverse(root, modulus_);
    h_inv_twiddle[0] = uint256_t(1);
    for (uint32_t i = 1; i < n_; i++) {
        h_inv_twiddle[i] = uint256_t(i);
    }
    
    // Compute n_inv = 1/n mod q
    n_inv_ = mod_inverse(uint256_t(n_), modulus_);
    
    // Copy to device
    cudaMemcpy(d_twiddle_factors_, h_twiddle.data(), n_ * sizeof(uint256_t), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_twiddle_factors_, h_inv_twiddle.data(), n_ * sizeof(uint256_t),
               cudaMemcpyHostToDevice);
}

// Utility functions
__host__ uint256_t find_primitive_root(uint32_t n, const uint256_t& modulus) {
    // Placeholder implementation
    // Real implementation needs to find a primitive n-th root of unity
    return uint256_t(3);
}

__host__ uint256_t mod_inverse(const uint256_t& a, const uint256_t& modulus) {
    // Placeholder for extended Euclidean algorithm
    return uint256_t(1);
}

// RNS-NTT Engine implementation
RNS_NTTEngine::RNS_NTTEngine(uint32_t polynomial_degree, 
                             const uint256_t* rns_moduli, 
                             uint32_t num_primes)
    : n_(polynomial_degree), num_primes_(num_primes) {
    
    // Allocate RNS moduli
    rns_moduli_ = new uint256_t[num_primes];
    memcpy(rns_moduli_, rns_moduli, num_primes * sizeof(uint256_t));
    
    // Create NTT engine for each RNS prime
    ntt_engines_ = new NTTEngine*[num_primes];
    for (uint32_t i = 0; i < num_primes; i++) {
        ntt_engines_[i] = new NTTEngine(n_, rns_moduli[i]);
    }
    
    // Create CUDA streams
    streams_ = new cudaStream_t[num_primes];
    for (uint32_t i = 0; i < num_primes; i++) {
        cudaStreamCreate(&streams_[i]);
    }
    
    // Allocate temporary storage
    cudaMalloc(&d_rns_temp_, n_ * num_primes * sizeof(uint256_t));
}

RNS_NTTEngine::~RNS_NTTEngine() {
    for (uint32_t i = 0; i < num_primes_; i++) {
        delete ntt_engines_[i];
        cudaStreamDestroy(streams_[i]);
    }
    delete[] ntt_engines_;
    delete[] streams_;
    delete[] rns_moduli_;
    cudaFree(d_rns_temp_);
}

void RNS_NTTEngine::forward_rns(uint256_t* d_rns_data) {
    // Launch NTT on each RNS component in parallel
    for (uint32_t i = 0; i < num_primes_; i++) {
        uint256_t* component = d_rns_data + i * n_;
        ntt_engines_[i]->forward(component);
    }
}

void RNS_NTTEngine::inverse_rns(uint256_t* d_rns_data) {
    for (uint32_t i = 0; i < num_primes_; i++) {
        uint256_t* component = d_rns_data + i * n_;
        ntt_engines_[i]->inverse(component);
    }
}

} // namespace fhe
