#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "bigint.cuh"
#include "ntt.cuh"

namespace fhe {

// Polynomial representation in coefficient form
struct Polynomial {
    uint256_t* coeffs;     // Coefficient array (device memory)
    uint32_t degree;       // Polynomial degree
    uint256_t modulus;     // Coefficient modulus
    bool is_ntt_form;      // Whether in NTT domain
    
    __host__ Polynomial(uint32_t deg, const uint256_t& mod);
    __host__ ~Polynomial();
};

// Polynomial arithmetic operations
class PolynomialOps {
public:
    PolynomialOps(uint32_t max_degree, const uint256_t& modulus, NTTEngine* ntt);
    ~PolynomialOps();
    
    // Basic operations
    void add(Polynomial& result, const Polynomial& a, const Polynomial& b);
    void sub(Polynomial& result, const Polynomial& a, const Polynomial& b);
    void mul(Polynomial& result, const Polynomial& a, const Polynomial& b);
    
    // Scalar operations
    void mul_scalar(Polynomial& result, const Polynomial& a, const uint256_t& scalar);
    void add_scalar(Polynomial& result, const Polynomial& a, const uint256_t& scalar);
    
    // NTT-based fast multiplication
    void mul_ntt(Polynomial& result, const Polynomial& a, const Polynomial& b);
    
    // Negacyclic convolution for Ring-LWE (x^n + 1)
    void mul_negacyclic(Polynomial& result, const Polynomial& a, const Polynomial& b);
    
    // Modulus switching
    void mod_switch(Polynomial& result, const Polynomial& a, const uint256_t& new_modulus);
    
    // Noise estimation
    double estimate_noise(const Polynomial& poly);
    
private:
    uint32_t max_degree_;
    uint256_t modulus_;
    MontgomeryParams mont_params_;
    NTTEngine* ntt_engine_;
    
    // Temporary storage
    uint256_t* d_temp_a_;
    uint256_t* d_temp_b_;
    uint256_t* d_temp_result_;
    
    cudaStream_t stream_;
};

// Kernels for polynomial operations
__global__ void poly_add_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    uint32_t n
);

__global__ void poly_sub_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    uint32_t n
);

__global__ void poly_mul_scalar_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t scalar,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
);

__global__ void poly_add_scalar_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t scalar,
    const uint256_t modulus,
    uint32_t n
);

// Modulus switching with rounding
__global__ void poly_mod_switch_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t old_modulus,
    const uint256_t new_modulus,
    uint32_t n
);

// Negacyclic reduction: reduce poly mod (x^n + 1)
__global__ void negacyclic_reduce_kernel(
    uint256_t* data,
    const uint256_t modulus,
    uint32_t n
);

// Sample from discrete Gaussian distribution for noise
__global__ void sample_gaussian_kernel(
    uint256_t* result,
    const uint256_t modulus,
    float sigma,
    uint64_t seed,
    uint32_t n
);

// Sample uniform random polynomial
__global__ void sample_uniform_kernel(
    uint256_t* result,
    const uint256_t modulus,
    uint64_t seed,
    uint32_t n
);

// Sample ternary polynomial {-1, 0, 1}
__global__ void sample_ternary_kernel(
    uint256_t* result,
    const uint256_t modulus,
    float probability,
    uint64_t seed,
    uint32_t n
);

// Noise measurement
__global__ void compute_noise_norm_kernel(
    double* result,
    const uint256_t* poly,
    const uint256_t modulus,
    uint32_t n
);

} // namespace fhe
