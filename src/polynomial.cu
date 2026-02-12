#include "polynomial.cuh"
#include <cstring>

namespace fhe {

Polynomial::Polynomial(uint32_t deg, const uint256_t& mod) 
    : degree(deg), modulus(mod), is_ntt_form(false) {
    cudaMalloc(&coeffs, (degree + 1) * sizeof(uint256_t));
    cudaMemset(coeffs, 0, (degree + 1) * sizeof(uint256_t));
}

Polynomial::~Polynomial() {
    cudaFree(coeffs);
}

PolynomialOps::PolynomialOps(uint32_t max_degree, const uint256_t& modulus, NTTEngine* ntt)
    : max_degree_(max_degree), modulus_(modulus), ntt_engine_(ntt) {
    
    mont_params_ = compute_montgomery_params(modulus);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_a_, (max_degree + 1) * sizeof(uint256_t));
    cudaMalloc(&d_temp_b_, (max_degree + 1) * sizeof(uint256_t));
    cudaMalloc(&d_temp_result_, (max_degree + 1) * sizeof(uint256_t));
    
    cudaStreamCreate(&stream_);
}

PolynomialOps::~PolynomialOps() {
    cudaFree(d_temp_a_);
    cudaFree(d_temp_b_);
    cudaFree(d_temp_result_);
    cudaStreamDestroy(stream_);
}

void PolynomialOps::add(Polynomial& result, const Polynomial& a, const Polynomial& b) {
    uint32_t n = std::min(a.degree, b.degree) + 1;
    uint32_t num_blocks = (n + 255) / 256;
    
    poly_add_kernel<<<num_blocks, 256, 0, stream_>>>(
        result.coeffs, a.coeffs, b.coeffs, modulus_, n
    );
}

void PolynomialOps::sub(Polynomial& result, const Polynomial& a, const Polynomial& b) {
    uint32_t n = std::min(a.degree, b.degree) + 1;
    uint32_t num_blocks = (n + 255) / 256;
    
    poly_sub_kernel<<<num_blocks, 256, 0, stream_>>>(
        result.coeffs, a.coeffs, b.coeffs, modulus_, n
    );
}

void PolynomialOps::mul_ntt(Polynomial& result, const Polynomial& a, const Polynomial& b) {
    if (ntt_engine_) {
        ntt_engine_->multiply(result.coeffs, a.coeffs, b.coeffs);
    }
}

void PolynomialOps::mul_scalar(Polynomial& result, const Polynomial& a, const uint256_t& scalar) {
    uint32_t n = a.degree + 1;
    uint32_t num_blocks = (n + 255) / 256;
    
    poly_mul_scalar_kernel<<<num_blocks, 256, 0, stream_>>>(
        result.coeffs, a.coeffs, scalar, modulus_, mont_params_.inv, n
    );
}

// Kernels
__global__ void poly_add_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = add_mod(a[idx], b[idx], modulus);
    }
}

__global__ void poly_sub_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t* b,
    const uint256_t modulus,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = sub_mod(a[idx], b[idx], modulus);
    }
}

__global__ void poly_mul_scalar_kernel(
    uint256_t* result,
    const uint256_t* a,
    const uint256_t scalar,
    const uint256_t modulus,
    const uint256_t mont_inv,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = mul_mod_montgomery(a[idx], scalar, modulus, mont_inv);
    }
}

__global__ void sample_gaussian_kernel(
    uint256_t* result,
    const uint256_t modulus,
    float sigma,
    uint64_t seed,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Placeholder: Simple random for now
        // Real implementation needs Box-Muller transform or ziggurat
        uint64_t val = (seed + idx) % modulus.limbs[0];
        result[idx] = uint256_t(val);
    }
}

__global__ void sample_uniform_kernel(
    uint256_t* result,
    const uint256_t modulus,
    uint64_t seed,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple LCG for demonstration
        uint64_t val = ((seed + idx) * 1103515245 + 12345) % modulus.limbs[0];
        result[idx] = uint256_t(val);
    }
}

} // namespace fhe
