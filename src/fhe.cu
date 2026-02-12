#include "fhe.cuh"
#include <random>
#include <iostream>

namespace fhe {

FHEContext::FHEContext(const SecurityParams& sec_params) {
    params_.security = sec_params;
    params_.n = sec_params.poly_degree;
    
    // Set moduli based on security level
    // Placeholder: real implementation needs proper parameter selection
    params_.q = uint256_t(1ULL << 60);
    params_.t = uint256_t(65537); // Plaintext modulus
    
    // Compute delta = floor(q/t)
    params_.delta = uint256_t(1ULL << 44);
    
    // Initialize RNS context with multiple primes
    std::vector<uint256_t> rns_primes;
    for (uint32_t i = 0; i < 3; i++) {
        rns_primes.push_back(find_ntt_prime(40, params_.n));
    }
    params_.rns_ctx = new RNSContext(rns_primes);
    
    // Initialize NTT engine
    params_.ntt_engine = new NTTEngine(params_.n, params_.q);
    
    // Initialize polynomial operations
    params_.poly_ops = new PolynomialOps(params_.n, params_.q, params_.ntt_engine);
    
    // Create CUDA streams for pipelining
    streams_.resize(4);
    for (auto& stream : streams_) {
        cudaStreamCreate(&stream);
    }
    
    // Initialize RNG
    init_rng(12345);
}

FHEContext::~FHEContext() {
    delete params_.rns_ctx;
    delete params_.ntt_engine;
    delete params_.poly_ops;
    
    for (auto& stream : streams_) {
        cudaStreamDestroy(stream);
    }
    
    cudaFree(d_rng_states_);
}

void FHEContext::keygen(PublicKey& pk, SecretKey& sk) {
    // Generate secret key: ternary polynomial
    sk.sk = new Polynomial(params_.n, params_.q);
    sample_ternary_polynomial(*sk.sk);
    
    // Generate public key
    pk.pk0 = new Polynomial(params_.n, params_.q);
    pk.pk1 = new Polynomial(params_.n, params_.q);
    
    // pk1 = uniform random
    sample_uniform_polynomial(*pk.pk1);
    
    // Generate error
    Polynomial e(params_.n, params_.q);
    sample_error_polynomial(e);
    
    // pk0 = -(pk1 * sk) + e
    Polynomial temp(params_.n, params_.q);
    params_.poly_ops->mul_ntt(temp, *pk.pk1, *sk.sk);
    params_.poly_ops->sub(*pk.pk0, e, temp);
}

void FHEContext::relinkey_gen(RelinKeys& rlk, const SecretKey& sk, uint32_t decomp_bits) {
    rlk.decomp_bits = decomp_bits;
    
    // Compute s^2
    Polynomial s_squared(params_.n, params_.q);
    params_.poly_ops->mul_ntt(s_squared, *sk.sk, *sk.sk);
    
    // Generate relinearization keys for each decomposition level
    uint32_t num_levels = (256 + decomp_bits - 1) / decomp_bits;
    
    for (uint32_t i = 0; i < num_levels; i++) {
        PublicKey* rlk_key = new PublicKey();
        
        // Generate uniform random a
        rlk_key->pk1 = new Polynomial(params_.n, params_.q);
        sample_uniform_polynomial(*rlk_key->pk1);
        
        // Generate error
        Polynomial e(params_.n, params_.q);
        sample_error_polynomial(e);
        
        // Compute 2^(i*decomp_bits) * s^2
        uint256_t scalar = uint256_t(1ULL << (i * decomp_bits));
        Polynomial scaled_s2(params_.n, params_.q);
        params_.poly_ops->mul_scalar(scaled_s2, s_squared, scalar);
        
        // b = -a*s + e + 2^(i*w) * s^2
        rlk_key->pk0 = new Polynomial(params_.n, params_.q);
        Polynomial temp(params_.n, params_.q);
        params_.poly_ops->mul_ntt(temp, *rlk_key->pk1, *sk.sk);
        params_.poly_ops->sub(temp, e, temp);
        params_.poly_ops->add(*rlk_key->pk0, temp, scaled_s2);
        
        rlk.rlk_keys.push_back(rlk_key);
    }
}

void FHEContext::encode(Plaintext& pt, const std::vector<uint64_t>& values) {
    pt.poly = new Polynomial(params_.n, params_.t);
    pt.is_ntt_form = false;
    
    // Simple coefficient encoding
    std::vector<uint256_t> h_coeffs(params_.n);
    for (size_t i = 0; i < std::min(values.size(), size_t(params_.n)); i++) {
        h_coeffs[i] = uint256_t(values[i]);
    }
    
    cudaMemcpy(pt.poly->coeffs, h_coeffs.data(), params_.n * sizeof(uint256_t),
               cudaMemcpyHostToDevice);
}

void FHEContext::decode(std::vector<uint64_t>& values, const Plaintext& pt) {
    std::vector<uint256_t> h_coeffs(params_.n);
    cudaMemcpy(h_coeffs.data(), pt.poly->coeffs, params_.n * sizeof(uint256_t),
               cudaMemcpyDeviceToHost);
    
    values.clear();
    for (uint32_t i = 0; i < params_.n; i++) {
        values.push_back(h_coeffs[i].limbs[0]);
    }
}

void FHEContext::encrypt(Ciphertext& ct, const Plaintext& pt, const PublicKey& pk) {
    ct.components.resize(2);
    ct.components[0] = new Polynomial(params_.n, params_.q);
    ct.components[1] = new Polynomial(params_.n, params_.q);
    ct.level = 0;
    ct.is_ntt_form = false;
    
    // Generate random ternary u
    Polynomial u(params_.n, params_.q);
    sample_ternary_polynomial(u);
    
    // Generate errors
    Polynomial e1(params_.n, params_.q), e2(params_.n, params_.q);
    sample_error_polynomial(e1);
    sample_error_polynomial(e2);
    
    // Scale plaintext by delta
    Polynomial scaled_pt(params_.n, params_.q);
    params_.poly_ops->mul_scalar(scaled_pt, *pt.poly, params_.delta);
    
    // c0 = pk0 * u + e1 + delta * m
    Polynomial temp(params_.n, params_.q);
    params_.poly_ops->mul_ntt(temp, *pk.pk0, u);
    params_.poly_ops->add(temp, temp, e1);
    params_.poly_ops->add(*ct.components[0], temp, scaled_pt);
    
    // c1 = pk1 * u + e2
    params_.poly_ops->mul_ntt(temp, *pk.pk1, u);
    params_.poly_ops->add(*ct.components[1], temp, e2);
    
    ct.noise_budget = params_.security.sigma * 10; // Rough estimate
}

void FHEContext::decrypt(Plaintext& pt, const Ciphertext& ct, const SecretKey& sk) {
    pt.poly = new Polynomial(params_.n, params_.q);
    pt.is_ntt_form = false;
    
    // Compute c0 + c1 * s
    Polynomial temp(params_.n, params_.q);
    params_.poly_ops->mul_ntt(temp, *ct.components[1], *sk.sk);
    params_.poly_ops->add(temp, *ct.components[0], temp);
    
    // Scale down by delta and reduce mod t
    uint32_t num_blocks = (params_.n + 255) / 256;
    poly_mod_switch_kernel<<<num_blocks, 256>>>(
        pt.poly->coeffs, temp.coeffs, params_.q, params_.t, params_.n
    );
}

void FHEContext::add(Ciphertext& result, const Ciphertext& a, const Ciphertext& b) {
    result.components.resize(2);
    result.components[0] = new Polynomial(params_.n, params_.q);
    result.components[1] = new Polynomial(params_.n, params_.q);
    
    params_.poly_ops->add(*result.components[0], *a.components[0], *b.components[0]);
    params_.poly_ops->add(*result.components[1], *a.components[1], *b.components[1]);
    
    result.noise_budget = std::min(a.noise_budget, b.noise_budget);
    result.level = std::max(a.level, b.level);
}

void FHEContext::multiply(Ciphertext& result, const Ciphertext& a, const Ciphertext& b, 
                         const RelinKeys& rlk) {
    // Multiply ciphertexts: creates 3 components
    result.components.resize(3);
    for (int i = 0; i < 3; i++) {
        result.components[i] = new Polynomial(params_.n, params_.q);
    }
    
    // c0 = a0 * b0
    params_.poly_ops->mul_ntt(*result.components[0], *a.components[0], *b.components[0]);
    
    // c1 = a0*b1 + a1*b0
    Polynomial temp1(params_.n, params_.q), temp2(params_.n, params_.q);
    params_.poly_ops->mul_ntt(temp1, *a.components[0], *b.components[1]);
    params_.poly_ops->mul_ntt(temp2, *a.components[1], *b.components[0]);
    params_.poly_ops->add(*result.components[1], temp1, temp2);
    
    // c2 = a1 * b1
    params_.poly_ops->mul_ntt(*result.components[2], *a.components[1], *b.components[1]);
    
    // Relinearize to reduce back to 2 components
    relinearize(result, rlk);
    
    result.noise_budget = a.noise_budget + b.noise_budget + 10; // Rough estimate
    result.level = std::max(a.level, b.level);
}

void FHEContext::relinearize(Ciphertext& ct, const RelinKeys& rlk) {
    if (ct.components.size() <= 2) return;
    
    // Decompose c2 into base-2^w representation
    // Apply key switching
    // This is a simplified placeholder
    
    // After relinearization, reduce to 2 components
    ct.components.resize(2);
}

// Helper functions
void FHEContext::sample_error_polynomial(Polynomial& poly) {
    uint32_t num_blocks = (params_.n + 255) / 256;
    sample_gaussian_kernel<<<num_blocks, 256>>>(
        poly.coeffs, params_.q, params_.security.sigma, rand(), params_.n
    );
}

void FHEContext::sample_uniform_polynomial(Polynomial& poly) {
    uint32_t num_blocks = (params_.n + 255) / 256;
    sample_uniform_kernel<<<num_blocks, 256>>>(
        poly.coeffs, params_.q, rand(), params_.n
    );
}

void FHEContext::sample_ternary_polynomial(Polynomial& poly) {
    uint32_t num_blocks = (params_.n + 255) / 256;
    sample_ternary_kernel<<<num_blocks, 256>>>(
        poly.coeffs, params_.q, 0.5f, rand(), params_.n
    );
}

void FHEContext::init_rng(uint64_t seed) {
    // Allocate RNG states on device
    cudaMalloc(&d_rng_states_, params_.n * sizeof(curandState_t));
    
    // Initialize states (simplified)
}

// Batch encoder
BatchEncoder::BatchEncoder(const FHEContext& context) 
    : context_(context), slot_count_(context.params().n / 2) {
}

void BatchEncoder::encode(Plaintext& pt, const std::vector<uint64_t>& values) {
    // SIMD encoding using Chinese Remainder Theorem
    // Placeholder implementation
    context_.encode(pt, values);
}

void BatchEncoder::decode(std::vector<uint64_t>& values, const Plaintext& pt) {
    context_.decode(values, pt);
}

} // namespace fhe
