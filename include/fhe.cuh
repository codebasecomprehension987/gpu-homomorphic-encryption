#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "bigint.cuh"
#include "ntt.cuh"
#include "polynomial.cuh"
#include "rns.cuh"

namespace fhe {

// Security parameters
struct SecurityParams {
    uint32_t lambda;           // Security level (e.g., 128, 192, 256 bits)
    uint32_t poly_degree;      // Polynomial degree (power of 2, e.g., 4096, 8192, 16384)
    uint32_t log_q;            // Log of ciphertext modulus
    float sigma;               // Gaussian noise standard deviation
    uint32_t hamming_weight;   // For ternary secret keys
};

// BGV/BFV scheme parameters
struct SchemeParams {
    SecurityParams security;
    
    uint32_t n;                // Polynomial degree
    uint256_t q;               // Ciphertext modulus
    uint256_t t;               // Plaintext modulus
    uint256_t delta;           // Scaling factor: ⌊q/t⌉
    
    RNSContext* rns_ctx;       // RNS context for large moduli
    NTTEngine* ntt_engine;     // NTT engine
    PolynomialOps* poly_ops;   // Polynomial operations
    
    // Bootstrapping parameters
    uint32_t num_levels;       // Modulus chain length
    std::vector<uint256_t> modulus_chain;
};

// Public key components
struct PublicKey {
    Polynomial* pk0;  // b = -a*s + e
    Polynomial* pk1;  // a (uniform random)
};

// Secret key
struct SecretKey {
    Polynomial* sk;   // Secret polynomial s (typically ternary)
};

// Relinearization keys for multiplication
struct RelinKeys {
    std::vector<PublicKey*> rlk_keys;  // Key switching keys
    uint32_t decomp_bits;              // Decomposition bit length
};

// Galois keys for rotation
struct GaloisKeys {
    std::vector<PublicKey*> gal_keys;  // One per Galois element
};

// Ciphertext structure
struct Ciphertext {
    std::vector<Polynomial*> components;  // (c0, c1, ..., cn)
    uint32_t level;                       // Current modulus level
    float noise_budget;                   // Estimated remaining noise budget
    bool is_ntt_form;                     // Whether in NTT domain
};

// Plaintext structure
struct Plaintext {
    Polynomial* poly;
    bool is_ntt_form;
};

// Main FHE context
class FHEContext {
public:
    FHEContext(const SecurityParams& params);
    ~FHEContext();
    
    // Key generation
    void keygen(PublicKey& pk, SecretKey& sk);
    void relinkey_gen(RelinKeys& rlk, const SecretKey& sk, uint32_t decomp_bits = 16);
    void galoiskey_gen(GaloisKeys& gal_keys, const SecretKey& sk);
    
    // Encoding/Decoding
    void encode(Plaintext& pt, const std::vector<uint64_t>& values);
    void decode(std::vector<uint64_t>& values, const Plaintext& pt);
    
    // Encryption/Decryption
    void encrypt(Ciphertext& ct, const Plaintext& pt, const PublicKey& pk);
    void decrypt(Plaintext& pt, const Ciphertext& ct, const SecretKey& sk);
    
    // Homomorphic operations
    void add(Ciphertext& result, const Ciphertext& a, const Ciphertext& b);
    void add_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
    void sub(Ciphertext& result, const Ciphertext& a, const Ciphertext& b);
    void sub_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
    
    void multiply(Ciphertext& result, const Ciphertext& a, const Ciphertext& b, 
                  const RelinKeys& rlk);
    void multiply_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
    
    void relinearize(Ciphertext& ct, const RelinKeys& rlk);
    
    // Modulus switching for noise management
    void mod_switch_to_next(Ciphertext& ct);
    void mod_switch_to_level(Ciphertext& ct, uint32_t target_level);
    
    // Rotation and permutation
    void rotate_rows(Ciphertext& result, const Ciphertext& ct, int steps, 
                     const GaloisKeys& gal_keys);
    void rotate_columns(Ciphertext& result, const Ciphertext& ct, 
                       const GaloisKeys& gal_keys);
    
    // Bootstrapping (noise refresh)
    void bootstrap(Ciphertext& ct, const SecretKey& sk);
    
    // Noise estimation
    float estimate_noise_budget(const Ciphertext& ct, const SecretKey& sk);
    
    const SchemeParams& params() const { return params_; }
    
private:
    SchemeParams params_;
    
    // Internal helper functions
    void sample_error_polynomial(Polynomial& poly);
    void sample_uniform_polynomial(Polynomial& poly);
    void sample_ternary_polynomial(Polynomial& poly);
    
    void key_switch(Ciphertext& result, const Ciphertext& ct, 
                   const PublicKey& ksk);
    
    // Bootstrapping helpers
    void extract_lsb(Ciphertext& result, const Ciphertext& ct);
    void blind_rotate(Ciphertext& result, const Ciphertext& ct, const SecretKey& sk);
    void modulus_raise(Ciphertext& ct);
    
    // CUDA streams for pipelining
    std::vector<cudaStream_t> streams_;
    
    // Random number generation
    curandState_t* d_rng_states_;
    void init_rng(uint64_t seed);
};

// Batch encoder for SIMD operations
class BatchEncoder {
public:
    BatchEncoder(const FHEContext& context);
    
    void encode(Plaintext& pt, const std::vector<uint64_t>& values);
    void decode(std::vector<uint64_t>& values, const Plaintext& pt);
    
    uint32_t slot_count() const { return slot_count_; }
    
private:
    uint32_t slot_count_;
    const FHEContext& context_;
    
    // Cached NTT for batch encoding
    std::vector<uint256_t> matrix_reps_;
};

// Performance monitoring
struct PerfStats {
    double encrypt_time_ms;
    double decrypt_time_ms;
    double add_time_ms;
    double mul_time_ms;
    double relin_time_ms;
    double mod_switch_time_ms;
    double bootstrap_time_ms;
    
    uint64_t num_encryptions;
    uint64_t num_multiplications;
    uint64_t num_bootstraps;
};

class PerformanceMonitor {
public:
    void start_timer(const std::string& operation);
    void stop_timer(const std::string& operation);
    
    void record_operation(const std::string& op_name);
    
    PerfStats get_stats() const;
    void print_stats() const;
    void reset();
    
private:
    std::map<std::string, cudaEvent_t> start_events_;
    std::map<std::string, cudaEvent_t> stop_events_;
    PerfStats stats_;
};

} // namespace fhe
