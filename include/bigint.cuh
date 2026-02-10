#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "ptx_bigint.cuh"

namespace fhe {

// 256-bit unsigned integer for FHE operations
struct uint256_t {
    uint64_t limbs[4]; // Little-endian: limbs[0] is least significant
    
    __host__ __device__ uint256_t() {
        limbs[0] = limbs[1] = limbs[2] = limbs[3] = 0;
    }
    
    __host__ __device__ uint256_t(uint64_t val) {
        limbs[0] = val;
        limbs[1] = limbs[2] = limbs[3] = 0;
    }
    
    __host__ __device__ uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
        limbs[0] = l0; limbs[1] = l1; limbs[2] = l2; limbs[3] = l3;
    }
};

// Device functions for modular arithmetic
__device__ __forceinline__
uint256_t add_mod(const uint256_t& a, const uint256_t& b, const uint256_t& modulus) {
    uint256_t result;
    
    // Add with carry chain
    ptx::add_cc(result.limbs[0], a.limbs[0], b.limbs[0]);
    ptx::addc_cc(result.limbs[1], a.limbs[1], b.limbs[1]);
    ptx::addc_cc(result.limbs[2], a.limbs[2], b.limbs[2]);
    ptx::addc(result.limbs[3], a.limbs[3], b.limbs[3]);
    
    // Conditional subtraction if result >= modulus
    uint256_t temp;
    ptx::sub_cc(temp.limbs[0], result.limbs[0], modulus.limbs[0]);
    ptx::subc_cc(temp.limbs[1], result.limbs[1], modulus.limbs[1]);
    ptx::subc_cc(temp.limbs[2], result.limbs[2], modulus.limbs[2]);
    ptx::subc(temp.limbs[3], result.limbs[3], modulus.limbs[3]);
    
    // Check if subtraction underflowed (result was < modulus)
    bool underflow = (temp.limbs[3] > result.limbs[3]);
    
    return underflow ? result : temp;
}

__device__ __forceinline__
uint256_t sub_mod(const uint256_t& a, const uint256_t& b, const uint256_t& modulus) {
    uint256_t result;
    
    // Subtract with borrow chain
    ptx::sub_cc(result.limbs[0], a.limbs[0], b.limbs[0]);
    ptx::subc_cc(result.limbs[1], a.limbs[1], b.limbs[1]);
    ptx::subc_cc(result.limbs[2], a.limbs[2], b.limbs[2]);
    ptx::subc(result.limbs[3], a.limbs[3], b.limbs[3]);
    
    // If borrow occurred (a < b), add modulus
    bool borrow = (result.limbs[3] > a.limbs[3]);
    
    if (borrow) {
        uint256_t temp;
        ptx::add_cc(temp.limbs[0], result.limbs[0], modulus.limbs[0]);
        ptx::addc_cc(temp.limbs[1], result.limbs[1], modulus.limbs[1]);
        ptx::addc_cc(temp.limbs[2], result.limbs[2], modulus.limbs[2]);
        ptx::addc(temp.limbs[3], result.limbs[3], modulus.limbs[3]);
        return temp;
    }
    
    return result;
}

// Montgomery multiplication for efficient modular multiplication
__device__ __forceinline__
uint256_t mul_mod_montgomery(const uint256_t& a, const uint256_t& b, 
                              const uint256_t& modulus, const uint256_t& inv) {
    // Implement CIOS (Coarsely Integrated Operand Scanning) method
    uint64_t t[8] = {0}; // Temporary storage for 512-bit product
    
    // Multiply a * b
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            ptx::mul_u64_u128(a.limbs[i], b.limbs[j], lo, hi);
            
            // Add to accumulator
            ptx::add_cc(lo, lo, carry);
            ptx::addc(hi, hi, 0);
            ptx::add_cc(lo, lo, t[i + j]);
            ptx::addc(carry, hi, 0);
            
            t[i + j] = lo;
        }
        t[i + 4] = carry;
    }
    
    // Montgomery reduction
    for (int i = 0; i < 4; i++) {
        uint64_t m = t[i] * inv.limbs[0]; // Multiply by inverse mod 2^64
        
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            ptx::mul_u64_u128(m, modulus.limbs[j], lo, hi);
            
            ptx::add_cc(lo, lo, carry);
            ptx::addc(hi, hi, 0);
            ptx::add_cc(lo, lo, t[i + j]);
            ptx::addc(carry, hi, 0);
            
            t[i + j] = lo;
        }
        
        // Propagate carry
        for (int j = 4; j < 8 - i; j++) {
            ptx::add_cc(t[i + j], t[i + j], carry);
            ptx::addc(carry, 0, 0);
        }
    }
    
    // Extract result from upper half
    uint256_t result;
    result.limbs[0] = t[4];
    result.limbs[1] = t[5];
    result.limbs[2] = t[6];
    result.limbs[3] = t[7];
    
    // Final conditional subtraction
    uint256_t temp;
    ptx::sub_cc(temp.limbs[0], result.limbs[0], modulus.limbs[0]);
    ptx::subc_cc(temp.limbs[1], result.limbs[1], modulus.limbs[1]);
    ptx::subc_cc(temp.limbs[2], result.limbs[2], modulus.limbs[2]);
    ptx::subc(temp.limbs[3], result.limbs[3], modulus.limbs[3]);
    
    bool underflow = (temp.limbs[3] > result.limbs[3]);
    return underflow ? result : temp;
}

// Modular exponentiation using binary method
__device__ __forceinline__
uint256_t pow_mod(uint256_t base, uint256_t exp, const uint256_t& modulus, 
                  const uint256_t& mont_inv) {
    uint256_t result(1);
    
    for (int i = 255; i >= 0; i--) {
        // Square
        result = mul_mod_montgomery(result, result, modulus, mont_inv);
        
        // Check bit
        int limb = i / 64;
        int bit = i % 64;
        if (exp.limbs[limb] & (1ULL << bit)) {
            result = mul_mod_montgomery(result, base, modulus, mont_inv);
        }
    }
    
    return result;
}

// Compute modular inverse using extended Euclidean algorithm
__host__ uint256_t compute_montgomery_inverse(const uint256_t& modulus);

// Precompute Montgomery parameters
struct MontgomeryParams {
    uint256_t modulus;
    uint256_t r_squared;  // R^2 mod N where R = 2^256
    uint256_t inv;        // -N^(-1) mod R
};

__host__ MontgomeryParams compute_montgomery_params(const uint256_t& modulus);

} // namespace fhe
