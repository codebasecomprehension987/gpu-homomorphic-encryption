# API Reference

Complete API documentation for FHE-CUDA library.

---

## Table of Contents

1. [Core Types](#core-types)
2. [FHE Context](#fhe-context)
3. [Key Management](#key-management)
4. [Encryption Operations](#encryption-operations)
5. [Homomorphic Operations](#homomorphic-operations)
6. [Polynomial Operations](#polynomial-operations)
7. [NTT Engine](#ntt-engine)
8. [Utility Functions](#utility-functions)

---

## Core Types

### uint256_t

256-bit unsigned integer for modular arithmetic.

```cpp
struct uint256_t {
    uint64_t limbs[4];  // Little-endian representation
    
    __host__ __device__ uint256_t();
    __host__ __device__ uint256_t(uint64_t val);
    __host__ __device__ uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3);
};
```

**Members:**
- `limbs[4]`: Four 64-bit limbs, little-endian (limbs[0] is LSB)

**Constructors:**
- `uint256_t()`: Initialize to zero
- `uint256_t(uint64_t val)`: Initialize from 64-bit value
- `uint256_t(uint64_t, uint64_t, uint64_t, uint64_t)`: Initialize all limbs

---

### SecurityParams

Security and scheme parameters.

```cpp
struct SecurityParams {
    uint32_t lambda;           // Security level (128, 192, 256)
    uint32_t poly_degree;      // Polynomial degree (power of 2)
    uint32_t log_q;            // Log of ciphertext modulus
    float sigma;               // Gaussian noise std deviation
    uint32_t hamming_weight;   // Secret key Hamming weight
};
```

**Parameters:**
- `lambda`: Security level in bits (128, 192, or 256)
- `poly_degree`: Must be power of 2 (1024, 2048, 4096, 8192, 16384)
- `log_q`: Logarithm of ciphertext modulus (60-218 bits)
- `sigma`: Standard deviation for discrete Gaussian (typically 3.2)
- `hamming_weight`: Number of non-zero coefficients in secret key

---

### Ciphertext

Encrypted data structure.

```cpp
struct Ciphertext {
    std::vector<Polynomial*> components;  // (c0, c1, ..., cn)
    uint32_t level;                       // Modulus level
    float noise_budget;                   // Remaining noise budget
    bool is_ntt_form;                     // NTT domain flag
};
```

**Members:**
- `components`: Vector of polynomial components (typically 2)
- `level`: Current modulus chain level
- `noise_budget`: Estimated remaining noise budget in bits
- `is_ntt_form`: Whether polynomials are in NTT domain

---

### Plaintext

Plaintext data structure.

```cpp
struct Plaintext {
    Polynomial* poly;
    bool is_ntt_form;
};
```

**Members:**
- `poly`: Polynomial representation
- `is_ntt_form`: Whether in NTT domain

---

## FHE Context

### FHEContext

Main class for FHE operations.

```cpp
class FHEContext {
public:
    FHEContext(const SecurityParams& params);
    ~FHEContext();
    
    // Key operations
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
    void multiply(Ciphertext& result, const Ciphertext& a, const Ciphertext& b, 
                  const RelinKeys& rlk);
    void multiply_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
    
    // Advanced operations
    void relinearize(Ciphertext& ct, const RelinKeys& rlk);
    void mod_switch_to_next(Ciphertext& ct);
    void rotate_rows(Ciphertext& result, const Ciphertext& ct, int steps, 
                     const GaloisKeys& gal_keys);
    
    const SchemeParams& params() const;
};
```

#### Constructor

```cpp
FHEContext(const SecurityParams& params);
```

**Parameters:**
- `params`: Security and scheme parameters

**Example:**
```cpp
fhe::SecurityParams params;
params.lambda = 128;
params.poly_degree = 4096;
params.log_q = 120;
params.sigma = 3.2;

fhe::FHEContext ctx(params);
```

---

## Key Management

### keygen

Generate public and secret key pair.

```cpp
void keygen(PublicKey& pk, SecretKey& sk);
```

**Parameters:**
- `pk` (output): Public key
- `sk` (output): Secret key

**Example:**
```cpp
fhe::PublicKey pk;
fhe::SecretKey sk;
ctx.keygen(pk, sk);
```

**Note:** Secret key is sampled from ternary distribution {-1, 0, 1}.

---

### relinkey_gen

Generate relinearization keys for multiplication.

```cpp
void relinkey_gen(RelinKeys& rlk, const SecretKey& sk, uint32_t decomp_bits = 16);
```

**Parameters:**
- `rlk` (output): Relinearization keys
- `sk`: Secret key
- `decomp_bits`: Decomposition bit width (default: 16)

**Example:**
```cpp
fhe::RelinKeys rlk;
ctx.relinkey_gen(rlk, sk, 16);
```

**Note:** Larger `decomp_bits` means fewer keys but slower relinearization.

---

### galoiskey_gen

Generate Galois keys for rotation operations.

```cpp
void galoiskey_gen(GaloisKeys& gal_keys, const SecretKey& sk);
```

**Parameters:**
- `gal_keys` (output): Galois keys
- `sk`: Secret key

**Example:**
```cpp
fhe::GaloisKeys gal_keys;
ctx.galoiskey_gen(gal_keys, sk);
```

---

## Encryption Operations

### encode

Encode plaintext values into polynomial.

```cpp
void encode(Plaintext& pt, const std::vector<uint64_t>& values);
```

**Parameters:**
- `pt` (output): Encoded plaintext
- `values`: Vector of integer values

**Example:**
```cpp
std::vector<uint64_t> values = {10, 20, 30, 40};
fhe::Plaintext pt;
ctx.encode(pt, values);
```

---

### decode

Decode polynomial to plaintext values.

```cpp
void decode(std::vector<uint64_t>& values, const Plaintext& pt);
```

**Parameters:**
- `values` (output): Decoded integer values
- `pt`: Encoded plaintext

**Example:**
```cpp
std::vector<uint64_t> values;
ctx.decode(values, pt);
```

---

### encrypt

Encrypt plaintext to ciphertext.

```cpp
void encrypt(Ciphertext& ct, const Plaintext& pt, const PublicKey& pk);
```

**Parameters:**
- `ct` (output): Encrypted ciphertext
- `pt`: Plaintext to encrypt
- `pk`: Public key

**Example:**
```cpp
fhe::Ciphertext ct;
ctx.encrypt(ct, pt, pk);
```

**Complexity:** O(N log N) where N is polynomial degree

---

### decrypt

Decrypt ciphertext to plaintext.

```cpp
void decrypt(Plaintext& pt, const Ciphertext& ct, const SecretKey& sk);
```

**Parameters:**
- `pt` (output): Decrypted plaintext
- `ct`: Ciphertext to decrypt
- `sk`: Secret key

**Example:**
```cpp
fhe::Plaintext pt_result;
ctx.decrypt(pt_result, ct, sk);
```

**Complexity:** O(N log N)

---

## Homomorphic Operations

### add

Homomorphic addition of two ciphertexts.

```cpp
void add(Ciphertext& result, const Ciphertext& a, const Ciphertext& b);
```

**Parameters:**
- `result` (output): Sum ciphertext
- `a`: First operand
- `b`: Second operand

**Example:**
```cpp
fhe::Ciphertext ct_sum;
ctx.add(ct_sum, ct1, ct2);
```

**Complexity:** O(N)
**Noise Growth:** Linear (noise_a + noise_b)

---

### add_plain

Add plaintext to ciphertext.

```cpp
void add_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
```

**Parameters:**
- `result` (output): Result ciphertext
- `ct`: Ciphertext operand
- `pt`: Plaintext operand

**Example:**
```cpp
fhe::Ciphertext ct_result;
ctx.add_plain(ct_result, ct, pt);
```

**Complexity:** O(N)
**Noise Growth:** Minimal

---

### multiply

Homomorphic multiplication with relinearization.

```cpp
void multiply(Ciphertext& result, const Ciphertext& a, const Ciphertext& b, 
              const RelinKeys& rlk);
```

**Parameters:**
- `result` (output): Product ciphertext
- `a`: First operand
- `b`: Second operand
- `rlk`: Relinearization keys

**Example:**
```cpp
fhe::Ciphertext ct_product;
ctx.multiply(ct_product, ct1, ct2, rlk);
```

**Complexity:** O(N log N)
**Noise Growth:** Multiplicative (noise_a * noise_b * N)

---

### multiply_plain

Multiply ciphertext by plaintext.

```cpp
void multiply_plain(Ciphertext& result, const Ciphertext& ct, const Plaintext& pt);
```

**Parameters:**
- `result` (output): Result ciphertext
- `ct`: Ciphertext operand
- `pt`: Plaintext operand

**Example:**
```cpp
fhe::Ciphertext ct_result;
ctx.multiply_plain(ct_result, ct, pt);
```

**Complexity:** O(N log N)
**Noise Growth:** Moderate

---

### relinearize

Reduce ciphertext size after multiplication.

```cpp
void relinearize(Ciphertext& ct, const RelinKeys& rlk);
```

**Parameters:**
- `ct` (in/out): Ciphertext to relinearize
- `rlk`: Relinearization keys

**Example:**
```cpp
ctx.relinearize(ct, rlk);
```

**Note:** Reduces 3-component ciphertext to 2 components

---

### mod_switch_to_next

Switch to next modulus level for noise management.

```cpp
void mod_switch_to_next(Ciphertext& ct);
```

**Parameters:**
- `ct` (in/out): Ciphertext to switch

**Example:**
```cpp
ctx.mod_switch_to_next(ct);
```

**Effect:** Reduces noise and modulus simultaneously

---

## Polynomial Operations

### PolynomialOps

Class for polynomial arithmetic.

```cpp
class PolynomialOps {
public:
    PolynomialOps(uint32_t max_degree, const uint256_t& modulus, NTTEngine* ntt);
    
    void add(Polynomial& result, const Polynomial& a, const Polynomial& b);
    void sub(Polynomial& result, const Polynomial& a, const Polynomial& b);
    void mul_ntt(Polynomial& result, const Polynomial& a, const Polynomial& b);
    void mul_scalar(Polynomial& result, const Polynomial& a, const uint256_t& scalar);
};
```

---

## NTT Engine

### NTTEngine

Number Theoretic Transform engine.

```cpp
class NTTEngine {
public:
    NTTEngine(uint32_t polynomial_degree, const uint256_t& modulus);
    
    void forward(uint256_t* d_data);
    void inverse(uint256_t* d_data);
    void multiply(uint256_t* d_result, const uint256_t* d_a, const uint256_t* d_b);
};
```

#### forward

Forward NTT transform.

```cpp
void forward(uint256_t* d_data);
```

**Parameters:**
- `d_data` (in/out): Device pointer to polynomial coefficients

**Example:**
```cpp
ntt.forward(d_polynomial);
```

**Complexity:** O(N log N)

---

#### inverse

Inverse NTT transform.

```cpp
void inverse(uint256_t* d_data);
```

**Parameters:**
- `d_data` (in/out): Device pointer to NTT coefficients

**Example:**
```cpp
ntt.inverse(d_polynomial);
```

**Complexity:** O(N log N)

---

## Utility Functions

### Device Functions

Modular arithmetic device functions.

```cpp
__device__ uint256_t add_mod(const uint256_t& a, const uint256_t& b, 
                             const uint256_t& modulus);
                             
__device__ uint256_t sub_mod(const uint256_t& a, const uint256_t& b, 
                             const uint256_t& modulus);
                             
__device__ uint256_t mul_mod_montgomery(const uint256_t& a, const uint256_t& b, 
                                        const uint256_t& modulus, 
                                        const uint256_t& inv);
```

---

## Error Codes

The library uses exceptions for error handling:

- `std::runtime_error`: General runtime errors
- `cudaError_t`: CUDA-specific errors (check with `cudaGetLastError()`)

---

## Performance Considerations

### Operation Complexity

| Operation | Time Complexity | GPU Time (N=8192) |
|-----------|----------------|-------------------|
| Key Generation | O(N log N) | ~100ms |
| Encryption | O(N log N) | ~8ms |
| Decryption | O(N log N) | ~3ms |
| Addition | O(N) | ~0.1ms |
| Multiplication | O(N log N) | ~40ms |
| NTT Forward | O(N log N) | ~2ms |

### Memory Requirements

| Object | Size (N=8192) |
|--------|--------------|
| Polynomial | 32 KB |
| Ciphertext | 64 KB |
| Public Key | 64 KB |
| Relin Keys | 512 KB |

---

## Thread Safety

The FHE context is **not thread-safe**. For multi-threaded applications:
- Create separate FHEContext instances per thread, OR
- Use external synchronization (mutex/lock)

CUDA streams are used internally for GPU parallelism.

---

## Best Practices

1. **Reuse Keys**: Generate keys once, reuse for multiple encryptions
2. **Batch Operations**: Encode multiple values using SIMD slots
3. **Modulus Switching**: Use after multiplication to reduce noise
4. **Memory Management**: Reuse ciphertext objects when possible
5. **Stream Management**: Library handles CUDA streams internally

---

## Example: Complete Workflow

```cpp
#include "fhe.cuh"

int main() {
    // 1. Setup
    fhe::SecurityParams params;
    params.lambda = 128;
    params.poly_degree = 4096;
    params.log_q = 120;
    params.sigma = 3.2;
    
    fhe::FHEContext ctx(params);
    
    // 2. Key Generation
    fhe::PublicKey pk;
    fhe::SecretKey sk;
    fhe::RelinKeys rlk;
    
    ctx.keygen(pk, sk);
    ctx.relinkey_gen(rlk, sk);
    
    // 3. Encode and Encrypt
    std::vector<uint64_t> data1 = {10, 20};
    std::vector<uint64_t> data2 = {3, 4};
    
    fhe::Plaintext pt1, pt2;
    ctx.encode(pt1, data1);
    ctx.encode(pt2, data2);
    
    fhe::Ciphertext ct1, ct2;
    ctx.encrypt(ct1, pt1, pk);
    ctx.encrypt(ct2, pt2, pk);
    
    // 4. Homomorphic Operations
    fhe::Ciphertext ct_sum, ct_product;
    ctx.add(ct_sum, ct1, ct2);              // [13, 24]
    ctx.multiply(ct_product, ct1, ct2, rlk); // [30, 80]
    
    // 5. Decrypt and Decode
    fhe::Plaintext pt_result;
    ctx.decrypt(pt_result, ct_product, sk);
    
    std::vector<uint64_t> result;
    ctx.decode(result, pt_result);
    
    return 0;
}
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [NTT_OPTIMIZATION.md](NTT_OPTIMIZATION.md) - Performance optimization
- [BUILDING.md](BUILDING.md) - Build instructions
