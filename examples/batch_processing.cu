/**
 * Batch Processing Example
 * 
 * Demonstrates:
 * - SIMD batch encoding
 * - Processing multiple ciphertexts efficiently
 * - Parallel encryption/decryption
 * - Performance measurement
 */

#include <iostream>
#include <vector>
#include <chrono>
#include "fhe.cuh"

// Timing utility
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    void reset() {
        start = std::chrono::high_resolution_clock::now();
    }
};

int main() {
    std::cout << "=== FHE Batch Processing Example ===" << std::endl;
    std::cout << std::endl;
    
    // Setup
    std::cout << "Setting up FHE context..." << std::endl;
    fhe::SecurityParams params;
    params.lambda = 128;
    params.poly_degree = 4096;
    params.log_q = 120;
    params.sigma = 3.2;
    params.hamming_weight = 64;
    
    fhe::FHEContext ctx(params);
    
    // Create batch encoder
    fhe::BatchEncoder encoder(ctx);
    uint32_t slot_count = encoder.slot_count();
    
    std::cout << "  Polynomial degree: " << params.poly_degree << std::endl;
    std::cout << "  Available slots (SIMD): " << slot_count << std::endl;
    std::cout << std::endl;
    
    // Generate keys
    std::cout << "Generating keys..." << std::endl;
    Timer timer;
    
    fhe::PublicKey pk;
    fhe::SecretKey sk;
    fhe::RelinKeys rlk;
    
    ctx.keygen(pk, sk);
    double keygen_time = timer.elapsed_ms();
    
    timer.reset();
    ctx.relinkey_gen(rlk, sk);
    double rlk_time = timer.elapsed_ms();
    
    std::cout << "  Key generation: " << keygen_time << " ms" << std::endl;
    std::cout << "  Relin key generation: " << rlk_time << " ms" << std::endl;
    std::cout << std::endl;
    
    // ===== BATCH ENCODING =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 1: Batch Encoding (SIMD)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Prepare batch data (fill all slots)
    std::vector<uint64_t> batch_data(slot_count);
    for (uint32_t i = 0; i < slot_count; i++) {
        batch_data[i] = i + 1;  // 1, 2, 3, ..., slot_count
    }
    
    std::cout << "Encoding " << slot_count << " values into single plaintext..." << std::endl;
    fhe::Plaintext pt_batch;
    encoder.encode(pt_batch, batch_data);
    std::cout << "  ✓ All " << slot_count << " values packed into one polynomial" << std::endl;
    std::cout << std::endl;
    
    // Encrypt batch
    std::cout << "Encrypting batch..." << std::endl;
    timer.reset();
    fhe::Ciphertext ct_batch;
    ctx.encrypt(ct_batch, pt_batch, pk);
    double batch_encrypt_time = timer.elapsed_ms();
    
    std::cout << "  Encryption time: " << batch_encrypt_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (slot_count / batch_encrypt_time * 1000.0) << " values/sec" << std::endl;
    std::cout << std::endl;
    
    // Decrypt batch
    std::cout << "Decrypting batch..." << std::endl;
    timer.reset();
    fhe::Plaintext pt_batch_result;
    ctx.decrypt(pt_batch_result, ct_batch, sk);
    double batch_decrypt_time = timer.elapsed_ms();
    
    std::vector<uint64_t> batch_result;
    encoder.decode(batch_result, pt_batch_result);
    
    std::cout << "  Decryption time: " << batch_decrypt_time << " ms" << std::endl;
    
    // Verify first 10 values
    std::cout << "  First 10 values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << batch_result[i] << " ";
    }
    std::cout << "..." << std::endl;
    std::cout << std::endl;
    
    // ===== BATCH HOMOMORPHIC OPERATIONS =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 2: Batch Homomorphic Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Create two batches
    std::vector<uint64_t> batch_a(slot_count);
    std::vector<uint64_t> batch_b(slot_count);
    
    for (uint32_t i = 0; i < slot_count; i++) {
        batch_a[i] = i + 1;      // 1, 2, 3, ...
        batch_b[i] = 2;          // All 2s
    }
    
    std::cout << "Batch A: First value = " << batch_a[0] << ", Last value = " << batch_a[slot_count-1] << std::endl;
    std::cout << "Batch B: All values = 2" << std::endl;
    std::cout << std::endl;
    
    // Encode and encrypt
    fhe::Plaintext pt_a, pt_b;
    encoder.encode(pt_a, batch_a);
    encoder.encode(pt_b, batch_b);
    
    fhe::Ciphertext ct_a, ct_b;
    ctx.encrypt(ct_a, pt_a, pk);
    ctx.encrypt(ct_b, pt_b, pk);
    
    // Batch addition
    std::cout << "Computing: ct_a + ct_b (adds each slot independently)" << std::endl;
    timer.reset();
    fhe::Ciphertext ct_add_batch;
    ctx.add(ct_add_batch, ct_a, ct_b);
    double add_time = timer.elapsed_ms();
    
    std::cout << "  Addition time: " << add_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (slot_count / add_time * 1000.0) << " ops/sec" << std::endl;
    std::cout << std::endl;
    
    // Batch multiplication
    std::cout << "Computing: ct_a * ct_b (multiplies each slot independently)" << std::endl;
    timer.reset();
    fhe::Ciphertext ct_mul_batch;
    ctx.multiply(ct_mul_batch, ct_a, ct_b, rlk);
    double mul_time = timer.elapsed_ms();
    
    std::cout << "  Multiplication time: " << mul_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (slot_count / mul_time * 1000.0) << " ops/sec" << std::endl;
    std::cout << std::endl;
    
    // Decrypt and verify
    fhe::Plaintext pt_add_result, pt_mul_result;
    ctx.decrypt(pt_add_result, ct_add_batch, sk);
    ctx.decrypt(pt_mul_result, ct_mul_batch, sk);
    
    std::vector<uint64_t> add_result, mul_result;
    encoder.decode(add_result, pt_add_result);
    encoder.decode(mul_result, pt_mul_result);
    
    std::cout << "Addition results (first 10): ";
    for (int i = 0; i < 10; i++) {
        std::cout << add_result[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Multiplication results (first 10): ";
    for (int i = 0; i < 10; i++) {
        std::cout << mul_result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // ===== PROCESSING MULTIPLE CIPHERTEXTS =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 3: Processing Multiple Ciphertexts" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    const int num_ciphertexts = 10;
    std::vector<fhe::Ciphertext> ciphertexts(num_ciphertexts);
    
    std::cout << "Encrypting " << num_ciphertexts << " ciphertexts..." << std::endl;
    timer.reset();
    
    for (int i = 0; i < num_ciphertexts; i++) {
        std::vector<uint64_t> data(slot_count, i + 1);
        fhe::Plaintext pt;
        encoder.encode(pt, data);
        ctx.encrypt(ciphertexts[i], pt, pk);
    }
    
    double multi_encrypt_time = timer.elapsed_ms();
    std::cout << "  Total time: " << multi_encrypt_time << " ms" << std::endl;
    std::cout << "  Time per ciphertext: " << (multi_encrypt_time / num_ciphertexts) << " ms" << std::endl;
    std::cout << "  Total values encrypted: " << (num_ciphertexts * slot_count) << std::endl;
    std::cout << std::endl;
    
    // Compute sum of all ciphertexts
    std::cout << "Computing sum of all " << num_ciphertexts << " ciphertexts..." << std::endl;
    timer.reset();
    
    fhe::Ciphertext ct_total = ciphertexts[0];
    for (int i = 1; i < num_ciphertexts; i++) {
        fhe::Ciphertext ct_temp;
        ctx.add(ct_temp, ct_total, ciphertexts[i]);
        ct_total = ct_temp;
    }
    
    double sum_time = timer.elapsed_ms();
    std::cout << "  Sum time: " << sum_time << " ms" << std::endl;
    std::cout << std::endl;
    
    // Decrypt sum
    fhe::Plaintext pt_total;
    ctx.decrypt(pt_total, ct_total, sk);
    
    std::vector<uint64_t> total_result;
    encoder.decode(total_result, pt_total);
    
    // Expected: 1+2+3+...+10 = 55 in each slot
    std::cout << "Sum result (first 10 slots): ";
    for (int i = 0; i < 10; i++) {
        std::cout << total_result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "  Expected: All slots = 55" << std::endl;
    std::cout << std::endl;
    
    // ===== PERFORMANCE SUMMARY =====
    std::cout << "========================================" << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Single Ciphertext Operations:" << std::endl;
    std::cout << "  Encryption:    " << batch_encrypt_time << " ms" << std::endl;
    std::cout << "  Decryption:    " << batch_decrypt_time << " ms" << std::endl;
    std::cout << "  Addition:      " << add_time << " ms" << std::endl;
    std::cout << "  Multiplication: " << mul_time << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Batch Performance:" << std::endl;
    std::cout << "  Slots per ciphertext: " << slot_count << std::endl;
    std::cout << "  Encryption throughput: " << (slot_count / batch_encrypt_time * 1000.0) << " values/sec" << std::endl;
    std::cout << "  Addition throughput:   " << (slot_count / add_time * 1000.0) << " ops/sec" << std::endl;
    std::cout << "  Multiplication throughput: " << (slot_count / mul_time * 1000.0) << " ops/sec" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Multiple Ciphertexts:" << std::endl;
    std::cout << "  Processed: " << num_ciphertexts << " ciphertexts" << std::endl;
    std::cout << "  Total values: " << (num_ciphertexts * slot_count) << std::endl;
    std::cout << "  Average time per ciphertext: " << (multi_encrypt_time / num_ciphertexts) << " ms" << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Example Complete ===" << std::endl;
    
    return 0;
}
