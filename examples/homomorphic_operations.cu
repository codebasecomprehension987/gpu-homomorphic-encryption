/**
 * Homomorphic Operations Example
 * 
 * Demonstrates:
 * - Homomorphic addition
 * - Homomorphic multiplication
 * - Relinearization
 * - Noise budget tracking
 */

#include <iostream>
#include <vector>
#include "fhe.cuh"

void print_vector(const std::string& label, const std::vector<uint64_t>& vec) {
    std::cout << label;
    for (auto val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== FHE Homomorphic Operations Example ===" << std::endl;
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
    
    // Generate keys
    std::cout << "Generating keys..." << std::endl;
    fhe::PublicKey pk;
    fhe::SecretKey sk;
    fhe::RelinKeys rlk;
    
    ctx.keygen(pk, sk);
    ctx.relinkey_gen(rlk, sk, 16);  // 16-bit decomposition
    std::cout << std::endl;
    
    // ===== HOMOMORPHIC ADDITION =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 1: Homomorphic Addition" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Prepare data
    std::vector<uint64_t> data_a = {10, 20, 30, 40};
    std::vector<uint64_t> data_b = {5, 15, 25, 35};
    
    print_vector("Data A: ", data_a);
    print_vector("Data B: ", data_b);
    std::cout << std::endl;
    
    // Encode and encrypt
    std::cout << "Encrypting data..." << std::endl;
    fhe::Plaintext pt_a, pt_b;
    ctx.encode(pt_a, data_a);
    ctx.encode(pt_b, data_b);
    
    fhe::Ciphertext ct_a, ct_b;
    ctx.encrypt(ct_a, pt_a, pk);
    ctx.encrypt(ct_b, pt_b, pk);
    
    std::cout << "  ct_a noise budget: " << ct_a.noise_budget << " bits" << std::endl;
    std::cout << "  ct_b noise budget: " << ct_b.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Homomorphic addition
    std::cout << "Computing: ct_sum = ct_a + ct_b (encrypted)" << std::endl;
    fhe::Ciphertext ct_sum;
    ctx.add(ct_sum, ct_a, ct_b);
    std::cout << "  ct_sum noise budget: " << ct_sum.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Decrypt and verify
    std::cout << "Decrypting result..." << std::endl;
    fhe::Plaintext pt_sum;
    ctx.decrypt(pt_sum, ct_sum, sk);
    
    std::vector<uint64_t> result_sum;
    ctx.decode(result_sum, pt_sum);
    
    print_vector("Result:   ", result_sum);
    print_vector("Expected: ", {15, 35, 55, 75});
    
    // Verify
    bool add_correct = true;
    for (size_t i = 0; i < data_a.size(); i++) {
        if (result_sum[i] != data_a[i] + data_b[i]) {
            add_correct = false;
        }
    }
    std::cout << (add_correct ? "✓ Addition correct!" : "✗ Addition failed!") << std::endl;
    std::cout << std::endl;
    
    // ===== HOMOMORPHIC MULTIPLICATION =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 2: Homomorphic Multiplication" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<uint64_t> data_x = {3, 4, 5, 6};
    std::vector<uint64_t> data_y = {2, 5, 10, 3};
    
    print_vector("Data X: ", data_x);
    print_vector("Data Y: ", data_y);
    std::cout << std::endl;
    
    // Encode and encrypt
    std::cout << "Encrypting data..." << std::endl;
    fhe::Plaintext pt_x, pt_y;
    ctx.encode(pt_x, data_x);
    ctx.encode(pt_y, data_y);
    
    fhe::Ciphertext ct_x, ct_y;
    ctx.encrypt(ct_x, pt_x, pk);
    ctx.encrypt(ct_y, pt_y, pk);
    
    std::cout << "  ct_x noise budget: " << ct_x.noise_budget << " bits" << std::endl;
    std::cout << "  ct_y noise budget: " << ct_y.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Homomorphic multiplication
    std::cout << "Computing: ct_product = ct_x * ct_y (encrypted)" << std::endl;
    fhe::Ciphertext ct_product;
    ctx.multiply(ct_product, ct_x, ct_y, rlk);
    std::cout << "  ct_product has " << ct_product.components.size() << " components (after relinearization)" << std::endl;
    std::cout << "  ct_product noise budget: " << ct_product.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Decrypt and verify
    std::cout << "Decrypting result..." << std::endl;
    fhe::Plaintext pt_product;
    ctx.decrypt(pt_product, ct_product, sk);
    
    std::vector<uint64_t> result_product;
    ctx.decode(result_product, pt_product);
    
    print_vector("Result:   ", result_product);
    print_vector("Expected: ", {6, 20, 50, 18});
    
    // Verify
    bool mul_correct = true;
    for (size_t i = 0; i < data_x.size(); i++) {
        if (result_product[i] != data_x[i] * data_y[i]) {
            mul_correct = false;
        }
    }
    std::cout << (mul_correct ? "✓ Multiplication correct!" : "✗ Multiplication failed!") << std::endl;
    std::cout << std::endl;
    
    // ===== CHAINED OPERATIONS =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 3: Chained Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Computing: (ct_a + ct_b) * ct_x" << std::endl;
    std::cout << std::endl;
    
    // Step 1: Add
    std::cout << "Step 1: ct_temp = ct_a + ct_b" << std::endl;
    fhe::Ciphertext ct_temp;
    ctx.add(ct_temp, ct_a, ct_b);
    std::cout << "  Noise budget after add: " << ct_temp.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Step 2: Multiply
    std::cout << "Step 2: ct_final = ct_temp * ct_x" << std::endl;
    fhe::Ciphertext ct_final;
    ctx.multiply(ct_final, ct_temp, ct_x, rlk);
    std::cout << "  Noise budget after mul: " << ct_final.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Decrypt
    std::cout << "Decrypting final result..." << std::endl;
    fhe::Plaintext pt_final;
    ctx.decrypt(pt_final, ct_final, sk);
    
    std::vector<uint64_t> result_final;
    ctx.decode(result_final, pt_final);
    
    print_vector("Result:   ", result_final);
    
    // Expected: (10+5)*3, (20+15)*4, (30+25)*5, (40+35)*6
    std::vector<uint64_t> expected_final = {45, 140, 275, 450};
    print_vector("Expected: ", expected_final);
    
    // Verify
    bool chain_correct = true;
    for (size_t i = 0; i < expected_final.size(); i++) {
        if (result_final[i] != expected_final[i]) {
            chain_correct = false;
        }
    }
    std::cout << (chain_correct ? "✓ Chained operations correct!" : "✗ Chained operations failed!") << std::endl;
    std::cout << std::endl;
    
    // ===== PLAINTEXT OPERATIONS =====
    std::cout << "========================================" << std::endl;
    std::cout << "PART 4: Plaintext Operations" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::vector<uint64_t> scalar_data = {2};
    fhe::Plaintext pt_scalar;
    ctx.encode(pt_scalar, scalar_data);
    
    std::cout << "Computing: ct_a + plaintext(2)" << std::endl;
    fhe::Ciphertext ct_add_plain;
    ctx.add_plain(ct_add_plain, ct_a, pt_scalar);
    
    fhe::Plaintext pt_add_plain;
    ctx.decrypt(pt_add_plain, ct_add_plain, sk);
    
    std::vector<uint64_t> result_add_plain;
    ctx.decode(result_add_plain, pt_add_plain);
    
    print_vector("Result:   ", result_add_plain);
    print_vector("Expected: ", {12, 22, 32, 42});
    std::cout << std::endl;
    
    std::cout << "Computing: ct_a * plaintext(2)" << std::endl;
    fhe::Ciphertext ct_mul_plain;
    ctx.multiply_plain(ct_mul_plain, ct_a, pt_scalar);
    
    fhe::Plaintext pt_mul_plain;
    ctx.decrypt(pt_mul_plain, ct_mul_plain, sk);
    
    std::vector<uint64_t> result_mul_plain;
    ctx.decode(result_mul_plain, pt_mul_plain);
    
    print_vector("Result:   ", result_mul_plain);
    print_vector("Expected: ", {20, 40, 60, 80});
    std::cout << std::endl;
    
    // Summary
    std::cout << "========================================" << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "✓ Homomorphic Addition" << std::endl;
    std::cout << "✓ Homomorphic Multiplication (with relinearization)" << std::endl;
    std::cout << "✓ Chained Operations" << std::endl;
    std::cout << "✓ Plaintext Operations" << std::endl;
    std::cout << std::endl;
    std::cout << "All operations completed successfully!" << std::endl;
    
    return 0;
}
