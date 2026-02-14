/**
 * Basic Encryption Example
 * 
 * Demonstrates:
 * - Key generation
 * - Encoding plaintext
 * - Encryption
 * - Decryption
 * - Decoding
 */

#include <iostream>
#include <vector>
#include "fhe.cuh"

int main() {
    std::cout << "=== FHE Basic Encryption Example ===" << std::endl;
    std::cout << std::endl;
    
    // Step 1: Setup security parameters
    std::cout << "1. Setting up parameters..." << std::endl;
    fhe::SecurityParams params;
    params.lambda = 128;           // 128-bit security
    params.poly_degree = 4096;     // Polynomial degree
    params.log_q = 120;            // Ciphertext modulus (120 bits)
    params.sigma = 3.2;            // Noise standard deviation
    params.hamming_weight = 64;    // Secret key weight
    
    // Create FHE context
    fhe::FHEContext ctx(params);
    std::cout << "   Security level: " << params.lambda << " bits" << std::endl;
    std::cout << "   Polynomial degree: " << params.poly_degree << std::endl;
    std::cout << std::endl;
    
    // Step 2: Generate keys
    std::cout << "2. Generating keys..." << std::endl;
    fhe::PublicKey pk;
    fhe::SecretKey sk;
    
    ctx.keygen(pk, sk);
    std::cout << "   Keys generated successfully!" << std::endl;
    std::cout << std::endl;
    
    // Step 3: Prepare data
    std::cout << "3. Preparing plaintext data..." << std::endl;
    std::vector<uint64_t> plaintext_data = {42, 100, 255, 1337};
    
    std::cout << "   Original data: ";
    for (auto val : plaintext_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Step 4: Encode plaintext
    std::cout << "4. Encoding plaintext..." << std::endl;
    fhe::Plaintext pt;
    ctx.encode(pt, plaintext_data);
    std::cout << "   Data encoded into polynomial" << std::endl;
    std::cout << std::endl;
    
    // Step 5: Encrypt
    std::cout << "5. Encrypting..." << std::endl;
    fhe::Ciphertext ct;
    ctx.encrypt(ct, pt, pk);
    std::cout << "   Data encrypted successfully!" << std::endl;
    std::cout << "   Ciphertext has " << ct.components.size() << " components" << std::endl;
    std::cout << "   Initial noise budget: " << ct.noise_budget << " bits" << std::endl;
    std::cout << std::endl;
    
    // Step 6: Decrypt
    std::cout << "6. Decrypting..." << std::endl;
    fhe::Plaintext pt_result;
    ctx.decrypt(pt_result, ct, sk);
    std::cout << "   Data decrypted successfully!" << std::endl;
    std::cout << std::endl;
    
    // Step 7: Decode result
    std::cout << "7. Decoding result..." << std::endl;
    std::vector<uint64_t> decrypted_data;
    ctx.decode(decrypted_data, pt_result);
    
    std::cout << "   Decrypted data: ";
    for (size_t i = 0; i < plaintext_data.size(); i++) {
        std::cout << decrypted_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Step 8: Verify correctness
    std::cout << "8. Verifying correctness..." << std::endl;
    bool correct = true;
    for (size_t i = 0; i < plaintext_data.size(); i++) {
        if (plaintext_data[i] != decrypted_data[i]) {
            correct = false;
            std::cout << "   Mismatch at index " << i << ": "
                      << plaintext_data[i] << " != " << decrypted_data[i] << std::endl;
        }
    }
    
    if (correct) {
        std::cout << "   ✓ All values match! Encryption/Decryption successful!" << std::endl;
    } else {
        std::cout << "   ✗ Verification failed!" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    
    std::cout << "=== Example Complete ===" << std::endl;
    
    return 0;
}
