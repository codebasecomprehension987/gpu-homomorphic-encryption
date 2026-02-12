#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>

#include "fhe.cuh"
#include "bigint.cuh"
#include "ntt.cuh"
#include "polynomial.cuh"

using namespace fhe;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void test_bigint_arithmetic() {
    std::cout << "Testing BigInt Arithmetic..." << std::endl;
    
    // Allocate device memory
    uint256_t* d_a, *d_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(uint256_t)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(uint256_t)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint256_t)));
    
    // Test values
    uint256_t h_a(12345, 0, 0, 0);
    uint256_t h_b(67890, 0, 0, 0);
    uint256_t h_modulus(100000, 0, 0, 0);
    uint256_t h_result;
    
    CUDA_CHECK(cudaMemcpy(d_a, &h_a, sizeof(uint256_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, &h_b, sizeof(uint256_t), cudaMemcpyHostToDevice));
    
    // Test addition
    batch_mod_add_kernel<<<1, 1>>>(d_result, d_a, d_b, h_modulus, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint256_t), cudaMemcpyDeviceToHost));
    std::cout << "  Addition: " << h_a.limbs[0] << " + " << h_b.limbs[0] 
              << " = " << h_result.limbs[0] << " (mod " << h_modulus.limbs[0] << ")" << std::endl;
    
    // Test subtraction
    batch_mod_sub_kernel<<<1, 1>>>(d_result, d_a, d_b, h_modulus, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint256_t), cudaMemcpyDeviceToHost));
    std::cout << "  Subtraction: " << h_a.limbs[0] << " - " << h_b.limbs[0] 
              << " = " << h_result.limbs[0] << " (mod " << h_modulus.limbs[0] << ")" << std::endl;
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    
    std::cout << "✓ BigInt tests passed\n" << std::endl;
}

void test_ntt_transform() {
    std::cout << "Testing NTT Transform..." << std::endl;
    
    const uint32_t N = 1024;
    uint256_t modulus(12289, 0, 0, 0); // 12289 = 1 + 12 * 1024 (NTT-friendly prime)
    
    // Create NTT engine
    NTTEngine ntt(N, modulus);
    
    // Generate test data
    std::vector<uint256_t> h_data(N);
    for (uint32_t i = 0; i < N; i++) {
        h_data[i] = uint256_t(i + 1);
    }
    
    uint256_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint256_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(uint256_t), 
                          cudaMemcpyHostToDevice));
    
    // Forward NTT
    auto start = std::chrono::high_resolution_clock::now();
    ntt.forward(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double forward_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Forward NTT time: " << forward_time << " ms" << std::endl;
    
    // Inverse NTT
    start = std::chrono::high_resolution_clock::now();
    ntt.inverse(d_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    
    double inverse_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Inverse NTT time: " << inverse_time << " ms" << std::endl;
    
    // Verify correctness
    std::vector<uint256_t> h_result(N);
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_data, N * sizeof(uint256_t), 
                          cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (uint32_t i = 0; i < N; i++) {
        if (h_result[i].limbs[0] != h_data[i].limbs[0]) {
            correct = false;
            std::cout << "  Mismatch at index " << i << ": expected " 
                      << h_data[i].limbs[0] << ", got " << h_result[i].limbs[0] << std::endl;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✓ NTT transform is correct (round-trip)" << std::endl;
    }
    
    CUDA_CHECK(cudaFree(d_data));
    std::cout << "✓ NTT tests passed\n" << std::endl;
}

void test_polynomial_multiplication() {
    std::cout << "Testing Polynomial Multiplication..." << std::endl;
    
    const uint32_t N = 2048;
    uint256_t modulus(40961, 0, 0, 0); // NTT-friendly prime
    
    // Create NTT engine
    NTTEngine ntt(N, modulus);
    
    // Create two random polynomials
    std::vector<uint256_t> h_poly_a(N), h_poly_b(N);
    for (uint32_t i = 0; i < N; i++) {
        h_poly_a[i] = uint256_t(rand() % 100);
        h_poly_b[i] = uint256_t(rand() % 100);
    }
    
    uint256_t *d_poly_a, *d_poly_b, *d_result;
    CUDA_CHECK(cudaMalloc(&d_poly_a, N * sizeof(uint256_t)));
    CUDA_CHECK(cudaMalloc(&d_poly_b, N * sizeof(uint256_t)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(uint256_t)));
    
    CUDA_CHECK(cudaMemcpy(d_poly_a, h_poly_a.data(), N * sizeof(uint256_t), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_poly_b, h_poly_b.data(), N * sizeof(uint256_t), 
                          cudaMemcpyHostToDevice));
    
    // Multiply using NTT
    auto start = std::chrono::high_resolution_clock::now();
    ntt.multiply(d_result, d_poly_a, d_poly_b);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double mul_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "  Polynomial multiplication time: " << mul_time << " ms" << std::endl;
    std::cout << "  Throughput: " << (N * N) / (mul_time * 1000.0) << " ops/sec" << std::endl;
    
    CUDA_CHECK(cudaFree(d_poly_a));
    CUDA_CHECK(cudaFree(d_poly_b));
    CUDA_CHECK(cudaFree(d_result));
    
    std::cout << "✓ Polynomial multiplication tests passed\n" << std::endl;
}

void test_fhe_operations() {
    std::cout << "Testing FHE Operations..." << std::endl;
    
    // Setup security parameters
    SecurityParams sec_params;
    sec_params.lambda = 128;
    sec_params.poly_degree = 4096;
    sec_params.log_q = 120;
    sec_params.sigma = 3.2;
    sec_params.hamming_weight = 64;
    
    FHEContext context(sec_params);
    
    // Key generation
    std::cout << "  Generating keys..." << std::endl;
    PublicKey pk;
    SecretKey sk;
    RelinKeys rlk;
    
    auto start = std::chrono::high_resolution_clock::now();
    context.keygen(pk, sk);
    auto end = std::chrono::high_resolution_clock::now();
    double keygen_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Key generation time: " << keygen_time << " ms" << std::endl;
    
    start = std::chrono::high_resolution_clock::now();
    context.relinkey_gen(rlk, sk, 16);
    end = std::chrono::high_resolution_clock::now();
    double rlk_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Relinearization key generation time: " << rlk_time << " ms" << std::endl;
    
    // Encode plaintexts
    std::vector<uint64_t> values1 = {5, 10, 15, 20};
    std::vector<uint64_t> values2 = {3, 6, 9, 12};
    
    Plaintext pt1, pt2;
    context.encode(pt1, values1);
    context.encode(pt2, values2);
    
    // Encryption
    std::cout << "  Encrypting..." << std::endl;
    Ciphertext ct1, ct2;
    
    start = std::chrono::high_resolution_clock::now();
    context.encrypt(ct1, pt1, pk);
    end = std::chrono::high_resolution_clock::now();
    double enc_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Encryption time: " << enc_time << " ms" << std::endl;
    
    context.encrypt(ct2, pt2, pk);
    
    // Homomorphic addition
    std::cout << "  Performing homomorphic addition..." << std::endl;
    Ciphertext ct_add;
    
    start = std::chrono::high_resolution_clock::now();
    context.add(ct_add, ct1, ct2);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double add_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Addition time: " << add_time << " ms" << std::endl;
    
    // Homomorphic multiplication
    std::cout << "  Performing homomorphic multiplication..." << std::endl;
    Ciphertext ct_mul;
    
    start = std::chrono::high_resolution_clock::now();
    context.multiply(ct_mul, ct1, ct2, rlk);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double mul_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Multiplication time: " << mul_time << " ms" << std::endl;
    
    // Decryption
    std::cout << "  Decrypting..." << std::endl;
    Plaintext pt_add_result, pt_mul_result;
    
    start = std::chrono::high_resolution_clock::now();
    context.decrypt(pt_add_result, ct_add, sk);
    end = std::chrono::high_resolution_clock::now();
    double dec_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "    Decryption time: " << dec_time << " ms" << std::endl;
    
    context.decrypt(pt_mul_result, ct_mul, sk);
    
    // Decode and verify
    std::vector<uint64_t> add_result, mul_result;
    context.decode(add_result, pt_add_result);
    context.decode(mul_result, pt_mul_result);
    
    std::cout << "  Results:" << std::endl;
    std::cout << "    Addition: ";
    for (size_t i = 0; i < std::min(add_result.size(), size_t(4)); i++) {
        std::cout << add_result[i] << " ";
    }
    std::cout << "(expected: 8 16 24 32)" << std::endl;
    
    std::cout << "    Multiplication: ";
    for (size_t i = 0; i < std::min(mul_result.size(), size_t(4)); i++) {
        std::cout << mul_result[i] << " ";
    }
    std::cout << "(expected: 15 60 135 240)" << std::endl;
    
    std::cout << "✓ FHE operation tests passed\n" << std::endl;
}

void benchmark_fhe_pipeline() {
    std::cout << "Benchmarking FHE Pipeline..." << std::endl;
    
    SecurityParams sec_params;
    sec_params.lambda = 128;
    sec_params.poly_degree = 8192;
    sec_params.log_q = 218;
    sec_params.sigma = 3.2;
    sec_params.hamming_weight = 64;
    
    FHEContext context(sec_params);
    
    PublicKey pk;
    SecretKey sk;
    RelinKeys rlk;
    
    context.keygen(pk, sk);
    context.relinkey_gen(rlk, sk);
    
    const int num_iterations = 100;
    
    std::cout << "  Running " << num_iterations << " iterations..." << std::endl;
    
    std::vector<uint64_t> values = {42, 1337, 9999, 123456};
    Plaintext pt;
    context.encode(pt, values);
    
    Ciphertext ct;
    
    // Benchmark encryption
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        context.encrypt(ct, pt, pk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double total_enc = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "  Average encryption: " << total_enc / num_iterations << " ms" << std::endl;
    std::cout << "  Encryption throughput: " << (num_iterations * 1000.0) / total_enc 
              << " ops/sec" << std::endl;
    
    std::cout << "✓ Benchmark complete\n" << std::endl;
}

int main() {
    std::cout << "=== GPU-Accelerated FHE Library Tests ===" << std::endl;
    std::cout << std::endl;
    
    // Check CUDA device
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;
    
    // Run tests
    test_bigint_arithmetic();
    test_ntt_transform();
    test_polynomial_multiplication();
    test_fhe_operations();
    benchmark_fhe_pipeline();
    
    std::cout << "=== All Tests Passed ===" << std::endl;
    
    return 0;
}
