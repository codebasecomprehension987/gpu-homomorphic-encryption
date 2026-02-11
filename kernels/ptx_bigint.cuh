#pragma once
#include <cstdint>

namespace fhe {
namespace ptx {

// 128-bit addition with carry using PTX
__device__ __forceinline__ 
void add_u128(uint64_t a_lo, uint64_t a_hi, 
              uint64_t b_lo, uint64_t b_hi,
              uint64_t& c_lo, uint64_t& c_hi) {
    asm volatile(
        "add.cc.u64 %0, %2, %4;\n\t"
        "addc.u64 %1, %3, %5;"
        : "=l"(c_lo), "=l"(c_hi)
        : "l"(a_lo), "l"(a_hi), "l"(b_lo), "l"(b_hi)
    );
}

// 128-bit subtraction with borrow
__device__ __forceinline__
void sub_u128(uint64_t a_lo, uint64_t a_hi,
              uint64_t b_lo, uint64_t b_hi,
              uint64_t& c_lo, uint64_t& c_hi) {
    asm volatile(
        "sub.cc.u64 %0, %2, %4;\n\t"
        "subc.u64 %1, %3, %5;"
        : "=l"(c_lo), "=l"(c_hi)
        : "l"(a_lo), "l"(a_hi), "l"(b_lo), "l"(b_hi)
    );
}

// 64-bit multiplication producing 128-bit result
__device__ __forceinline__
void mul_u64_u128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    asm volatile(
        "mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(lo), "=l"(hi)
        : "l"(a), "l"(b)
    );
}

// Multi-precision multiplication accumulate
__device__ __forceinline__
void mad_lo_cc(uint64_t& d, uint64_t a, uint64_t b, uint64_t c) {
    asm volatile(
        "mad.lo.cc.u64 %0, %1, %2, %3;"
        : "=l"(d)
        : "l"(a), "l"(b), "l"(c)
    );
}

__device__ __forceinline__
void madc_hi(uint64_t& d, uint64_t a, uint64_t b, uint64_t c) {
    asm volatile(
        "madc.hi.u64 %0, %1, %2, %3;"
        : "=l"(d)
        : "l"(a), "l"(b), "l"(c)
    );
}

// Multi-precision addition chain
__device__ __forceinline__
void add_cc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "add.cc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

__device__ __forceinline__
void addc_cc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "addc.cc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

__device__ __forceinline__
void addc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "addc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

// Multi-precision subtraction chain
__device__ __forceinline__
void sub_cc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "sub.cc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

__device__ __forceinline__
void subc_cc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "subc.cc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

__device__ __forceinline__
void subc(uint64_t& d, uint64_t a, uint64_t b) {
    asm volatile(
        "subc.u64 %0, %1, %2;"
        : "=l"(d)
        : "l"(a), "l"(b)
    );
}

} // namespace ptx
} // namespace fhe
