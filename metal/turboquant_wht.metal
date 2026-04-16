#include <metal_stdlib>
using namespace metal;

/// Walsh-Hadamard butterfly transform utilities for TurboQuant Metal kernels.
///
/// Provides two device-level functions used by the dequantization and
/// fused dequant-matmul kernels:
///
///   tq_sign()         — Deterministic sign generation from a seed and index,
///                        producing +1.0 or -1.0 for the randomized Hadamard
///                        rotation described in the TurboQuant paper.
///
///   tq_fwht_shared()  — In-place Fast Walsh-Hadamard Transform operating on
///                        threadgroup shared memory. Uses log2(n) butterfly
///                        stages with threadgroup barriers between stages.
///                        The transform is self-inverse up to a 1/n scale factor,
///                        so both forward and inverse WHT use this same function.

/// Generate a deterministic sign (+1 or -1) for a given seed and element index.
/// Uses multiplicative hashing with Knuth constants to produce a pseudorandom
/// bit from the low bit of the hash result. The same (seed, index) pair always
/// produces the same sign, enabling reproducible rotation across CPU and GPU paths.
inline float tq_sign(uint seed, uint index) {
    uint h = seed * 2654435761u + index * 2246822519u;
    return (h & 1u) ? 1.0f : -1.0f;
}

/// In-place Fast Walsh-Hadamard Transform on threadgroup shared memory.
/// Each thread is responsible for one element identified by `tid`. The
/// transform executes log2(n) butterfly stages, with a threadgroup barrier
/// after each stage to ensure all reads and writes within that stage are
/// visible before the next stage begins.
///
/// Parameters:
///   buf — pointer to threadgroup-shared float buffer of length >= n
///   tid — thread index within the threadgroup (0..n-1)
///   n   — transform length, must be a power of two
inline void tq_fwht_shared(threadgroup float* buf, uint tid, uint n) {
    for (uint h = 1; h < n; h <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid % (2 * h)) < h) {
            float a = buf[tid];
            float b = buf[tid + h];
            buf[tid] = a + b;
            buf[tid + h] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
