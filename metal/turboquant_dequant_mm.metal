#include <metal_stdlib>
using namespace metal;

/// Fused dequantization and matrix-vector multiplication kernel for TurboQuant.
///
/// Performs y = W_dequant * x without materializing the full dequantized weight
/// matrix, significantly reducing memory bandwidth during inference. Each
/// threadgroup processes one (output_row, batch) pair by iterating over all
/// WHT blocks along the input dimension, dequantizing each block on the fly,
/// multiplying element-wise by the corresponding input activation, and
/// accumulating partial dot products via parallel reduction.
///
/// Shared-rotation mode (seed_residual == 0 && has_resid != 0): both the
/// primary and residual passes were quantized in the same rotated domain.
/// Their centroids are summed before the single inverse WHT butterfly,
/// eliminating the second butterfly pass and its threadgroup barrier overhead.
///
/// Simdgroup optimizations (Metal 2.1+, Apple Silicon):
///   - WHT butterfly stages with stride < 32 use simd_shuffle_xor instead of
///     threadgroup shared memory and barriers. For block_size=512 this replaces
///     5 of 9 barrier-synchronised stages with register-level shuffles.
///   - Dot-product reduction uses simd_sum for the intra-simdgroup phase,
///     falling back to shared memory only for cross-simdgroup accumulation.
///
/// Threadgroup layout: one threadgroup per (output_row, batch) pair.
///   grid.x  = out_features * block_size   (total threads in X)
///   grid.y  = batch_size                  (total threads in Y)
///   threads per threadgroup = block_size
///
/// Buffer layout:
///   buffer(0)  — packed_primary   : uint8   [out_features, in_features/2] (4-bit)
///                                            or [out_features, in_features]  (5-bit)
///   buffer(1)  — packed_residual  : uint8   [out_features, in_features/2] (always 4-bit)
///   buffer(2)  — codebook_primary : float   [16] for 4-bit or [32] for 5-bit
///   buffer(3)  — codebook_residual: float   [16]
///   buffer(4)  — norms            : float   [out_features]
///   buffer(5)  — params           : uint32  [7] {block_sz, out_feat, in_feat,
///                                                has_resid, seed_primary,
///                                                seed_residual, primary_bits}
///   buffer(6)  — x                : half    [batch_size, in_features]
///   buffer(7)  — y                : half    [batch_size, out_features]
///   buffer(8)  — lut              : float   [256] (precomputed primary×residual
///                                            centroid sums; only present when
///                                            shared_rot && primary_bits <= 4)

constant uint BLOCK_SIZE = 512;

/// Width of a simdgroup on all current Apple GPU architectures (M1–M5).
constant uint SIMD_WIDTH = 32;

// ---------------------------------------------------------------------------
// Inline WHT utilities (self-contained; no cross-file Metal linking required).
// ---------------------------------------------------------------------------

/// Deterministic sign scalar derived from the quantizer's WHT rotation seed.
/// Matches the MT19937-compatible hash used during offline quantization.
inline float tq_sign(uint seed, uint index) {
    uint h = seed * 2654435761u + index * 2246822519u;
    h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
    return (h & 1u) ? 1.0f : -1.0f;
}

/// In-place inverse Fast Walsh-Hadamard Transform over n elements in shared
/// memory, using simdgroup shuffles for intra-warp stages. For each butterfly
/// stage with stride h:
///   - h < SIMD_WIDTH (32): partners reside in the same simdgroup, so
///     simd_shuffle_xor exchanges values through the register file without
///     touching shared memory or issuing threadgroup barriers.
///   - h >= SIMD_WIDTH: partners may span different simdgroups, requiring
///     shared memory writes and threadgroup barriers.
///
/// For block_size=512 this eliminates 5 of 9 barriers (h=1,2,4,8,16).
inline void tq_fwht_shared(threadgroup float* buf, uint tid, uint n) {
    // Phase 1: simdgroup-local butterfly stages (no barriers required).
    // Load from shared memory once, then operate entirely in registers.
    float val = buf[tid];

    for (uint h = 1; h < SIMD_WIDTH && h < n; h <<= 1) {
        float partner = simd_shuffle_xor(val, static_cast<ushort>(h));
        // Standard FWHT butterfly: pair (j, j^h) where j & h == 0 is the
        // lower thread. Lower computes a+b, upper computes a-b.
        // Upper thread receives the lower's value as 'partner' and must
        // produce partner - val (i.e., a - b where a = partner, b = val).
        if ((tid & h) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }

    // Write register values back to shared memory for remaining stages
    buf[tid] = val;

    // Phase 2: cross-simdgroup butterfly stages via shared memory + barriers
    for (uint h = SIMD_WIDTH; h < n; h <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((tid % (2 * h)) < h) {
            float a = buf[tid];
            float b = buf[tid + h];
            buf[tid]     = a + b;
            buf[tid + h] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// Parallel reduction: sum n elements down to index 0, using simd_sum for
// the intra-simdgroup phase and shared memory for cross-simdgroup accumulation.
// ---------------------------------------------------------------------------

inline void parallel_reduce_sum(threadgroup float* buf, uint tid, uint n) {
    float val = buf[tid];

    // Intra-simdgroup reduction via hardware shuffle network
    val = simd_sum(val);

    if (n <= SIMD_WIDTH) {
        // All elements fit within one simdgroup — write final sum directly
        if (tid == 0) {
            buf[0] = val;
        }
    } else {
        // Barrier ensures all simdgroups have finished reading their buf[tid]
        // values before any simdgroup overwrites the partial-sum slots.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Cross-simdgroup accumulation: each simdgroup lane 0 writes its
        // partial sum, then thread 0 performs a serial reduction over the
        // (n / SIMD_WIDTH) partial sums.
        if (tid % SIMD_WIDTH == 0) {
            buf[tid / SIMD_WIDTH] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float sum = 0.0f;
            uint num_simdgroups = n / SIMD_WIDTH;
            for (uint s = 0; s < num_simdgroups; ++s) {
                sum += buf[s];
            }
            buf[0] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// Fused dequant-matmul kernel
// ---------------------------------------------------------------------------

kernel void tq_dequant_mm(
    device const uint8_t*  packed_primary    [[buffer(0)]],
    device const uint8_t*  packed_residual   [[buffer(1)]],
    device const float*    codebook_primary  [[buffer(2)]],
    device const float*    codebook_residual [[buffer(3)]],
    device const float*    norms             [[buffer(4)]],
    device const uint32_t* params            [[buffer(5)]],
    device const half*     x                 [[buffer(6)]],
    device half*           y                 [[buffer(7)]],
    device const float*    lut              [[buffer(8)]],
    uint  tid [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    // Decode runtime parameters — layout must match the C++ dispatch site
    // in fused_dequant_matmul() (src/dequantizer.cpp):
    //   params[0] = block_size
    //   params[1] = out_features
    //   params[2] = in_features
    //   params[3] = has_residual   (0 or 1)
    //   params[4] = seed_primary
    //   params[5] = seed_residual  (0 signals shared-rotation mode)
    //   params[6] = primary_bits   (4 or 5)
    const uint block_sz      = params[0];
    const uint out_feat      = params[1];
    const uint in_feat       = params[2];
    const uint has_resid     = params[3];
    const uint seed_primary  = params[4];
    const uint seed_residual = params[5];
    const uint primary_bits  = params[6];
    const uint packed_cols   = in_feat / 2;
    const uint num_blocks    = in_feat / block_sz;

    // Shared-rotation mode: seed_residual == 0 means both passes share the
    // primary WHT domain. Sum centroids before the butterfly for a single WHT.
    const uint shared_rot = (seed_residual == 0u && has_resid != 0u) ? 1u : 0u;

    // LUT mode: shared-rotation with 4-bit primary enables the precomputed
    // 256-entry lookup table, replacing two codebook reads and an addition
    // with a single threadgroup memory read per element.
    const uint use_lut = (shared_rot != 0u && primary_bits <= 4u) ? 1u : 0u;

    const uint out_row   = gid.x;
    const uint batch_idx = gid.y;

    // Scale factor applied after each WHT: 1/sqrt(in_feat) * 1/sqrt(block_sz)
    const float combined_scale = rsqrt(float(in_feat)) * rsqrt(float(block_sz));

    threadgroup float shared_buf[BLOCK_SIZE];

    // Load the 256-entry centroid sum LUT into threadgroup shared memory.
    // When the threadgroup has fewer than 256 threads (block_sz < 256), each
    // thread loads multiple entries in a strided loop to cover all 256 slots.
    // Subsequent accesses hit threadgroup SRAM with single-cycle latency
    // instead of traversing the device memory hierarchy for each codebook lookup.
    threadgroup float shared_lut[256];
    if (use_lut != 0u) {
        for (uint i = tid; i < 256u; i += block_sz) {
            shared_lut[i] = lut[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float dot_accum = 0.0f;

    for (uint b = 0; b < num_blocks; ++b) {
        // ------ Primary centroid lookup (and optional residual pre-sum) ------

        if (tid < block_sz) {
            const uint col = b * block_sz + tid;

            float val;
            if (use_lut != 0u) {
                // LUT path: combine primary and residual nibbles into a single
                // 8-bit index and read the precomputed centroid sum directly.
                // Eliminates two separate codebook reads and the float addition.
                const uint packed_offset = out_row * packed_cols + col / 2;
                const uint8_t p_byte = packed_primary[packed_offset];
                const uint8_t r_byte = packed_residual[packed_offset];

                uint combined;
                if (col % 2 == 0) {
                    combined = (p_byte & 0xF) | ((r_byte & 0xF) << 4);
                } else {
                    combined = ((p_byte >> 4) & 0xF) | (r_byte & 0xF0);
                }
                val = shared_lut[combined];
            } else {
                // Standard codebook lookup path for 5-bit primary or non-LUT modes
                uint primary_idx;
                if (primary_bits <= 4u) {
                    const uint packed_offset = out_row * packed_cols + col / 2;
                    const uint nibble_shift  = (col % 2 == 0) ? 0 : 4;
                    primary_idx = (packed_primary[packed_offset] >> nibble_shift) & 0xF;
                } else {
                    primary_idx = packed_primary[out_row * in_feat + col];
                }
                val = codebook_primary[primary_idx];

                // In shared-rotation mode the residual was quantized in the same
                // WHT domain as the primary, so sum centroids here and apply a
                // single inverse WHT instead of two independent transforms.
                if (shared_rot != 0u) {
                    const uint packed_offset = out_row * packed_cols + col / 2;
                    const uint nibble_shift  = (col % 2 == 0) ? 0 : 4;
                    const uint residual_idx  = (packed_residual[packed_offset] >> nibble_shift) & 0xF;
                    val += codebook_residual[residual_idx];
                }
            }

            shared_buf[tid] = val;
        }

        // Inverse WHT over the (possibly combined) centroid values
        tq_fwht_shared(shared_buf, tid, block_sz);

        float w_primary = 0.0f;
        if (tid < block_sz) {
            const float sign_p = tq_sign(seed_primary, tid);
            w_primary = shared_buf[tid] * combined_scale * sign_p;
        }

        // ------ Residual WHT pass (legacy dual-rotation models only) ------
        // Skipped entirely when shared_rot is active, eliminating the second
        // butterfly pass and its associated threadgroup barrier overhead.
        float w_residual = 0.0f;
        if (has_resid != 0u && shared_rot == 0u) {
            if (tid < block_sz) {
                const uint col           = b * block_sz + tid;
                const uint packed_offset = out_row * packed_cols + col / 2;
                const uint nibble_shift  = (col % 2 == 0) ? 0 : 4;
                const uint residual_idx  = (packed_residual[packed_offset] >> nibble_shift) & 0xF;
                shared_buf[tid] = codebook_residual[residual_idx];
            }

            tq_fwht_shared(shared_buf, tid, block_sz);

            if (tid < block_sz) {
                const float sign_r = tq_sign(seed_residual, tid);
                w_residual = shared_buf[tid] * combined_scale * sign_r;
            }
        }

        // ------ Multiply by activation and accumulate via reduction ------
        if (tid < block_sz) {
            const uint  col   = b * block_sz + tid;
            const float x_val = float(x[batch_idx * in_feat + col]);
            shared_buf[tid] = (w_primary + w_residual) * x_val;
        }

        parallel_reduce_sum(shared_buf, tid, block_sz);

        if (tid == 0) {
            dot_accum += shared_buf[0];
        }
    }

    // Apply the corrected row norm and write the output activation.
    if (tid == 0) {
        const float result = dot_accum * norms[out_row];
        y[batch_idx * out_feat + out_row] = half(result);
    }
}
