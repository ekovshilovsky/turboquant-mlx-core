#include <metal_stdlib>
using namespace metal;

/// Standalone dequantization kernel for TurboQuant-compressed weight matrices.
///
/// Reconstructs a full-precision weight matrix from packed 4-bit indices by
/// performing codebook lookup, inverse Walsh-Hadamard rotation, and row-norm
/// scaling entirely on the GPU. Used when the dequantized matrix is needed
/// explicitly (e.g., for debugging or non-fused inference paths).
///
/// Threadgroup layout: one threadgroup per (row, block) pair.
///   grid.x  = number of WHT blocks per row (in_features / block_size)
///   grid.y  = out_features (one row per y-index)
///   threads per threadgroup = block_size
///
/// Buffer layout:
///   buffer(0)  — packed_primary   : uint8  [out_features, in_features/2]
///   buffer(1)  — packed_residual  : uint8  [out_features, in_features/2]
///   buffer(2)  — codebook_primary : float  [16]
///   buffer(3)  — codebook_residual: float  [16]
///   buffer(4)  — norms            : float  [out_features]
///   buffer(5)  — params           : uint32 [5] {seed_primary, seed_residual,
///                                               block_size, out_features, in_features}
///   buffer(6)  — output           : float  [out_features, in_features]

constant uint BLOCK_SIZE = 512;

// ---------------------------------------------------------------------------
// Inline WHT utilities (duplicated here so the kernel is self-contained and
// does not depend on cross-file linking in the Metal shader compiler).
// ---------------------------------------------------------------------------

/// Deterministic sign generation matching the CPU reference hash.
inline float tq_sign(uint seed, uint index) {
    uint h = seed * 2654435761u + index * 2246822519u;
    return (h & 1u) ? 1.0f : -1.0f;
}

/// In-place Fast Walsh-Hadamard Transform on threadgroup shared memory.
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

// ---------------------------------------------------------------------------
// Dequantization kernel
// ---------------------------------------------------------------------------

kernel void tq_dequant(
    device const uint8_t*  packed_primary    [[buffer(0)]],
    device const uint8_t*  packed_residual   [[buffer(1)]],
    device const float*    codebook_primary  [[buffer(2)]],
    device const float*    codebook_residual [[buffer(3)]],
    device const float*    norms             [[buffer(4)]],
    device const uint32_t* params            [[buffer(5)]],
    device float*          output            [[buffer(6)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    // Unpack runtime parameters from the params buffer.
    const uint seed_primary   = params[0];
    // params[1] = seed_residual (reserved for per-stage rotation seeds)
    const uint block_size     = params[2];
    // params[3] = out_features (used implicitly via grid dims)
    const uint in_features    = params[4];

    // Early exit for threads beyond the actual block size. Allows launching
    // with BLOCK_SIZE threads even when the real block_size is smaller.
    if (tid >= block_size) return;

    const uint row        = gid.y;
    const uint block_idx  = gid.x;
    const uint packed_cols = in_features / 2;

    // Global column index for this thread within the weight matrix.
    const uint col = block_idx * block_size + tid;
    if (col >= in_features) return;

    // Byte offset into the packed array and nibble position within that byte.
    const uint packed_offset = row * packed_cols + col / 2;
    const uint nibble_shift  = (col % 2 == 0) ? 0 : 4;

    // Unpack the 4-bit primary index and look up its centroid.
    const uint primary_idx = (packed_primary[packed_offset] >> nibble_shift) & 0xF;
    float val = codebook_primary[primary_idx];

    // Unpack the 4-bit residual index and add its centroid contribution.
    const uint residual_idx = (packed_residual[packed_offset] >> nibble_shift) & 0xF;
    val += codebook_residual[residual_idx];

    // Store the combined centroid value into threadgroup shared memory for the
    // inverse WHT butterfly transform.
    threadgroup float shared_buf[BLOCK_SIZE];
    shared_buf[tid] = val;

    // Inverse WHT: apply FWHT then scale by 1/sqrt(block_size) and remove the
    // random sign flip that was applied during forward rotation.
    tq_fwht_shared(shared_buf, tid, block_size);

    const float inv_sqrt = rsqrt(float(block_size));
    const float sign_val = tq_sign(seed_primary, tid);
    float dequantized = shared_buf[tid] * inv_sqrt * sign_val;

    // Scale by the corrected row norm to restore the original weight magnitude.
    dequantized *= norms[row];

    // Write the reconstructed weight element to the output matrix.
    output[row * in_features + col] = dequantized;
}
