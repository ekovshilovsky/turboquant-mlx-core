#include <metal_stdlib>
using namespace metal;

/// Online KV cache quantization kernel for TurboQuant.
///
/// Quantizes raw fp16 key or value vectors into packed 3-bit Lloyd-Max indices
/// with per-vector L2 norms. The full PolarQuant pipeline is performed per
/// vector: L2-normalize, apply randomized WHT rotation (sign flip + FWHT),
/// scale by sqrt(head_dim), and map each rotated coordinate to the nearest
/// 3-bit centroid via boundary comparison.
///
/// 10 indices are packed into each uint32 (30 bits used, 2 bits padding),
/// matching the storage format defined in LayerKVStorage.
///
/// Threadgroup layout:
///   One threadgroup per vector (per head, per token).
///   grid.x  = num_heads
///   grid.y  = num_tokens
///   threads_per_threadgroup = head_dim (must be power of two, max 1024)
///
/// Buffer layout:
///   buffer(0) — input      : half   [num_tokens, num_heads, head_dim]
///   buffer(1) — params     : uint32 [3] {head_dim, num_heads, wht_seed}
///   buffer(2) — packed_out : uint32 [num_tokens * num_heads * ceil(head_dim/10)]
///   buffer(3) — norms_out  : float  [num_tokens * num_heads]

// ---------------------------------------------------------------------------
// 3-bit Lloyd-Max decision boundaries for N(0,1) rotated coordinates.
// Midpoints between adjacent centroids, precomputed from Zandieh et al. (ICLR 2026).
// The quantize kernel only needs boundaries (not centroids) for index assignment.
// ---------------------------------------------------------------------------

constant float TQ3_BOUNDARIES[7] = {
    -1.748247f, -1.050224f, -0.500755f, 0.0f,
     0.500755f,  1.050224f,  1.748247f
};

// Maximum head dimension supported (determines shared memory allocation).
constant uint MAX_HEAD_DIM = 1024;

// ---------------------------------------------------------------------------
// Inline WHT utilities (self-contained, no cross-file Metal linking required).
// ---------------------------------------------------------------------------

inline float tq_sign(uint seed, uint index) {
    uint h = seed * 2654435761u + index * 2246822519u;
    return (h & 1u) ? 1.0f : -1.0f;
}

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
// Online quantization kernel
// ---------------------------------------------------------------------------

kernel void tq_online_quantize(
    device const half*     input      [[buffer(0)]],
    device const uint32_t* params     [[buffer(1)]],
    device uint32_t*       packed_out [[buffer(2)]],
    device float*          norms_out  [[buffer(3)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint head_dim  = params[0];
    const uint num_heads = params[1];
    const uint wht_seed  = params[2];

    const uint head_idx  = gid.x;
    const uint token_idx = gid.y;

    if (tid >= head_dim) return;

    // Shared memory for parallel reduction and WHT butterfly.
    threadgroup float shared_buf[MAX_HEAD_DIM];

    // ---- Step 1: Load input element and compute squared value for L2 norm ----

    const uint input_offset = token_idx * num_heads * head_dim
                            + head_idx * head_dim
                            + tid;
    float val = float(input[input_offset]);
    shared_buf[tid] = val * val;

    // ---- Step 2: Parallel reduction to compute sum of squares ----

    for (uint stride = head_dim / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float l2_norm = sqrt(shared_buf[0]);

    // Store the L2 norm (one thread per vector writes it).
    if (tid == 0) {
        norms_out[token_idx * num_heads + head_idx] = l2_norm;
    }

    // ---- Step 3: Normalize by the L2 norm ----

    // Guard against zero-norm vectors to avoid NaN propagation.
    float inv_norm = (l2_norm > 1e-12f) ? (1.0f / l2_norm) : 0.0f;
    float normalized = val * inv_norm;

    // ---- Step 4: Apply sign vector (randomized Hadamard rotation) ----

    float sign_val = tq_sign(wht_seed, tid);
    shared_buf[tid] = normalized * sign_val;

    // ---- Step 5: Fast Walsh-Hadamard Transform in shared memory ----

    tq_fwht_shared(shared_buf, tid, head_dim);

    // Scale by 1/sqrt(head_dim) from WHT normalization, then by sqrt(head_dim)
    // from the PolarQuant scaling, yielding a net unity factor. The WHT output
    // is used directly as the rotated coordinate.
    float rotated = shared_buf[tid];

    // ---- Step 6: Quantize to 3-bit index via boundary comparison ----

    uint index = 0;
    for (uint b = 0; b < 7; ++b) {
        if (rotated > TQ3_BOUNDARIES[b]) {
            index = b + 1;
        }
    }

    // ---- Step 7: Cooperative 3-bit packing (10 indices per uint32) ----
    //
    // Thread group of size head_dim cooperates to pack indices. Each group
    // of 10 consecutive threads writes one uint32. Thread tid belongs to
    // pack group (tid / 10) at position (tid % 10) within that group.
    //
    // Reuse shared memory to gather the 3-bit indices, then pack.

    // Store each thread's index into shared memory (reinterpret as uint).
    threadgroup uint shared_idx[MAX_HEAD_DIM];
    shared_idx[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each "packer" thread (one per group of 10) assembles the uint32.
    const uint pack_group = tid / 10;
    const uint pack_pos   = tid % 10;
    const uint words_per_vector = (head_dim + 9) / 10;

    if (pack_pos == 0 && pack_group < words_per_vector) {
        uint packed = 0;
        uint base = pack_group * 10;
        for (uint i = 0; i < 10 && (base + i) < head_dim; ++i) {
            packed |= (shared_idx[base + i] & 0x7u) << (i * 3);
        }

        uint out_offset = (token_idx * num_heads + head_idx) * words_per_vector
                        + pack_group;
        packed_out[out_offset] = packed;
    }
}
