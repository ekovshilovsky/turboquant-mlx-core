#include <metal_stdlib>
using namespace metal;

/// Fused TQ-compressed attention kernel for TurboQuant KV cache.
///
/// Performs scaled dot-product attention against compressed K and V vectors
/// without materializing the full dequantized KV cache. Exploits the
/// pre-rotated query identity to avoid per-key inverse WHT:
///
///   dot(Q, inv_WHT(K_quant)) = dot(WHT(signs * Q), codebook[K_indices])
///
/// Pipeline for each query vector (one threadgroup per query per head):
///   1. Pre-rotate query: compute WHT(signs * Q) in shared memory
///   2. Score each compressed K position:
///      - Unpack 3-bit indices, dot pre-rotated Q with codebook values
///      - Scale by K norm
///   3. Two-pass softmax over attention scores (find max, then exp-sum)
///   4. Value aggregation:
///      - Unpack 3-bit V indices, look up centroids
///      - Inverse WHT to recover V vector
///      - Scale by V norm and attention weight, accumulate output
///   5. Write final output vector
///
/// Threadgroup layout:
///   grid.x  = num_heads
///   grid.y  = num_queries (typically 1 during decode)
///   threads_per_threadgroup = head_dim (must be power of two, max 1024)
///
/// Buffer layout:
///   buffer(0) — queries      : half   [num_queries, num_heads, head_dim]
///   buffer(1) — packed_keys  : uint32 [num_kv_tokens * num_heads * words_per_vec]
///   buffer(2) — packed_vals  : uint32 [num_kv_tokens * num_heads * words_per_vec]
///   buffer(3) — key_norms    : float  [num_kv_tokens * num_heads]
///   buffer(4) — val_norms    : float  [num_kv_tokens * num_heads]
///   buffer(5) — params       : uint32 [6] {head_dim, num_heads, num_kv_tokens,
///                                          wht_seed_k, wht_seed_v, kv_start}
///   buffer(6) — output       : half   [num_queries, num_heads, head_dim]

// ---------------------------------------------------------------------------
// 3-bit Lloyd-Max codebook for N(0,1) rotated coordinates.
// ---------------------------------------------------------------------------

constant float TQ3_CENTROIDS[8] = {
    -2.152176f, -1.344318f, -0.756130f, -0.245379f,
     0.245379f,  0.756130f,  1.344318f,  2.152176f
};

constant uint MAX_HEAD_DIM  = 1024;
constant uint MAX_KV_TOKENS = 4096;  // Max tokens processed per kernel dispatch

// ---------------------------------------------------------------------------
// Inline WHT utilities (self-contained for Metal shader compilation).
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

inline void parallel_reduce_sum(threadgroup float* buf, uint tid, uint n) {
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            buf[tid] += buf[tid + stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void parallel_reduce_max(threadgroup float* buf, uint tid, uint n) {
    for (uint stride = n / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            buf[tid] = max(buf[tid], buf[tid + stride]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---------------------------------------------------------------------------
// Unpack a 3-bit index from a packed uint32 word array.
// Words contain 10 indices each (30 bits used, 2 bits padding).
// ---------------------------------------------------------------------------

inline uint unpack_3bit(device const uint32_t* packed, uint base_word, uint element_idx) {
    uint word_idx   = element_idx / 10;
    uint bit_offset = (element_idx % 10) * 3;
    return (packed[base_word + word_idx] >> bit_offset) & 0x7u;
}

// ---------------------------------------------------------------------------
// Fused TQ compressed attention kernel
// ---------------------------------------------------------------------------

kernel void tq_compressed_attention(
    device const half*     queries     [[buffer(0)]],
    device const uint32_t* packed_keys [[buffer(1)]],
    device const uint32_t* packed_vals [[buffer(2)]],
    device const float*    key_norms   [[buffer(3)]],
    device const float*    val_norms   [[buffer(4)]],
    device const uint32_t* params      [[buffer(5)]],
    device half*           output      [[buffer(6)]],
    uint tid  [[thread_index_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    const uint head_dim      = params[0];
    const uint num_heads     = params[1];
    const uint num_kv_tokens = params[2];
    const uint wht_seed_k    = params[3];
    const uint wht_seed_v    = params[4];
    const uint kv_start      = params[5];  // Absolute token offset for indexing norms

    const uint head_idx  = gid.x;
    const uint query_idx = gid.y;

    if (tid >= head_dim) return;

    const uint words_per_vec = (head_dim + 9) / 10;

    // Attention scale factor: 1/sqrt(head_dim) for scaled dot-product attention.
    const float attn_scale = rsqrt(float(head_dim));

    // ========================================================================
    // Phase 1: Pre-rotate query vector — WHT(signs_k * Q)
    //
    // This enables the identity: dot(Q, inv_WHT(K)) = dot(WHT(signs*Q), codebook[K])
    // so we never need to inverse-WHT each compressed K vector.
    // ========================================================================

    threadgroup float shared_q[MAX_HEAD_DIM];

    // Load query element and apply K sign vector.
    const uint q_offset = query_idx * num_heads * head_dim
                        + head_idx * head_dim
                        + tid;
    float q_val = float(queries[q_offset]);
    shared_q[tid] = q_val * tq_sign(wht_seed_k, tid);

    // Forward WHT on the sign-flipped query.
    tq_fwht_shared(shared_q, tid, head_dim);

    // Each thread holds one element of the pre-rotated query.
    float prerot_q = shared_q[tid];

    // ========================================================================
    // Phase 2: Compute attention scores against all compressed K positions.
    //
    // For each K position, unpack 3-bit indices, look up centroids, and
    // accumulate the dot product with the pre-rotated query. Scale by K norm.
    // ========================================================================

    // Attention scores are computed cooperatively: each thread contributes its
    // dimension's product, then we reduce. We process keys one at a time to
    // minimize shared memory pressure.

    // Storage for attention scores. For large num_kv_tokens, the host dispatches
    // multiple kernel invocations with bounded ranges.
    threadgroup float scores[MAX_KV_TOKENS];

    for (uint k = 0; k < num_kv_tokens; ++k) {
        // Base word offset for this key vector's packed storage.
        uint k_base = ((kv_start + k) * num_heads + head_idx) * words_per_vec;

        // Unpack this thread's 3-bit index and look up the centroid.
        uint k_idx = unpack_3bit(packed_keys, k_base, tid);
        float k_centroid = TQ3_CENTROIDS[k_idx];

        // Partial dot product: pre-rotated query element * centroid.
        shared_q[tid] = prerot_q * k_centroid;

        // Parallel reduction to sum the dot product across all dimensions.
        parallel_reduce_sum(shared_q, tid, head_dim);

        if (tid == 0) {
            float k_norm = key_norms[(kv_start + k) * num_heads + head_idx];
            // The WHT output has an implicit 1/n factor that cancels with the
            // sqrt(head_dim) PolarQuant scale, leaving the raw dot product
            // scaled by the K norm only. Apply attention scaling.
            scores[k] = shared_q[0] * k_norm * attn_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ========================================================================
    // Phase 3: Softmax over attention scores (two-pass: max, then exp-sum).
    // ========================================================================

    // Find the maximum score for numerical stability.
    if (tid < num_kv_tokens) {
        shared_q[tid] = scores[tid];
    } else {
        shared_q[tid] = -INFINITY;
    }

    // Reduce over head_dim threads (padding with -INFINITY for unused lanes).
    parallel_reduce_max(shared_q, tid, head_dim);
    float max_score = shared_q[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exponentiate and compute the partition sum.
    if (tid < num_kv_tokens) {
        scores[tid] = exp(scores[tid] - max_score);
        shared_q[tid] = scores[tid];
    } else {
        shared_q[tid] = 0.0f;
    }

    parallel_reduce_sum(shared_q, tid, head_dim);
    float sum_exp = shared_q[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize scores to produce attention weights.
    float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
    if (tid < num_kv_tokens) {
        scores[tid] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ========================================================================
    // Phase 4: Value aggregation — weighted sum of dequantized V vectors.
    //
    // Unlike K (where pre-rotation avoids inverse WHT), V must be fully
    // dequantized: inverse WHT(codebook[V_indices]) * V_norm.
    // ========================================================================

    // Accumulate the output vector element-by-element across V positions.
    float output_accum = 0.0f;

    threadgroup float shared_v[MAX_HEAD_DIM];

    for (uint v = 0; v < num_kv_tokens; ++v) {
        float weight = scores[v];

        // Skip negligible weights to reduce computation.
        if (weight < 1e-8f) continue;

        // Unpack this thread's 3-bit V index and look up centroid.
        uint v_base = ((kv_start + v) * num_heads + head_idx) * words_per_vec;
        uint v_idx = unpack_3bit(packed_vals, v_base, tid);
        float v_centroid = TQ3_CENTROIDS[v_idx];

        // Inverse WHT to recover the rotated value vector.
        shared_v[tid] = v_centroid;
        tq_fwht_shared(shared_v, tid, head_dim);

        // Remove sign rotation and apply WHT normalization (1/n factor).
        float sign_val = tq_sign(wht_seed_v, tid);
        float v_elem = shared_v[tid] * sign_val / float(head_dim);

        // Scale by V norm and attention weight, accumulate into output.
        float v_norm = val_norms[(kv_start + v) * num_heads + head_idx];
        output_accum += weight * v_elem * v_norm;
    }

    // ========================================================================
    // Phase 5: Write output vector.
    // ========================================================================

    uint out_offset = query_idx * num_heads * head_dim
                    + head_idx * head_dim
                    + tid;
    output[out_offset] = half(output_accum);
}
